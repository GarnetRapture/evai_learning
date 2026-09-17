"""Choose correction supervision and spirit/language-matched conversational replay."""

import hashlib
import json
import math
import random
from collections import defaultdict
from typing import Any

from spirit_dataset.records import TrainingTask
from training.corpus import TrainingCorpus
from training.records import EncodedRecord


def training_signature(
    config: dict[str, Any], provenance: dict[str, Any], partition_moves: dict[str, int]
) -> str:
    """Identify the exact learning objective, data selection and optimizer schedule."""
    settings = dict(config["training"])
    for resource_setting in (
        "save_interval_seconds", "preparation_workers", "token_memory_limit_mib"
    ):
        settings.pop(resource_setting)
    material = {
        "objective": "assistant_example_mean_v1",
        "selection": "unconsumed_with_speech_replay_v2",
        "training": settings,
        "optimizer": config["optimizer"],
        # Manifest creation timestamps/profile-file hashes are not model inputs.
        # The split hashes cover source text, targets, ownership and event keys.
        "datasets": {slug: entry["splits_sha256"] for slug, entry in provenance.items()},
        "partition_moves": partition_moves,
    }
    return hashlib.sha256(json.dumps(material, sort_keys=True).encode("utf-8")).hexdigest()


def select_correction_curriculum(
    corpus: TrainingCorpus, curriculum: str, replay_ratio: float, rng: random.Random,
    consumed: set[str] | frozenset[str] = frozenset(),
) -> dict[str, int]:
    focused: dict[tuple[str, str], list[EncodedRecord]] = defaultdict(list)
    existing: dict[tuple[str, str], list[EncodedRecord]] = defaultdict(list)
    unique: dict[str, tuple[EncodedRecord, bool]] = {}
    for record in corpus.splits["train"]:
        if curriculum == "full":
            is_focus = True
        elif curriculum == "knowledge_completion":
            is_focus = record.promoted_knowledge
        elif curriculum == "dialogue_extension":
            is_focus = record.dialogue_extension
        elif curriculum == "dialogue_context":
            is_focus = record.dialogue_context
        else:
            is_focus = record.alignment_target
        previous = unique.get(record.fingerprint)
        if previous is None or is_focus:
            unique[record.fingerprint] = (record, is_focus)
    for record, is_focus in unique.values():
        destination = focused if is_focus and record.fingerprint not in consumed else existing
        destination[(record.spirit_id, record.language)].append(record)
    selected: list[EncodedRecord] = []
    replay_count = 0
    for owner in sorted(focused.keys() | existing.keys()):
        corrections = focused[owner]
        candidates = [
            record for record in existing[owner] if record.task is TrainingTask.PERSONA_SPEECH
        ]
        count = min(len(candidates), math.ceil(len(corrections) * replay_ratio))
        selected.extend(corrections)
        selected.extend(rng.sample(candidates, count))
        replay_count += count
    counts = {
        "available_train": len(corpus.splits["train"]),
        "correction_examples": len(selected) - replay_count,
        "replay_examples": replay_count,
        "selected_train": len(selected),
        "duplicate_examples_removed": len(corpus.splits["train"]) - len(unique),
        "previously_consumed_examples": sum(
            fingerprint in consumed for fingerprint in unique
        ),
    }
    corpus.splits["train"] = selected
    for slug, source in corpus.provenance.items():
        source["selected_train"] = sum(record.spirit_id == slug for record in selected)
    return counts
