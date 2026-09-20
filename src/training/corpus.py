"""Tokenize once into bounded int32 CPU storage; never cache tokens on disk."""

import hashlib
import json
import random
from array import array
from collections import Counter
from collections.abc import Iterator
from contextlib import closing
from dataclasses import dataclass, replace
from typing import Any

import torch

from common.errors import EvaiError
from common.hashing import compute_file_sha256
from common.model_contract import DATASET_VERSION, TRAINING_LANGUAGES
from common.paths import GENERAL_CORPUS_FILE
from general_corpus.records import training_records
from general_corpus.store import read_general_corpus
from sft_dataset.split import connected_indices
from sft_dataset.storage import sft_split_path
from spirit_dataset.curriculum import SpiritGrade, required_tasks
from spirit_dataset.records import DIALOGUE_LESSON_PREFIX, SourceClass, TrainingTask
from spirit_dataset.runtime_prompt import GENERAL_CORPUS_ID, spirit_file_path
from training.data import example_fingerprint
from training.records import IGNORE_INDEX, SHUFFLE_POOL_BATCHES, EncodedRecord
from training.tokenization import ParallelTokenizer

SPLITS = ("train", "validation", "test")
GENERAL_PROGRESS_INTERVAL = 500000


@dataclass
class TrainingCorpus:
    splits: dict[str, list[EncodedRecord]]
    provenance: dict[str, Any]
    excluded: dict[str, int]
    partition_moves: dict[str, int]
    token_ids: torch.Tensor
    max_length: int

    @classmethod
    def prepare(
        cls,
        slugs: list[str],
        tokenizer: Any,
        max_length: int,
        token_memory_limit_bytes: int,
        preparation_workers: int,
        general_corpus: bool,
    ) -> TrainingCorpus:
        with closing(ParallelTokenizer(tokenizer, preparation_workers)) as encoder:
            return cls._index(
                slugs, encoder, max_length, token_memory_limit_bytes, general_corpus
            )

    @classmethod
    def _index(
        cls,
        slugs: list[str],
        encoder: ParallelTokenizer,
        max_length: int,
        token_memory_limit_bytes: int,
        general_corpus: bool = False,
    ) -> TrainingCorpus:
        splits: dict[str, list[EncodedRecord]] = {split: [] for split in SPLITS}
        token_storage = array("i")
        provenance: dict[str, Any] = {}
        excluded: Counter[str] = Counter()
        locations: list[tuple[str, EncodedRecord]] = []
        keys_by_record: list[tuple[str, ...]] = []
        for number, slug in enumerate(slugs, 1):
            profile_path = spirit_file_path(slug)
            profile = json.loads(profile_path.read_text(encoding="utf-8"))
            manifest = profile["manifest"]
            if manifest["dataset_version"] != DATASET_VERSION:
                raise EvaiError(f"Stale canonical dataset for {slug}")
            provenance[slug] = {
                "profile_sha256": compute_file_sha256(profile_path),
                "manifest": manifest,
                "splits_sha256": {},
            }
            tasks: set[str] = set()
            languages: set[str] = set()
            for split in SPLITS:
                path = sft_split_path(slug, split)
                provenance[slug]["splits_sha256"][split] = compute_file_sha256(path)
                before = len(splits[split])
                for prepared in encoder.records(path, max_length, slug):
                    record, example = prepared.record, prepared.example
                    if example is None:
                        excluded[f"{slug}/{split}"] += 1
                        continue
                    targets = sum(label != IGNORE_INDEX for label in example.labels[1:])
                    if targets == 0:
                        raise EvaiError(f"Record has no supervised response: {example.record_id}")
                    required_bytes = (
                        len(token_storage) + len(example.input_ids)
                    ) * token_storage.itemsize
                    if required_bytes > token_memory_limit_bytes:
                        raise EvaiError(
                            "Encoded corpus exceeds the configured CPU token memory limit: "
                            f"{required_bytes} > {token_memory_limit_bytes} bytes"
                        )
                    location = EncodedRecord(
                        slug,
                        len(token_storage),
                        len(example.input_ids),
                        len(example.input_ids) - targets,
                        example.language,
                        record["source_class"] == SourceClass.DERIVED_SPEECH
                        or (
                            example.task == TrainingTask.SELF_MEMORY
                            and any(
                                "Hero.NameSno" in item["reference"]
                                for item in record.get("evidence", ())
                            )
                        ),
                        example.task == TrainingTask.SELF_MEMORY,
                        example.record_id.removeprefix(f"{example.language}:").startswith(
                            DIALOGUE_LESSON_PREFIX
                        ),
                        fingerprint=example_fingerprint(
                            example.input_ids, len(example.input_ids) - targets
                        ),
                        task=TrainingTask(example.task),
                        dialogue_context=(
                            record["source_class"] == SourceClass.CANON_DIALOGUE
                            and [turn["role"] for turn in example.prompt]
                            == ["system", "user", "assistant", "user"]
                        ),
                    )
                    token_storage.extend(example.input_ids)
                    splits[split].append(location)
                    locations.append((split, location))
                    keys = list(record.get("event_keys", ()))
                    keys.append(f"origin:{slug}:{record['id']}")
                    if record.get("origin_id"):
                        keys.append(f"origin:{slug}:{record['origin_id']}")
                    answer = hashlib.sha256(example.reference.encode("utf-8")).hexdigest()
                    keys.append(f"answer:{slug}:{answer}")
                    keys_by_record.append(tuple(keys))
                    if split == "train":
                        tasks.add(example.task)
                        languages.add(example.language)
                if len(splits[split]) == before:
                    raise EvaiError(f"No usable {split} records for {slug}")
            required = {task.value for task in required_tasks(SpiritGrade(manifest["grade_sno"]))}
            if not required <= tasks or languages != set(TRAINING_LANGUAGES):
                raise EvaiError(
                    f"Incomplete multilingual curriculum for {slug}: {tasks}, {languages}"
                )
            print(f"Indexed {number}/{len(slugs)}: {slug}", flush=True)
        # Preserve every existing training record. Move connected holdout records
        # with it rather than evaluating a shared event the model has already seen.
        # Local record IDs and identical speech are spirit-owned; world/story event
        # keys remain global so translations and cross-speaker scenes stay together.
        splits = {split: [] for split in SPLITS}
        moves: Counter[str] = Counter()
        for group in connected_indices(keys_by_record):
            prior_destination = min((locations[index][0] for index in group), key=SPLITS.index)
            contains_knowledge = any(locations[index][1].fixed_knowledge for index in group)
            destination = "train" if contains_knowledge else prior_destination
            for index in group:
                original, location = locations[index]
                if location.fixed_knowledge and prior_destination != "train":
                    location = replace(location, promoted_knowledge=True)
                splits[destination].append(location)
                if original != destination:
                    moves[f"{original}->{destination}"] += 1
        if general_corpus:
            general_counts = cls._index_general(
                encoder, max_length, token_memory_limit_bytes, token_storage, splits, excluded
            )
            provenance[GENERAL_CORPUS_ID] = {
                "splits_sha256": {"corpus": compute_file_sha256(GENERAL_CORPUS_FILE)},
                "records": general_counts,
            }
        if any(not records for records in splits.values()):
            raise EvaiError("Joint curriculum requires independent train/validation/test events")
        for slug, source in provenance.items():
            source["joint_splits"] = {
                split: sum(item.spirit_id == slug for item in records)
                for split, records in splits.items()
            }
        print(f"Joint event partitions: {dict(moves)}", flush=True)
        token_ids = torch.frombuffer(token_storage, dtype=torch.int32)
        print(
            f"CPU token storage: {token_ids.numel() * token_ids.element_size():,} bytes",
            flush=True,
        )
        return cls(splits, provenance, dict(excluded), dict(moves), token_ids, max_length)

    @staticmethod
    def _index_general(
        encoder: ParallelTokenizer,
        max_length: int,
        token_memory_limit_bytes: int,
        token_storage: array,
        splits: dict[str, list[EncodedRecord]],
        excluded: Counter[str],
    ) -> dict[str, int]:
        counts: Counter[str] = Counter()
        records = (
            record
            for conversation in read_general_corpus()
            for record in training_records(conversation)
        )
        for prepared in encoder.stream(records, max_length, GENERAL_CORPUS_ID):
            record, example = prepared.record, prepared.example
            split = str(record["split"])
            if example is None:
                excluded[f"{GENERAL_CORPUS_ID}/{split}"] += 1
                continue
            targets = sum(label != IGNORE_INDEX for label in example.labels[1:])
            if targets == 0:
                excluded[f"{GENERAL_CORPUS_ID}/{split}"] += 1
                continue
            required_bytes = (len(token_storage) + len(example.input_ids)) * token_storage.itemsize
            if required_bytes > token_memory_limit_bytes:
                raise EvaiError(
                    "Encoded corpus exceeds the configured CPU token memory limit: "
                    f"{required_bytes} > {token_memory_limit_bytes} bytes"
                )
            splits[split].append(
                EncodedRecord(
                    GENERAL_CORPUS_ID,
                    len(token_storage),
                    len(example.input_ids),
                    len(example.input_ids) - targets,
                    example.language,
                    False,
                    False,
                    False,
                    fingerprint=example_fingerprint(
                        example.input_ids, len(example.input_ids) - targets
                    ),
                    task=TrainingTask(example.task),
                )
            )
            token_storage.extend(example.input_ids)
            counts[f"{split}:{example.task}:{example.language}"] += 1
            total = sum(counts.values())
            if total % GENERAL_PROGRESS_INTERVAL == 0:
                print(f"Indexed general corpus records: {total:,}", flush=True)
        return dict(counts)

    def batches(
        self,
        split: str,
        batch_size: int,
        rng: random.Random | None = None,
    ) -> Iterator[list[EncodedRecord]]:
        ordered = list(self.splits[split])
        if rng is not None:
            rng.shuffle(ordered)
        pool_size = batch_size * SHUFFLE_POOL_BATCHES
        for start in range(0, len(ordered), pool_size):
            pool = sorted(ordered[start : start + pool_size], key=lambda item: item.token_count)
            batches = [pool[i : i + batch_size] for i in range(0, len(pool), batch_size)]
            if rng is not None:
                rng.shuffle(batches)
            yield from batches
