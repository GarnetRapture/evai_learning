import hashlib
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

from common.errors import EvaiError
from common.messages import SFTRecordTurn
from common.paths import SPIRIT_JUDGMENT_DIR
from sft_dataset.split import DatasetSplit, SplitConfig, leakage_safe_split
from spirit_dataset.records import (
    ExclusionReason,
    JudgmentTrace,
    MemoryEvidence,
    SourceClass,
    SpiritExclusionRecord,
    SpiritTrainingRecord,
    TrainingTask,
)

SELF_JUDGMENT_CUE = "행동 판단 학습"
SELF_MEMORY_CUE = "내가 알고 겪은 것을 짧게 떠올린다."


def judgment_training_records(records: list[SpiritTrainingRecord]) -> list[SpiritTrainingRecord]:
    output: list[SpiritTrainingRecord] = []
    for record in records:
        output.append(record)
        trace = record.judgment
        if record.task is not TrainingTask.PERSONA_SPEECH or not trace.is_complete:
            continue
        # A separate, training-only label task; never reasoning followed by speech.
        prompt = [*record.prompt]
        prompt[0] = replace(
            prompt[0],
            content="\n".join(
                dict.fromkeys(
                    (
                        *prompt[0].content.splitlines(),
                        *(trace.activated_memory or ()),
                    )
                )
            ),
        )
        prompt[-1] = replace(prompt[-1], content=f"{SELF_JUDGMENT_CUE}\n{prompt[-1].content}")
        from common.model_contract import BEHAVIOR_FIELDS

        target = json.dumps(
            {name: getattr(trace, name) for name in BEHAVIOR_FIELDS},
            ensure_ascii=False,
            separators=(",", ":"),
        )
        output.append(
            replace(
                record,
                id=f"{record.id}:judgment",
                origin_id=record.id,
                task=TrainingTask.BEHAVIOR_JUDGMENT,
                source_class=SourceClass.DERIVED_BEHAVIOR,
                evidence=(
                    MemoryEvidence(
                        SourceClass.CANON_DIALOGUE,
                        f"record_key={judgment_record_key(record.to_dict())}",
                    ),
                ),
                prompt=prompt,
                completion=[SFTRecordTurn("assistant", target)],
            )
        )
    return output


def split_with_judgments(
    records: list[SpiritTrainingRecord],
    config: SplitConfig,
    exclusions: list[SpiritExclusionRecord] | None = None,
) -> DatasetSplit[SpiritTrainingRecord]:
    accepted = []
    for record in judgment_training_records(records):
        target = "\n".join(turn.content for turn in record.completion)
        context = "\n".join(turn.content for turn in record.prompt)
        if target in context:
            if exclusions is not None:
                exclusions.append(
                    SpiritExclusionRecord(
                        ExclusionReason.TARGET_LEAKAGE,
                        "Target already present in input",
                        record.source,
                        target,
                    )
                )
        else:
            accepted.append(record)
    return leakage_safe_split(accepted, config)


def spirit_judgment_path(slug: str) -> Path:
    return SPIRIT_JUDGMENT_DIR / f"{slug}.jsonl"


def judgment_record_key(record: dict[str, Any]) -> str:
    material = {
        "source": {key: record["source"][key] for key in ("kind", "table", "keys")},
        "context": [turn["content"] for turn in record["prompt"] if turn["role"] != "system"],
        "speech": [turn["content"] for turn in record["completion"]],
    }
    encoded = json.dumps(material, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def apply_source_judgments(
    slug: str, records: list[SpiritTrainingRecord]
) -> tuple[list[SpiritTrainingRecord], list[SpiritExclusionRecord]]:
    path = spirit_judgment_path(slug)
    if not path.exists():
        return records, []
    annotations = [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    by_key = {row["record_key"]: row for row in annotations}
    if len(by_key) != len(annotations):
        raise EvaiError(f"Duplicate source judgment in {path}")
    used: set[str] = set()
    output: list[SpiritTrainingRecord] = []
    exclusions: list[SpiritExclusionRecord] = []
    keyed_records = [(judgment_record_key(record.to_dict()), record) for record in records]
    source_by_key = dict(keyed_records)
    for key, record in keyed_records:
        annotation = by_key.get(key)
        if annotation is None:
            output.append(record)
            continue
        if annotation.get("annotation_kind") == "non_self_speech":
            evidence = annotation["evidence"]
            original = "\n".join(turn.content for turn in record.completion)
            supporting = source_by_key.get(annotation.get("evidence_record_key", key))
            if supporting is None or (
                supporting.source.kind,
                supporting.source.table,
                supporting.source.keys[:1],
            ) != (record.source.kind, record.source.table, record.source.keys[:1]):
                raise EvaiError(
                    f"Speaker evidence is outside the source episode: {slug}:{record.id}"
                )
            source_text = "\n".join(turn.content for turn in supporting.completion)
            if not evidence or any(text not in source_text for text in evidence):
                raise EvaiError(f"Speaker correction lacks source evidence: {slug}:{record.id}")
            exclusions.append(
                SpiritExclusionRecord(
                    ExclusionReason.NON_SELF_SPEECH,
                    annotation["reason"],
                    record.source,
                    original,
                )
            )
            used.add(key)
            continue
        raw = annotation["judgment"]
        trace = JudgmentTrace(
            situation=record.judgment.situation,
            activated_memory=tuple(raw["activated_memory"]),
            interpretation=raw["interpretation"],
            decision=raw["decision"],
            emotion=raw["emotion"],
            intention=raw["intention"],
            action=raw["action"],
            evidence=tuple(raw["evidence"]),
        )
        context = "\n".join(turn.content for turn in (*record.prompt, *record.completion))
        available_memory = record.prompt[0].content.splitlines()
        if not trace.is_complete or any(text not in context for text in trace.evidence):
            raise EvaiError(f"Judgment lacks complete source evidence: {slug}:{record.id}")
        if any(memory not in available_memory for memory in trace.activated_memory or ()):
            raise EvaiError(f"Judgment uses unavailable spirit memory: {slug}:{record.id}")
        output.append(replace(record, judgment=trace))
        used.add(key)
    if set(by_key) != used:
        raise EvaiError(f"Judgment source no longer matches dataset: {slug}, {set(by_key) - used}")
    return output, exclusions
