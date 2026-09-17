from dataclasses import dataclass, field
from typing import Any

from common.messages import SFTRecordTurn
from sft_dataset.dialogue import (
    DialogueExchange,
    DialogueExtractionResult,
    TurnClassification,
)
from sft_dataset.normalize import is_empty_text, normalize_text


@dataclass(frozen=True)
class SFTDatasetRecord:
    id: str
    persona_id: str
    persona_name: str
    language: str
    source_type: str
    source_file: str
    source_index: int
    prompt: list[SFTRecordTurn]
    completion: list[SFTRecordTurn]
    assistant_segments: list[str] = field(default_factory=list)

    @property
    def messages(self) -> list[SFTRecordTurn]:
        return [*self.prompt, *self.completion]

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "persona_id": self.persona_id,
            "persona_name": self.persona_name,
            "language": self.language,
            "source_type": self.source_type,
            "source_file": self.source_file,
            "source_index": self.source_index,
            "messages": [turn.to_dict() for turn in self.messages],
            "prompt": [t.to_dict() for t in self.prompt],
            "completion": [t.to_dict() for t in self.completion],
            "assistant_segments": self.assistant_segments,
        }


@dataclass(frozen=True)
class ExclusionRecord:
    persona_id: str
    persona_name: str
    classification: TurnClassification
    source_type: str
    source_file: str
    source_index: int
    speaker_names: list[str]
    completion_preview: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "persona_id": self.persona_id,
            "persona_name": self.persona_name,
            "classification": self.classification.value,
            "source_type": self.source_type,
            "source_file": self.source_file,
            "source_index": self.source_index,
            "speaker_names": self.speaker_names,
            "completion_preview": self.completion_preview,
        }


def _build_record_turns(turns: list) -> list[SFTRecordTurn]:
    return [
        SFTRecordTurn(role=turn.role.value, content=normalize_text(turn.content)) for turn in turns
    ]


def build_sft_record(
    exchange: DialogueExchange,
    persona_id: str,
    persona_name: str,
    language: str,
    source_file: str,
) -> SFTDatasetRecord | None:
    completion_turns = _build_record_turns(exchange.completion)
    if all(is_empty_text(t.content) for t in completion_turns):
        return None

    return SFTDatasetRecord(
        id=f"{persona_id}:{exchange.exchange_id}",
        persona_id=persona_id,
        persona_name=persona_name,
        language=language,
        source_type=exchange.source_type,
        source_file=source_file,
        source_index=exchange.source_index,
        prompt=_build_record_turns(exchange.prompt),
        completion=completion_turns,
        assistant_segments=[normalize_text(s) for s in exchange.assistant_segments],
    )


def build_exclusion_record(
    exchange: DialogueExchange, persona_id: str, persona_name: str, source_file: str
) -> ExclusionRecord:
    speaker_names = sorted({turn.speaker_name for turn in (*exchange.prompt, *exchange.completion)})
    preview = " ".join(exchange.assistant_segments)[:80]
    return ExclusionRecord(
        persona_id=persona_id,
        persona_name=persona_name,
        classification=exchange.classification,
        source_type=exchange.source_type,
        source_file=source_file,
        source_index=exchange.source_index,
        speaker_names=speaker_names,
        completion_preview=preview,
    )


def build_persona_dataset(
    extraction: DialogueExtractionResult,
    persona_id: str,
    language: str,
    source_file: str,
) -> tuple[list[SFTDatasetRecord], list[ExclusionRecord]]:
    records: list[SFTDatasetRecord] = []
    exclusions: list[ExclusionRecord] = []

    for exchange in (*extraction.evertalk_exchanges, *extraction.story_exchanges):
        if exchange.classification is TurnClassification.ACCEPTED:
            record = build_sft_record(
                exchange, persona_id, extraction.persona_name, language, source_file
            )
            if record is not None:
                records.append(record)
            else:
                exclusions.append(
                    build_exclusion_record(
                        exchange, persona_id, extraction.persona_name, source_file
                    )
                )
        else:
            exclusions.append(
                build_exclusion_record(exchange, persona_id, extraction.persona_name, source_file)
            )

    return records, exclusions
