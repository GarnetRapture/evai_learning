"""Deterministic SFT dataset construction from classified dialogue exchanges.

This module is a deterministic transformer, never a generator (plan.md
SS12): it converts already-extracted, already-classified DialogueExchange
records into SFTDatasetRecord entries, applying only conservative
normalization to turn text. It never invents dialogue, rewrites
personality, or paraphrases canonical lines. Every excluded exchange is
preserved as an ExclusionRecord carrying its original source_file/source_index
so no contamination decision destroys provenance (plan.md SS6).
"""

from dataclasses import dataclass, field
from typing import Any

from .dialogue import (
    DialogueExchange,
    DialogueExtractionResult,
    SpeakerRole,
    TurnClassification,
    classify_dialogue_turn,
)
from .normalize import is_empty_text, normalize_text


@dataclass(frozen=True)
class SFTRecordTurn:
    """A role-content pair in a conversational sample."""

    role: str
    content: str

    def to_dict(self) -> dict[str, str]:
        return {"role": self.role, "content": self.content}


@dataclass(frozen=True)
class SFTDatasetRecord:
    """Canonical SFT dataset entry conforming to EVAI fine-tuning specification."""

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
        """Return the standard conversational SFT message sequence."""
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
    """Provenance-preserving record of one dialogue exchange excluded from SFT training."""

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
    """Convert classified DialogueTurn objects into role/content SFT turns.

    Applies only conservative normalization (Unicode NFC, line-ending,
    surrounding whitespace) to each turn's text; no rewriting or paraphrasing.
    """
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
    """Convert one accepted DialogueExchange into a canonical SFTDatasetRecord.

    Returns None when the completion is empty after normalization, since an
    empty assistant turn carries no learnable persona signal.
    """
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
    """Build a traceable exclusion record for one non-accepted DialogueExchange."""
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
    """Split one persona's classified exchanges into accepted SFT records and exclusions.

    Only ACCEPTED exchanges become SFT training records; every other
    classification is preserved as an ExclusionRecord instead of being
    silently dropped (plan.md SS6).
    """
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


def build_completion_only_record(
    text: str,
    persona_id: str,
    persona_name: str,
    language: str,
    source_type: str,
    source_file: str,
    source_index: int,
) -> SFTDatasetRecord | ExclusionRecord | None:
    """Build a prompt-less SFT record from a single canonical persona utterance.

    Used for `speech_patterns` entries and `personality.greeting`, which are
    canonical persona speech but not part of a user/persona exchange
    (plan.md SS11: dataset priority ranks these below actual dialogue).
    Returns None for empty text; returns an ExclusionRecord instead of a
    training record when contamination classification rejects the text.
    """
    if is_empty_text(text):
        return None

    normalized = normalize_text(text)
    classification = classify_dialogue_turn(
        SpeakerRole.ASSISTANT, persona_name, normalized, source_type
    )
    if classification is not TurnClassification.ACCEPTED:
        return ExclusionRecord(
            persona_id=persona_id,
            persona_name=persona_name,
            classification=classification,
            source_type=source_type,
            source_file=source_file,
            source_index=source_index,
            speaker_names=[persona_name],
            completion_preview=normalized[:80],
        )

    completion = [SFTRecordTurn(role=SpeakerRole.ASSISTANT.value, content=normalized)]
    return SFTDatasetRecord(
        id=f"{persona_id}:{source_type}:{source_index}",
        persona_id=persona_id,
        persona_name=persona_name,
        language=language,
        source_type=source_type,
        source_file=source_file,
        source_index=source_index,
        prompt=[],
        completion=completion,
        assistant_segments=[normalized],
    )


def build_speech_pattern_records(
    speech_patterns: list[str],
    persona_id: str,
    persona_name: str,
    language: str,
    source_file: str,
) -> tuple[list[SFTDatasetRecord], list[ExclusionRecord]]:
    """Build completion-only SFT records from a persona's canonical speech_patterns."""
    records: list[SFTDatasetRecord] = []
    exclusions: list[ExclusionRecord] = []
    for index, pattern in enumerate(speech_patterns):
        result = build_completion_only_record(
            pattern, persona_id, persona_name, language, "speech_pattern", source_file, index
        )
        if isinstance(result, SFTDatasetRecord):
            records.append(result)
        elif isinstance(result, ExclusionRecord):
            exclusions.append(result)
    return records, exclusions


def build_greeting_record(
    greeting: str | None,
    persona_id: str,
    persona_name: str,
    language: str,
    source_file: str,
) -> tuple[SFTDatasetRecord | None, ExclusionRecord | None]:
    """Build a completion-only SFT record from a persona's canonical greeting."""
    if greeting is None:
        return None, None
    result = build_completion_only_record(
        greeting, persona_id, persona_name, language, "greeting", source_file, 0
    )
    if isinstance(result, SFTDatasetRecord):
        return result, None
    if isinstance(result, ExclusionRecord):
        return None, result
    return None, None
