from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from common.messages import SFTRecordTurn

MEMORY_LINE_MAX_LENGTH = 29


class SourceClass(StrEnum):
    CANON_TBL = "canon_tbl"
    CANON_STORY = "canon_story"
    CANON_DIALOGUE = "canon_dialogue"
    PROJECT_CONTRACT = "project_contract"
    DERIVED_MEMORY = "derived_memory"
    DERIVED_BEHAVIOR = "derived_behavior"
    DERIVED_SPEECH = "derived_speech"
    GENERAL_KNOWLEDGE = "general_knowledge"


class TrainingTask(StrEnum):
    PERSONA_SPEECH = "persona_speech"
    SELF_MEMORY = "self_memory"
    BEHAVIOR_JUDGMENT = "behavior_judgment"


@dataclass(frozen=True)
class MemoryEvidence:
    source_class: SourceClass
    reference: str

    def to_dict(self) -> dict[str, str]:
        return {"source_class": self.source_class.value, "reference": self.reference}


@dataclass(frozen=True)
class SelfMemory:
    text: str
    evidence: tuple[MemoryEvidence, ...]
    cue: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "source_class": SourceClass.DERIVED_MEMORY.value,
            "evidence": [item.to_dict() for item in self.evidence],
            "cue": self.cue,
        }


class SourceKind(StrEnum):
    PERSONA_LESSON = "persona_lesson"
    SELF_MEMORY = "self_memory"
    STORY = "story"
    EVERTALK = "evertalk"
    LOBBY = "lobby"
    BUBBLE = "bubble"
    TRIP = "trip"
    TOWN_LOST_ITEM = "town_lost_item"
    HERO_DESC = "hero_desc"
    HERO_COMMENT = "hero_comment"


class ExclusionReason(StrEnum):
    TARGET_LEAKAGE = "target_leakage"
    NON_SELF_SPEECH = "non_self_speech"
    CONTENT_CLASSIFICATION = "content_classification"
    GAME_FEATURE_GUIDE = "game_feature_guide"
    UNVERIFIED_TRIGGER = "unverified_trigger"
    UNRESOLVED_STRING = "unresolved_string"
    EMPTY_TEXT = "empty_text"
    UNANSWERED_CONTEXT = "unanswered_context"


@dataclass(frozen=True)
class SourceReference:
    kind: SourceKind
    table: str
    keys: tuple[int, ...]
    story_no: int | None = None
    source_class: SourceClass = SourceClass.CANON_TBL

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind.value,
            "table": self.table,
            "keys": list(self.keys),
            "source_class": self.source_class.value,
            **({"story_no": self.story_no} if self.story_no is not None else {}),
        }


@dataclass(frozen=True)
class JudgmentTrace:
    situation: str
    activated_memory: tuple[str, ...] | None = None
    interpretation: str | None = None
    decision: str | None = None
    emotion: str | None = None
    intention: str | None = None
    action: str | None = None
    evidence: tuple[str, ...] = ()

    @property
    def is_complete(self) -> bool:
        return bool(self.evidence) and all(
            (
                self.activated_memory,
                self.interpretation,
                self.decision,
                self.emotion,
                self.intention,
                self.action,
            )
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "situation": self.situation,
            "activated_memory": (
                None if self.activated_memory is None else list(self.activated_memory)
            ),
            "interpretation": self.interpretation,
            "decision": self.decision,
            "emotion": self.emotion,
            "intention": self.intention,
            "action": self.action,
            "evidence": list(self.evidence),
            "complete": self.is_complete,
            "source_class": SourceClass.DERIVED_BEHAVIOR.value if self.is_complete else None,
        }


@dataclass(frozen=True)
class SpiritTrainingRecord:
    id: str
    source: SourceReference
    love_level: int | None
    judgment: JudgmentTrace
    prompt: list[SFTRecordTurn]
    completion: list[SFTRecordTurn]
    source_class: SourceClass = SourceClass.CANON_DIALOGUE
    evidence: tuple[MemoryEvidence, ...] = ()
    task: TrainingTask = TrainingTask.PERSONA_SPEECH
    origin_id: str | None = None
    event_keys: tuple[str, ...] = ()
    language: str = "ko"

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "task": self.task.value,
            "origin_id": self.origin_id or self.id,
            "event_keys": list(self.event_keys),
            "language": self.language,
            "source_class": self.source_class.value,
            "source": self.source.to_dict(),
            "love_level": self.love_level,
            "judgment": self.judgment.to_dict(),
            "prompt": [turn.to_dict() for turn in self.prompt],
            "completion": [turn.to_dict() for turn in self.completion],
            "evidence": [item.to_dict() for item in self.evidence],
        }


@dataclass(frozen=True)
class SpiritExclusionRecord:
    reason: ExclusionReason
    detail: str
    source: SourceReference
    text_preview: str = field(default="")

    def to_dict(self) -> dict[str, Any]:
        return {
            "reason": self.reason.value,
            "detail": self.detail,
            "source": self.source.to_dict(),
            "text_preview": self.text_preview,
        }
