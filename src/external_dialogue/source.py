from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

from common.errors import EvaiError
from common.hashing import compute_file_sha256, stable_index


class SegmentKind(StrEnum):
    SPEECH = "speech"
    ACTION = "action"


@dataclass(frozen=True)
class TurnSegment:
    kind: SegmentKind
    text: str


@dataclass(frozen=True)
class SourceTurn:
    role: str
    segments: tuple[TurnSegment, ...]


@dataclass(frozen=True)
class SourceNames:
    character: tuple[str, ...]
    user: tuple[str, ...]


@dataclass(frozen=True)
class SourceDataset:
    name: str
    license: str
    id_prefix: str
    subset: str
    path: Path

    def require_path(self) -> Path:
        if not self.path.is_file():
            raise EvaiError(f"External dialogue source not downloaded: {self.path}")
        return self.path

    def sha256(self) -> str:
        return compute_file_sha256(self.require_path())


@dataclass(frozen=True)
class SourceConversation:
    dataset: SourceDataset
    row: int
    topic: str
    setting: str
    names: SourceNames
    turns: tuple[SourceTurn, ...]

    @property
    def subset(self) -> str:
        return self.dataset.subset

    @property
    def conversation_id(self) -> str:
        return f"{self.dataset.id_prefix}_{self.subset}_{self.row}"

    def provenance(self) -> dict[str, Any]:
        return {
            "dataset": self.dataset.name,
            "license": self.dataset.license,
            "subset": self.subset,
            "row": self.row,
            "topic": self.topic,
        }


def assign_source_conversations(
    slugs: list[str], conversations: list[SourceConversation]
) -> dict[str, list[SourceConversation]]:
    if not slugs:
        raise EvaiError("External dialogue assignment requires registered spirits")
    ordered = sorted(slugs)
    assignment: dict[str, list[SourceConversation]] = {slug: [] for slug in ordered}
    for conversation in conversations:
        assignment[ordered[stable_index(conversation.conversation_id, len(ordered))]].append(
            conversation
        )
    return assignment
