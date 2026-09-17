"""Canonical text message type shared by dataset and prompt composition."""

from dataclasses import dataclass


@dataclass(frozen=True)
class SFTRecordTurn:
    role: str
    content: str

    def to_dict(self) -> dict[str, str]:
        return {"role": self.role, "content": self.content}
