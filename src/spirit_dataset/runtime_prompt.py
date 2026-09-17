import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from common.errors import EvaiError
from common.paths import SPIRIT_FILE_NAME
from sft_dataset.dialogue import SpeakerRole
from sft_dataset.storage import persona_dataset_dir
from spirit_dataset.memory import (
    MAX_LOVE_LEVEL,
    MIN_LOVE_LEVEL,
    PastMemory,
    compose_identity_prompt,
    past_memory_from_dict,
)


@dataclass(frozen=True)
class SpiritPromptSource:
    slug: str
    name: str
    identity_memory: tuple[str, ...]
    past_memories: list[PastMemory]
    profile: dict[str, Any] = field(default_factory=dict)
    language: str = "kr"


def spirit_file_path(slug: str) -> Path:
    return persona_dataset_dir(slug) / SPIRIT_FILE_NAME


def load_spirit_prompt_source(slug: str, language: str = "ko") -> SpiritPromptSource:
    path = spirit_file_path(slug)
    if not path.exists():
        raise EvaiError(f"Spirit profile not found: {path}. Run `build-dataset` first.")
    raw = json.loads(path.read_text(encoding="utf-8"))
    try:
        profile = raw["profiles"][language]
        return SpiritPromptSource(
            slug=str(profile["slug"]),
            name=str(profile["name"]),
            identity_memory=tuple(str(line) for line in profile["identity_memory"]),
            past_memories=[past_memory_from_dict(item) for item in raw["episodic_memory"]],
            profile=profile,
            language=profile["language"],
        )
    except (KeyError, TypeError, ValueError) as err:
        raise EvaiError(f"Invalid spirit profile file {path}: {err}") from err


def build_chat_messages(
    source: SpiritPromptSource,
    user_message: str,
    love_level: int,
    previous_spirit_text: str | None = None,
    conversation_history: list[dict[str, str]] | None = None,
) -> list[dict[str, str]]:
    if not MIN_LOVE_LEVEL <= love_level <= MAX_LOVE_LEVEL:
        raise EvaiError(f"love_level must be {MIN_LOVE_LEVEL}..{MAX_LOVE_LEVEL}: {love_level}")
    system = compose_identity_prompt(source.identity_memory, love_level, language=source.language)
    messages = [{"role": SpeakerRole.SYSTEM.value, "content": system}]
    if conversation_history:
        messages.extend(conversation_history)
    elif previous_spirit_text:
        messages.append({"role": SpeakerRole.ASSISTANT.value, "content": previous_spirit_text})
    messages.append({"role": SpeakerRole.USER.value, "content": user_message})
    return messages
