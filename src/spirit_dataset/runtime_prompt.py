import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from common.errors import EvaiError
from common.paths import DATASETS_DIR, SPIRIT_FILE_NAME
from sft_dataset.dialogue import SpeakerRole
from sft_dataset.storage import persona_dataset_dir
from spirit_dataset.memory import (
    PastMemory,
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


GENERAL_CORPUS_ID = "general_corpus"
GENERAL_CONTEXT_HEADER = "Context: general\n"


def spirit_selection_header(slug: str) -> str:
    return f"SpiritId: {slug}\n"


def bind_general_context(messages: list[dict[str, str]]) -> list[dict[str, str]]:
    body = [message for message in messages if message["role"] != SpeakerRole.SYSTEM.value]
    return [{"role": SpeakerRole.SYSTEM.value, "content": GENERAL_CONTEXT_HEADER}, *body]


def bind_training_context(messages: list[dict[str, str]], context_id: str) -> list[dict[str, str]]:
    if context_id == GENERAL_CORPUS_ID:
        return bind_general_context(messages)
    return bind_spirit_identity(messages, context_id)


def bind_spirit_identity(messages: list[dict[str, str]], slug: str) -> list[dict[str, str]]:
    """The model receives the selected ID; identity and feelings are learned weights."""
    if not messages or messages[0]["role"] != SpeakerRole.SYSTEM.value:
        raise EvaiError("Spirit training and conversations require a system identity")
    content = messages[0]["content"]
    header = spirit_selection_header(slug)
    if content.startswith("SpiritId: "):
        if not content.startswith(header):
            raise EvaiError(f"Dataset/system identity differs from its owner: {slug}")
    return [
        {
            "role": SpeakerRole.SYSTEM.value,
            "content": header,
        },
        *messages[1:],
    ]


def load_spirit_prompt_source(
    slug: str, language: str = "ko", profiles_root: Path = DATASETS_DIR
) -> SpiritPromptSource:
    path = profiles_root / slug / SPIRIT_FILE_NAME
    if not path.exists():
        raise EvaiError(f"Spirit profile not found: {path}. Run `build-dataset` first.")
    raw = json.loads(path.read_text(encoding="utf-8"))
    try:
        profile = raw["profiles"][language]
        if profile["slug"] != slug:
            raise EvaiError(f"Profile identity differs from selected spirit: {slug}")
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
    previous_spirit_text: str | None = None,
    conversation_history: list[dict[str, str]] | None = None,
) -> list[dict[str, str]]:
    system = spirit_selection_header(source.slug)
    messages = [{"role": SpeakerRole.SYSTEM.value, "content": system}]
    if conversation_history:
        messages.extend(conversation_history)
    elif previous_spirit_text:
        messages.append({"role": SpeakerRole.ASSISTANT.value, "content": previous_spirit_text})
    messages.append({"role": SpeakerRole.USER.value, "content": user_message})
    return messages
