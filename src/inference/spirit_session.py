"""A platform conversation bound to one selected spirit and its own history."""

import threading
from typing import TYPE_CHECKING

from common.errors import EvaiError
from common.model_contract import TRAINING_LANGUAGES
from inference.generation import DEFAULT_GENERATION_SETTINGS, GenerationSettings
from spirit_dataset.memory import MAX_LOVE_LEVEL, MIN_LOVE_LEVEL
from spirit_dataset.runtime_prompt import build_chat_messages

if TYPE_CHECKING:
    from inference.spirit_runtime import SpiritRuntime


class SpiritSession:
    def __init__(
        self,
        runtime: SpiritRuntime,
        spirit_id: str,
        language: str = "ko",
        love_level: int = MIN_LOVE_LEVEL,
    ) -> None:
        if language not in TRAINING_LANGUAGES:
            raise EvaiError(f"Unsupported spirit language: {language}")
        if not MIN_LOVE_LEVEL <= love_level <= MAX_LOVE_LEVEL:
            raise EvaiError(f"love_level must be {MIN_LOVE_LEVEL}..{MAX_LOVE_LEVEL}")
        self._source = runtime.prompt_source(spirit_id, language)
        self._runtime = runtime
        self._love_level = love_level
        self._history: list[dict[str, str]] = []
        self._lock = threading.Lock()

    @property
    def spirit_id(self) -> str:
        return self._source.slug

    def respond(
        self, user_message: str, settings: GenerationSettings = DEFAULT_GENERATION_SETTINGS
    ) -> str:
        with self._lock:
            messages = build_chat_messages(
                self._source,
                user_message,
                self._love_level,
                conversation_history=self._history,
            )
            # The session's spirit ID and history are inputs to the same shared model.
            response = self._runtime.reply(self.spirit_id, messages, settings)
            self._history.extend(
                [
                    {"role": "user", "content": user_message},
                    {"role": "assistant", "content": response},
                ]
            )
            return response

    def clear_history(self) -> None:
        with self._lock:
            self._history.clear()
