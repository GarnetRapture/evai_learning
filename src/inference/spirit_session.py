"""A platform conversation bound to one selected spirit and its own history."""

import threading
from typing import TYPE_CHECKING

from common.errors import EvaiError
from common.model_contract import TRAINING_LANGUAGES
from inference.generation import DEFAULT_GENERATION_SETTINGS, GenerationSettings
from spirit_dataset.runtime_prompt import build_chat_messages

if TYPE_CHECKING:
    from inference.spirit_runtime import SpiritRuntime


class SpiritSession:
    def __init__(
        self,
        runtime: SpiritRuntime,
        spirit_id: str,
        language: str = "ko",
    ) -> None:
        if language not in TRAINING_LANGUAGES:
            raise EvaiError(f"Unsupported spirit language: {language}")
        self._source = runtime.prompt_source(spirit_id, language)
        self._language = language
        self._runtime = runtime
        self._history: list[dict[str, str]] = []
        self._lock = threading.Lock()

    @property
    def spirit_id(self) -> str:
        return self._source.slug

    @property
    def spirit_name(self) -> str:
        return self._source.name

    @property
    def language(self) -> str:
        return self._language

    def history(self) -> list[dict[str, str]]:
        with self._lock:
            return [dict(turn) for turn in self._history]

    def respond(
        self, user_message: str, settings: GenerationSettings = DEFAULT_GENERATION_SETTINGS
    ) -> str:
        with self._lock:
            messages = build_chat_messages(
                self._source,
                user_message,
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
