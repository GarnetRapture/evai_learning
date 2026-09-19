import threading
import uuid
from dataclasses import dataclass
from typing import Any

from common.errors import EvaiError
from common.model_contract import TRAINING_LANGUAGES
from inference.spirit_runtime import SpiritRuntime
from inference.spirit_session import SpiritSession
from spirit_dataset.runtime_prompt import load_spirit_prompt_source, spirit_selection_header


@dataclass(frozen=True)
class SpiritCatalogEntry:
    slug: str
    names: dict[str, str]
    system_prompt: str

    def to_dict(self) -> dict[str, Any]:
        return {"slug": self.slug, "names": self.names, "system_prompt": self.system_prompt}


def build_spirit_catalog(slugs: list[str]) -> list[SpiritCatalogEntry]:
    return [
        SpiritCatalogEntry(
            slug=slug,
            names={
                language: load_spirit_prompt_source(slug, language).name
                for language in TRAINING_LANGUAGES
            },
            system_prompt=spirit_selection_header(slug),
        )
        for slug in slugs
    ]


class WebChatSessions:
    def __init__(self, runtime: SpiritRuntime) -> None:
        self._runtime = runtime
        self._sessions: dict[str, SpiritSession] = {}
        self._lock = threading.Lock()

    def open(self, spirit_id: str, language: str) -> tuple[str, SpiritSession]:
        session = self._runtime.open_session(spirit_id, language)
        session_id = uuid.uuid4().hex
        with self._lock:
            self._sessions[session_id] = session
        return session_id, session

    def get(self, session_id: str) -> SpiritSession:
        with self._lock:
            session = self._sessions.get(session_id)
        if session is None:
            raise EvaiError(f"Unknown web chat session: {session_id}")
        return session

    def close(self, session_id: str) -> None:
        with self._lock:
            self._sessions.pop(session_id, None)


def session_state(session_id: str, session: SpiritSession) -> dict[str, Any]:
    system_prompt = spirit_selection_header(session.spirit_id)
    return {
        "session_id": session_id,
        "spirit_id": session.spirit_id,
        "spirit_name": session.spirit_name,
        "language": session.language,
        "system_prompt": system_prompt,
        "messages": [{"role": "system", "content": system_prompt}, *session.history()],
    }
