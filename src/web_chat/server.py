import json
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, cast

from common.errors import EvaiError
from common.model_contract import TRAINING_LANGUAGES
from inference.spirit_runtime import SpiritRuntime
from web_chat.sessions import (
    SpiritCatalogEntry,
    WebChatSessions,
    build_spirit_catalog,
    session_state,
)

STATIC_DIR = Path(__file__).with_name("static")
STATIC_FILES: dict[str, tuple[str, str]] = {
    "/": ("index.html", "text/html; charset=utf-8"),
    "/static/style.css": ("style.css", "text/css; charset=utf-8"),
    "/static/app.js": ("app.js", "text/javascript; charset=utf-8"),
}
MAX_REQUEST_BYTES = 64 * 1024


class WebChatServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address: tuple[str, int], runtime: SpiritRuntime) -> None:
        super().__init__(address, WebChatHandler)
        self.sessions = WebChatSessions(runtime)
        self.catalog: list[SpiritCatalogEntry] = build_spirit_catalog(runtime.persona_ids)
        self.languages: tuple[str, ...] = tuple(TRAINING_LANGUAGES)


class WebChatHandler(BaseHTTPRequestHandler):
    @property
    def chat_server(self) -> WebChatServer:
        return cast(WebChatServer, self.server)

    def do_GET(self) -> None:
        if self.path in STATIC_FILES:
            file_name, content_type = STATIC_FILES[self.path]
            self._send(HTTPStatus.OK, (STATIC_DIR / file_name).read_bytes(), content_type)
        elif self.path == "/api/spirits":
            self._send_json(
                HTTPStatus.OK,
                {
                    "languages": list(self.chat_server.languages),
                    "spirits": [entry.to_dict() for entry in self.chat_server.catalog],
                },
            )
        else:
            self._send_json(HTTPStatus.NOT_FOUND, {"error": f"Unknown path: {self.path}"})

    def do_POST(self) -> None:
        try:
            payload = self._read_json()
            if self.path == "/api/sessions":
                session_id, session = self.chat_server.sessions.open(
                    self._text(payload, "spirit_id"), self._language(payload)
                )
                self._send_json(HTTPStatus.OK, session_state(session_id, session))
            elif self.path == "/api/chat":
                session_id = self._text(payload, "session_id")
                session = self.chat_server.sessions.get(session_id)
                response = session.respond(self._text(payload, "message"))
                self._send_json(
                    HTTPStatus.OK,
                    {"response": response, **session_state(session_id, session)},
                )
            elif self.path == "/api/sessions/close":
                self.chat_server.sessions.close(self._text(payload, "session_id"))
                self._send_json(HTTPStatus.OK, {"closed": True})
            else:
                self._send_json(HTTPStatus.NOT_FOUND, {"error": f"Unknown path: {self.path}"})
        except EvaiError as err:
            self._send_json(HTTPStatus.BAD_REQUEST, {"error": str(err)})

    def _language(self, payload: dict[str, Any]) -> str:
        language = self._text(payload, "language")
        if language not in self.chat_server.languages:
            raise EvaiError(f"Unsupported spirit language: {language}")
        return language

    @staticmethod
    def _text(payload: dict[str, Any], field: str) -> str:
        value = payload.get(field)
        if not isinstance(value, str) or not value.strip():
            raise EvaiError(f"Request field '{field}' must be non-empty text")
        return value.strip()

    def _read_json(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length") or 0)
        if length <= 0 or length > MAX_REQUEST_BYTES:
            raise EvaiError("Request body must be JSON within the size limit")
        try:
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as err:
            raise EvaiError(f"Invalid JSON request: {err}") from err
        if not isinstance(payload, dict):
            raise EvaiError("Request body must be a JSON object")
        return payload

    def _send_json(self, status: HTTPStatus, payload: dict[str, Any]) -> None:
        self._send(
            status,
            json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            "application/json; charset=utf-8",
        )

    def _send(self, status: HTTPStatus, body: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)


def serve_web_chat(runtime: SpiritRuntime, host: str, port: int) -> None:
    with WebChatServer((host, port), runtime) as server:
        print(f"* Web chat: http://{host}:{server.server_address[1]}/", flush=True)
        server.serve_forever()
