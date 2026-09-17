import json
import random
import re
import threading
import urllib.parse
import webbrowser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from common.errors import EvaiError
from common.paths import DATA_DIR, DATASETS_DIR, spirit_adapter_dir
from persona.loader import discover_persona_files, load_persona_file
from persona.schema import PersonaData
from spirit_dataset.builder import ROSTER_FILE_NAME
from spirit_dataset.memory import MAX_LOVE_LEVEL, MIN_LOVE_LEVEL
from spirit_dataset.runtime_prompt import (
    SpiritPromptSource,
    build_chat_messages,
    load_spirit_prompt_source,
    spirit_file_path,
)

WEB_DIR = Path(__file__).resolve().parent
INDEX_HTML_PATH = WEB_DIR / "index.html"
ASSETS_DIR = WEB_DIR / "assets"

RACE_BADGE_MAP: dict[str, str] = {
    "불사형": "undead.svg",
    "인간형": "human.svg",
    "야수형": "beast.svg",
    "요정형": "elf.svg",
    "천사형": "angel.svg",
    "악마형": "demon.svg",
    "혼돈형": "chaos.svg",
}

_TALK_BG_BASE = "/assets/eversoul-assets/backgrounds/talk"

TALK_BACKGROUND_PRESETS: list[dict[str, str]] = [
    {
        "id": "garnet_room_01",
        "name": "가넷의 방 (조명)",
        "url": f"{_TALK_BG_BASE}/Talk_BG_Garnet_Room_01.png",
    },
    {
        "id": "garnet_room_02",
        "name": "가넷의 방 (침실)",
        "url": f"{_TALK_BG_BASE}/Talk_BG_Garnet_Room_02.png",
    },
    {"id": "my_room", "name": "구원자의 방", "url": f"{_TALK_BG_BASE}/Talk_BG_MyRoom.png"},
    {"id": "ark", "name": "방주 내부", "url": f"{_TALK_BG_BASE}/Talk_BG_Ark.png"},
    {"id": "cafe", "name": "거리의 카페", "url": f"{_TALK_BG_BASE}/Talk_BG_Cafe_01.png"},
    {"id": "night_sky", "name": "밤하늘", "url": f"{_TALK_BG_BASE}/Talk_BG_Night_Sky.png"},
    {"id": "garden", "name": "정원", "url": f"{_TALK_BG_BASE}/Talk_BG_Garden.png"},
    {"id": "library", "name": "도서관", "url": f"{_TALK_BG_BASE}/Talk_BG_Library.png"},
    {
        "id": "ballroom_night",
        "name": "무도회장 (밤)",
        "url": f"{_TALK_BG_BASE}/Talk_BG_Ballroom_Night.png",
    },
]


def find_persona_assets(persona_id: str, race: str) -> dict[str, str | None]:
    spirits_dir = ASSETS_DIR / "eversoul-assets" / "spirits"
    avatar_url: str | None = None
    standing_url: str | None = None

    normalized_target = persona_id.replace("_", "").lower()

    if spirits_dir.exists() and spirits_dir.is_dir():
        matched_dir: Path | None = None
        for candidate in spirits_dir.iterdir():
            if not candidate.is_dir():
                continue
            cand_norm = candidate.name.lower()
            if cand_norm == normalized_target:
                matched_dir = candidate
                break

        if matched_dir is None:
            base_prefix = persona_id.split("_")[0].lower()
            for candidate in spirits_dir.iterdir():
                if not candidate.is_dir():
                    continue
                if candidate.name.lower().startswith(base_prefix):
                    matched_dir = candidate
                    break

        if matched_dir is not None:
            evertalk_dir = matched_dir / "evertalk"
            if evertalk_dir.exists():
                pngs = list(evertalk_dir.glob("*.png"))
                if pngs:
                    rel = pngs[0].relative_to(ASSETS_DIR).as_posix()
                    avatar_url = f"/assets/{rel}"

            if not avatar_url:
                icon_dir = matched_dir / "icon"
                if icon_dir.exists():
                    pngs = list(icon_dir.glob("*.png"))
                    if pngs:
                        rel = pngs[0].relative_to(ASSETS_DIR).as_posix()
                        avatar_url = f"/assets/{rel}"

            base_dir = matched_dir / "base"
            if base_dir.exists():
                for preferred in ("1024", "512", "2048"):
                    candidates = list(base_dir.glob(f"*{preferred}*.png"))
                    if candidates:
                        rel = candidates[0].relative_to(ASSETS_DIR).as_posix()
                        standing_url = f"/assets/{rel}"
                        break
                if not standing_url:
                    pngs = list(base_dir.glob("*.png"))
                    if pngs:
                        rel = pngs[0].relative_to(ASSETS_DIR).as_posix()
                        standing_url = f"/assets/{rel}"

    badge_file = RACE_BADGE_MAP.get(race)
    race_badge_url: str | None = None
    if badge_file:
        race_badge_url = f"/assets/eversoul-assets/ui/race-badges/{badge_file}"

    return {
        "avatar_url": avatar_url,
        "standing_url": standing_url,
        "race_badge_url": race_badge_url,
        "savior_avatar_url": "/assets/eversoul-assets/savior/User_128.png",
        "evertalk_logo_url": "/assets/eversoul-assets/ui/evertalk/evertalk.png",
        "heart_icon_url": "/assets/eversoul-assets/icon/Icon_Heart.png",
    }


class PersonaStore:
    def __init__(self, data_dir: Path | None = None) -> None:
        self.data_dir = data_dir or DATA_DIR
        self._cache: dict[str, PersonaData] = {}
        self._summaries: list[dict[str, Any]] = []
        self._load_all()

    def _load_all(self) -> None:
        files = discover_persona_files(self.data_dir)
        roster = json.loads((DATASETS_DIR / ROSTER_FILE_NAME).read_text(encoding="utf-8"))
        registered = {entry["slug"]: entry for entry in roster["spirits"]}
        self._cache.clear()
        self._summaries.clear()

        for file_path in files:
            persona_id = file_path.stem
            if persona_id not in registered:
                continue
            try:
                persona = load_persona_file(file_path)
                canonical = json.loads(spirit_file_path(persona_id).read_text(encoding="utf-8"))
                profile = canonical["profile"]
                fields = profile["fields"]
                persona.id = str(profile["hero_no"])
                persona.name = profile["name"]
                persona.race = fields.get("race", persona.race)
                persona.profile.nick_name = fields.get("nickname")
                persona.profile.union = fields.get("union")
                for field in ("like", "dislike", "hobby", "speciality"):
                    setattr(persona.profile, field, [
                        item.strip() for item in fields.get(field, "").split(",") if item.strip()
                    ])
                persona.personality.description = fields.get("introduction")
                persona.personality.greeting = fields.get("greeting")
                self._cache[persona_id] = persona

                assets = find_persona_assets(persona_id, persona.race)

                summary = {
                    "id": persona_id,
                    "in_game_id": persona.id,
                    "name": persona.name,
                    "name_en": persona.name_en,
                    "grade": persona.grade,
                    "race": persona.race,
                    "class": persona.class_,
                    "sub_class": persona.sub_class,
                    "nick_name": persona.profile.nick_name or "",
                    "union": persona.profile.union or "",
                    "birthday": persona.profile.birthday or "",
                    "cv_ko": persona.profile.cv_ko or "",
                    "cv_jp": persona.profile.cv_jp or "",
                    "likes": persona.profile.like,
                    "dislikes": persona.profile.dislike,
                    "greeting": persona.personality.greeting or "",
                    "speech_patterns_count": len(persona.speech_patterns),
                    "evertalk_count": len(persona.dialogues.evertalk),
                    "story_count": len(persona.dialogues.story),
                    "assets": assets,
                }
                self._summaries.append(summary)
            except (OSError, ValueError, KeyError) as err:
                raise EvaiError(
                    f"Failed to load registered spirit {file_path.name}: {err}"
                ) from err

        self._summaries.sort(key=lambda item: item["name"])

    def get_summaries(self) -> list[dict[str, Any]]:
        return self._summaries

    def get_persona(self, persona_id: str) -> PersonaData | None:
        return self._cache.get(persona_id)

    def count(self) -> int:
        return len(self._cache)


_STORE: PersonaStore | None = None


def get_persona_store() -> PersonaStore:
    global _STORE
    if _STORE is None:
        _STORE = PersonaStore()
    return _STORE


class SpiritAdapterStore:
    def __init__(self) -> None:
        self._runtime: Any = None
        self._sources: dict[str, SpiritPromptSource] = {}
        self._lock = threading.Lock()

    def has_adapter(self, persona_id: str) -> bool:
        return (spirit_adapter_dir(persona_id) / "adapter_config.json").exists() and (
            spirit_file_path(persona_id).exists()
        )

    def reply(
        self,
        persona_id: str,
        user_message: str,
        love_level: int,
        previous_spirit_text: str | None,
        history: list[dict[str, str]] | None = None,
    ) -> str:
        if not self.has_adapter(persona_id):
            raise EvaiError(f"No trained adapter available for registered spirit '{persona_id}'")
        from adapter.spirit_adapter import SpiritRuntime

        with self._lock:
            source = self._sources.get(persona_id)
            if source is None:
                source = load_spirit_prompt_source(persona_id)
                self._sources[persona_id] = source
            if self._runtime is None:
                self._runtime = SpiritRuntime([persona_id])
            messages = build_chat_messages(
                source, user_message, love_level, previous_spirit_text,
                conversation_history=history,
            )
            return self._runtime.reply(persona_id, messages)


_ADAPTER_STORE = SpiritAdapterStore()


def get_adapter_store() -> SpiritAdapterStore:
    return _ADAPTER_STORE


def previous_spirit_message(history: list[dict[str, Any]], persona_name: str) -> str | None:
    for turn in reversed(history[:-1] if history else []):
        if turn.get("role") == "assistant" and turn.get("sender") in (None, persona_name):
            content = str(turn.get("content", "")).strip()
            return content or None
        if turn.get("role") == "user":
            return None
    return None


def rule_based_reply(persona: PersonaData, user_message: str) -> str:
    cleaned = user_message.strip()
    lowered = cleaned.lower()

    if any(
        k in lowered
        for k in ("안녕", "반가", "처음", "하이", "hello", "hi", "좋은 아침", "잘 있었")
    ):
        if persona.personality.greeting:
            return persona.personality.greeting
        if persona.speech_patterns:
            return persona.speech_patterns[0]

    if any(k in lowered for k in ("좋아하", "좋아해", "취향", "선호", "like")):
        if persona.profile.like:
            likes_str = ", ".join(persona.profile.like)
            return f"내가 좋아하는 건 바로… {likes_str}! 구원자님도 혹시 마음에 드시나요?"

    if any(k in lowered for k in ("싫어하", "싫어해", "별로", "dislike")):
        if persona.profile.dislike:
            dislikes_str = ", ".join(persona.profile.dislike)
            return f"으음… 내가 꺼려하는 건 {dislikes_str} 쪽이야. 그건 좀 피하고 싶달까?"

    if any(k in lowered for k in ("취미", "특기", "뭐해", "잘해", "hobby")):
        items = [*persona.profile.hobby, *persona.profile.speciality]
        if items:
            sample = ", ".join(items)
            return f"내 특기나 취미를 꼽자면 {sample} 같은 거지! 궁금하면 내가 보여줄까?"

    if any(k in lowered for k in ("누구", "소개", "이름", "자기소개")):
        desc = persona.personality.description or ""
        first_p = desc.split("\n\n")[0] if desc else f"나는 {persona.name}! 기억해 줘."
        return first_p

    if persona.speech_patterns:
        words = re.findall(r"[\w가-힣]+", cleaned)
        best_match: str | None = None
        best_overlap = 0

        for pattern in persona.speech_patterns:
            overlap = sum(1 for w in words if w in pattern)
            if overlap > best_overlap:
                best_overlap = overlap
                best_match = pattern

        if best_match and best_overlap > 0:
            return best_match

        return random.choice(persona.speech_patterns)

    if persona.personality.greeting:
        return persona.personality.greeting

    return f"…구원자님, {persona.name}(이)랑 조금 더 이야기 나눠봐요."


def generate_persona_reply(
    persona: PersonaData,
    persona_id: str,
    user_message: str,
    history: list[dict[str, Any]] | None = None,
    love_level: int = MIN_LOVE_LEVEL,
) -> str:
    turns: list[dict[str, str]] = []
    for turn in (history or [])[:-1]:
        role = turn.get("role")
        if role == "user" or (
            role == "assistant" and turn.get("sender") in (None, persona.name)
        ):
            turns.append({"role": role, "content": str(turn["content"])})
        elif role == "assistant":
            turns.clear()
    return get_adapter_store().reply(
        persona_id,
        user_message,
        love_level,
        previous_spirit_message(history or [], persona.name),
        turns,
    )


class EVAIWebChatHandler(BaseHTTPRequestHandler):
    server_version = "EVAIWebChat/1.0"

    def _send_cors_headers(self) -> None:
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")

    def _send_json(self, data: Any, status: int = HTTPStatus.OK) -> None:
        payload = json.dumps(data, ensure_ascii=False, indent=2).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self._send_cors_headers()
        self.end_headers()
        self.wfile.write(payload)

    def _send_html(self, html_content: str, status: int = HTTPStatus.OK) -> None:
        payload = html_content.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self._send_cors_headers()
        self.end_headers()
        self.wfile.write(payload)

    def _serve_static_file(self, file_path: Path) -> None:
        if not file_path.exists() or not file_path.is_file():
            self._send_error(f"Asset not found: {file_path.name}", status=HTTPStatus.NOT_FOUND)
            return

        suffix = file_path.suffix.lower()
        mime_map = {
            ".png": "image/png",
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".svg": "image/svg+xml",
            ".ico": "image/x-icon",
            ".webp": "image/webp",
            ".json": "application/json; charset=utf-8",
            ".css": "text/css; charset=utf-8",
            ".js": "application/javascript; charset=utf-8",
        }
        content_type = mime_map.get(suffix, "application/octet-stream")

        try:
            payload = file_path.read_bytes()
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Cache-Control", "public, max-age=86400")
            self._send_cors_headers()
            self.end_headers()
            self.wfile.write(payload)
        except OSError as err:
            self._send_error(
                f"Failed to read asset: {err}", status=HTTPStatus.INTERNAL_SERVER_ERROR
            )

    def _send_error(self, message: str, status: int = HTTPStatus.BAD_REQUEST) -> None:
        self._send_json({"error": message, "status": status}, status=status)

    def do_OPTIONS(self) -> None:
        self.send_response(HTTPStatus.NO_CONTENT)
        self._send_cors_headers()
        self.end_headers()

    def do_GET(self) -> None:
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path
        query = urllib.parse.parse_qs(parsed.query)

        if path in ("/", "/index.html"):
            if not INDEX_HTML_PATH.exists():
                self._send_error(
                    f"index.html not found at {INDEX_HTML_PATH}",
                    status=HTTPStatus.NOT_FOUND,
                )
                return
            html_text = INDEX_HTML_PATH.read_text(encoding="utf-8")
            self._send_html(html_text)
            return

        if path == "/api/health":
            store = get_persona_store()
            self._send_json({"status": "ok", "persona_count": store.count()})
            return

        if path == "/api/personas":
            store = get_persona_store()
            self._send_json({"count": store.count(), "personas": store.get_summaries()})
            return

        if path == "/api/persona":
            pid_list = query.get("id")
            if not pid_list:
                self._send_error("Missing required query parameter: 'id'")
                return
            persona_id = pid_list[0]
            store = get_persona_store()
            persona = store.get_persona(persona_id)
            if persona is None:
                self._send_error(f"Persona not found: '{persona_id}'", status=HTTPStatus.NOT_FOUND)
                return
            data = persona.model_dump(mode="json")
            data["assets"] = find_persona_assets(persona_id, persona.race)
            self._send_json(data)
            return

        if path == "/api/backgrounds":
            self._send_json({"backgrounds": TALK_BACKGROUND_PRESETS})
            return

        if path.startswith("/assets/"):
            rel_path = urllib.parse.unquote(path[len("/assets/"):])
            resolved_file = (ASSETS_DIR / rel_path).resolve()
            if not resolved_file.is_relative_to(ASSETS_DIR):
                self._send_error("Forbidden path traversal", status=HTTPStatus.FORBIDDEN)
                return
            self._serve_static_file(resolved_file)
            return

        self._send_error(f"Not found: {path}", status=HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path

        if path == "/api/chat":
            content_len = int(self.headers.get("Content-Length", 0))
            if content_len == 0:
                self._send_error("Empty request body")
                return

            body_bytes = self.rfile.read(content_len)
            try:
                req_data = json.loads(body_bytes.decode("utf-8"))
            except Exception as err:
                self._send_error(f"Invalid JSON payload: {err}")
                return

            persona_id = req_data.get("persona_id")
            user_message = req_data.get("message", "").strip()
            history = req_data.get("history", [])
            try:
                love_level = int(req_data.get("love_level", MIN_LOVE_LEVEL))
            except (TypeError, ValueError):
                self._send_error("'love_level' must be an integer")
                return
            if not MIN_LOVE_LEVEL <= love_level <= MAX_LOVE_LEVEL:
                self._send_error(f"'love_level' must be {MIN_LOVE_LEVEL}..{MAX_LOVE_LEVEL}")
                return

            if not persona_id:
                self._send_error("Missing 'persona_id' in JSON body")
                return
            if not user_message:
                self._send_error("Missing 'message' in JSON body")
                return

            store = get_persona_store()
            persona = store.get_persona(persona_id)
            if persona is None:
                self._send_error(f"Persona not found: '{persona_id}'", status=HTTPStatus.NOT_FOUND)
                return

            try:
                reply = generate_persona_reply(
                    persona, persona_id, user_message, history, love_level
                )
            except EvaiError as err:
                self._send_error(str(err), status=HTTPStatus.INTERNAL_SERVER_ERROR)
                return
            self._send_json(
                {
                    "persona_id": persona_id,
                    "persona_name": persona.name,
                    "user_message": user_message,
                    "reply": reply,
                }
            )
            return

        self._send_error(f"Not found: {path}", status=HTTPStatus.NOT_FOUND)

    def log_message(self, format: str, *args: Any) -> None:
        sys_msg = format % args
        print(f"[web] {self.address_string()} - {sys_msg}")


def run_server(
    host: str = "127.0.0.1",
    port: int = 8000,
    open_browser: bool = False,
) -> None:
    store = get_persona_store()
    server_address = (host, port)
    httpd = ThreadingHTTPServer(server_address, EVAIWebChatHandler)

    url = f"http://{host}:{port}/"
    print("=" * 60)
    print(" [EVAI Web Chat Server]")
    print(f" * Serving at: {url}")
    print(f" * Personas loaded: {store.count()}")
    print(" * Zero external CDN, zero external framework, pure local CSS/JS")
    print(" * Press Ctrl+C to stop.")
    print("=" * 60)

    if open_browser:
        try:
            webbrowser.open(url)
        except Exception:
            pass

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n[web] Server stopped by user.")
    finally:
        httpd.server_close()


if __name__ == "__main__":
    run_server(open_browser=True)
