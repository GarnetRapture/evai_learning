"""Standalone Python HTTP server for EVAI persona web chat UI preview.

Uses only the Python standard library (http.server, json, urllib, pathlib)
with zero external web framework dependencies and zero external CDN usage.
"""

import json
import random
import re
import urllib.parse
import webbrowser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from GarnetRapture_evai.loader import discover_persona_files, load_persona_file
from GarnetRapture_evai.paths import DATA_DIR, MERGED_DIR
from GarnetRapture_evai.schema import PersonaData
from GarnetRapture_evai.train import load_base_model_and_tokenizer

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
    """Dynamically resolve asset paths for a persona from local assets directory."""
    spirits_dir = ASSETS_DIR / "eversoul-assets" / "spirits"
    avatar_url: str | None = None
    standing_url: str | None = None

    normalized_target = persona_id.replace("_", "").lower()

    if spirits_dir.exists() and spirits_dir.is_dir():
        # Find best matching folder
        matched_dir: Path | None = None
        for candidate in spirits_dir.iterdir():
            if not candidate.is_dir():
                continue
            cand_norm = candidate.name.lower()
            if cand_norm == normalized_target:
                matched_dir = candidate
                break

        if matched_dir is None:
            # Fallback: match prefix (e.g. garnet_rapture -> Garnet)
            base_prefix = persona_id.split("_")[0].lower()
            for candidate in spirits_dir.iterdir():
                if not candidate.is_dir():
                    continue
                if candidate.name.lower().startswith(base_prefix):
                    matched_dir = candidate
                    break

        if matched_dir is not None:
            # 1. Evertalk avatar icon
            evertalk_dir = matched_dir / "evertalk"
            if evertalk_dir.exists():
                pngs = list(evertalk_dir.glob("*.png"))
                if pngs:
                    rel = pngs[0].relative_to(ASSETS_DIR).as_posix()
                    avatar_url = f"/assets/{rel}"

            # Fallback avatar to icon/
            if not avatar_url:
                icon_dir = matched_dir / "icon"
                if icon_dir.exists():
                    pngs = list(icon_dir.glob("*.png"))
                    if pngs:
                        rel = pngs[0].relative_to(ASSETS_DIR).as_posix()
                        avatar_url = f"/assets/{rel}"

            # 2. Standing high-res illustration
            base_dir = matched_dir / "base"
            if base_dir.exists():
                # Prefer 1024, then 512, then any png
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

    # Race badge SVG
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
    """Thread-safe in-memory cache for loaded persona data."""

    def __init__(self, data_dir: Path | None = None) -> None:
        self.data_dir = data_dir or DATA_DIR
        self._cache: dict[str, PersonaData] = {}
        self._summaries: list[dict[str, Any]] = []
        self._load_all()

    def _load_all(self) -> None:
        """Load all persona JSON files once at server initialization."""
        files = discover_persona_files(self.data_dir)
        self._cache.clear()
        self._summaries.clear()

        for file_path in files:
            persona_id = file_path.stem
            try:
                persona = load_persona_file(file_path)
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
            except Exception as err:
                print(f"[warning] Failed to load persona {file_path.name}: {err}")

        # Deterministic sort by display name
        self._summaries.sort(key=lambda item: item["name"])

    def get_summaries(self) -> list[dict[str, Any]]:
        return self._summaries

    def get_persona(self, persona_id: str) -> PersonaData | None:
        return self._cache.get(persona_id)

    def count(self) -> int:
        return len(self._cache)


_STORE: PersonaStore | None = None


def get_persona_store() -> PersonaStore:
    """Singleton getter for the global persona store."""
    global _STORE
    if _STORE is None:
        _STORE = PersonaStore()
    return _STORE


class TrainedModelStore:
    """Single-slot cache for the currently loaded persona's fine-tuned model.

    Every persona has its own fully fine-tuned weights under
    `artifacts/merged/<persona_id>` (see docs/training_run_report.md) — there
    is no shared adapter, so switching persona means switching the whole
    model. Only one model is kept resident at a time: loading persona B
    evicts persona A's weights, which keeps this preview server usable on an
    8GB GPU regardless of how many of the 95 trained personas exist.
    """

    def __init__(self) -> None:
        self._loaded_persona_id: str | None = None
        self._model: Any = None
        self._tokenizer: Any = None

    def has_trained_model(self, persona_id: str) -> bool:
        return (MERGED_DIR / persona_id / "model.safetensors").exists()

    def get(self, persona_id: str) -> tuple[Any, Any] | None:
        """Return (model, tokenizer) for persona_id, loading/swapping as needed."""
        if not self.has_trained_model(persona_id):
            return None
        if self._loaded_persona_id != persona_id:
            self._model, self._tokenizer = load_base_model_and_tokenizer(
                MERGED_DIR / persona_id
            )
            self._loaded_persona_id = persona_id
        return self._model, self._tokenizer


_MODEL_STORE: TrainedModelStore | None = None


def get_model_store() -> TrainedModelStore:
    """Singleton getter for the global trained-model cache."""
    global _MODEL_STORE
    if _MODEL_STORE is None:
        _MODEL_STORE = TrainedModelStore()
    return _MODEL_STORE


def _generate_with_trained_model(model: Any, tokenizer: Any, user_message: str) -> str:
    """Run one inference turn through a persona's fine-tuned model.

    Uses the same chat template + decoding settings as
    `cli.py cmd_evaluate` / `tmp-claude/batch_evaluate.py`: raw-text prompts
    without the chat template collapse into repeated tokens because the
    model was trained exclusively on templated `messages` turns.
    """
    inputs = tokenizer.apply_chat_template(
        [{"role": "user", "content": user_message}],
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    )
    output_ids = model.generate(
        **inputs,
        max_new_tokens=128,
        do_sample=False,
        repetition_penalty=1.3,
        no_repeat_ngram_size=3,
    )
    new_tokens = output_ids[0][inputs["input_ids"].shape[1] :]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def _rule_based_reply(persona: PersonaData, user_message: str) -> str:
    """Deterministic fallback reply for a persona with no fine-tuned model yet.

    Only reached for personas with zero canonical source material
    (`canney`, `casper`, `irene`, `pixie` as of 2026-09-17 — see
    docs/training_run_report.md), where there is nothing to fine-tune on.
    """
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
) -> str:
    """Generate this persona's reply from its own fine-tuned model when one exists.

    Each of the 95 trained personas has fully independent fine-tuned weights
    (plan.md SS9: never merge distinct personas) — the reply always comes
    from that specific persona's model, never a shared generic model. Falls
    back to a deterministic rule-based reply only for the 4 personas with no
    canonical source material to train on.
    """
    loaded = get_model_store().get(persona_id)
    if loaded is not None:
        model, tokenizer = loaded
        return _generate_with_trained_model(model, tokenizer, user_message)
    return _rule_based_reply(persona, user_message)


class EVAIWebChatHandler(BaseHTTPRequestHandler):
    """Custom HTTP handler serving web chat UI and REST API."""

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

        # 1. Main web chat UI preview
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

        # 2. Health check
        if path == "/api/health":
            store = get_persona_store()
            self._send_json({"status": "ok", "persona_count": store.count()})
            return

        # 3. Persona list summary
        if path == "/api/personas":
            store = get_persona_store()
            self._send_json({"count": store.count(), "personas": store.get_summaries()})
            return

        # 4. Persona detailed inspection
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

        # 5. Background presets
        if path == "/api/backgrounds":
            self._send_json({"backgrounds": TALK_BACKGROUND_PRESETS})
            return

        # 6. Static asset delivery (/assets/...)
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

        # 1. Interactive chat simulation
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

            reply = generate_persona_reply(persona, persona_id, user_message, history)
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
        """Clean custom logging for HTTP requests."""
        sys_msg = format % args
        print(f"[web] {self.address_string()} - {sys_msg}")


def run_server(
    host: str = "127.0.0.1",
    port: int = 8000,
    open_browser: bool = False,
) -> None:
    """Start the Python HTTP server and block until interrupted."""
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
