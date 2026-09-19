import argparse
import json
from pathlib import Path

from cli.command_registry import SubParsers, add_command
from common.model_contract import TRAINING_LANGUAGES
from export.pc import load_pc_runtime
from inference.spirit_runtime import SpiritRuntime
from spirit_dataset.roster import roster_slugs
from web_chat.server import serve_web_chat


def cmd_chat(args: argparse.Namespace) -> int:
    runtime = (
        load_pc_runtime(args.runtime, use_gguf=args.gguf)
        if args.runtime is not None
        else SpiritRuntime([args.spirit], use_gguf=args.gguf)
    )
    session = runtime.open_session(args.spirit, args.language)
    for message in args.message:
        print(
            json.dumps(
                {
                    "spirit": session.spirit_id,
                    "language": args.language,
                    "response": session.respond(message),
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
    return 0


def cmd_web_chat(args: argparse.Namespace) -> int:
    runtime = (
        load_pc_runtime(args.runtime, use_gguf=args.gguf)
        if args.runtime is not None
        else SpiritRuntime(roster_slugs(), use_gguf=args.gguf)
    )
    serve_web_chat(runtime, args.host, args.port)
    return 0


def register(subparsers: SubParsers) -> None:
    web = add_command(
        subparsers,
        "web-chat",
        "Serve a browser chat that shows each selected spirit's model input",
        cmd_web_chat,
    )
    web.add_argument("--host", default="127.0.0.1")
    web.add_argument("--port", type=int, default=8765)
    web.add_argument("--runtime", type=Path, help="The single PC runtime manifest")
    web.add_argument("--gguf", action="store_true", help="Load the trained model's GGUF export")
    parser = add_command(
        subparsers, "chat", "Speak as the platform-selected spirit using the one model", cmd_chat
    )
    parser.add_argument("--spirit", required=True)
    parser.add_argument("--runtime", type=Path, help="The single PC runtime manifest")
    parser.add_argument("--gguf", action="store_true", help="Load the trained model's GGUF export")
    parser.add_argument("--language", choices=tuple(TRAINING_LANGUAGES), default="ko")
    parser.add_argument("--message", action="append", required=True)
