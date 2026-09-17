import argparse
import json
from pathlib import Path

from cli.command_registry import SubParsers, add_command
from common.model_contract import TRAINING_LANGUAGES
from export.pc import load_pc_runtime
from inference.spirit_runtime import SpiritRuntime


def cmd_chat(args: argparse.Namespace) -> int:
    runtime = (
        load_pc_runtime(args.runtime, use_gguf=args.gguf)
        if args.runtime is not None
        else SpiritRuntime([args.spirit], use_gguf=args.gguf)
    )
    session = runtime.open_session(args.spirit, args.language, args.love_level)
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


def register(subparsers: SubParsers) -> None:
    parser = add_command(
        subparsers, "chat", "Speak as the platform-selected spirit using the one model", cmd_chat
    )
    parser.add_argument("--spirit", required=True)
    parser.add_argument("--runtime", type=Path, help="The single PC runtime manifest")
    parser.add_argument("--gguf", action="store_true", help="Load the trained model's GGUF export")
    parser.add_argument("--language", choices=tuple(TRAINING_LANGUAGES), default="ko")
    parser.add_argument("--love-level", type=int, default=1)
    parser.add_argument("--message", action="append", required=True)
