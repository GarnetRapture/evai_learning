import argparse
from pathlib import Path

from cli.command_registry import SubParsers, add_command
from export.gguf import export_model_gguf
from export.pc import build_pc


def cmd_export(args: argparse.Namespace) -> int:
    print(f"PC runtime: {build_pc()}", flush=True)
    return 0


def cmd_export_gguf(args: argparse.Namespace) -> int:
    print(f"Trained model GGUF: {export_model_gguf()}", flush=True)
    return 0


def cmd_build_android(args: argparse.Namespace) -> int:
    from export.android import build_android_runtime

    print(f"Android native runtime: {build_android_runtime(args.llama_cpp, args.ndk, args.jobs)}")
    return 0


def register(subparsers: SubParsers) -> None:
    add_command(subparsers, "build-pc", "Connect PC runtime to the one trained model", cmd_export)
    add_command(subparsers, "export", "Write the single model PC runtime manifest", cmd_export)
    add_command(subparsers, "export-gguf", "Convert the one trained model to GGUF", cmd_export_gguf)
    android = add_command(
        subparsers, "build-android", "Build the deferred Android runtime", cmd_build_android
    )
    android.add_argument("--llama-cpp", type=Path, required=True)
    android.add_argument("--ndk", type=Path, required=True)
    android.add_argument("--jobs", type=int, default=4)
