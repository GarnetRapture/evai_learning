import argparse

from cli.command_registry import SubParsers, add_command, add_persona_id_argument, print_banner
from common.errors import EvaiError
from common.paths import GGUF_DIR, MERGED_DIR
from export.gguf_ollama import (
    GGUFExportConfig,
    OllamaModelConfig,
    export_gguf,
    register_ollama_model,
)


def cmd_export(args: argparse.Namespace) -> int:
    persona_id: str = args.persona_id
    model_dir = MERGED_DIR / persona_id
    if not model_dir.exists():
        print(f"! No trained model found for persona '{persona_id}': {model_dir}")
        print(f"  Run `garnet-evai train {persona_id}` first.")
        return 1

    print_banner(f"[export] GGUF conversion + Ollama registration: persona '{persona_id}'")

    gguf_path = GGUF_DIR / f"{persona_id}.gguf"
    try:
        export_gguf(GGUFExportConfig(model_dir=model_dir, output_path=gguf_path))
        print(f"* GGUF written: {gguf_path}")

        model_name = f"garnet-evai-{persona_id}"
        register_ollama_model(
            OllamaModelConfig(model_name=model_name, gguf_path=gguf_path),
            modelfile_dir=GGUF_DIR,
        )
        print(f"* Registered with Ollama as: {model_name}")
    except EvaiError as err:
        print(f"! Export failed: {err}")
        return 1

    print("-" * 70)
    print("Result: export complete.")
    return 0


def register(subparsers: SubParsers) -> None:
    add_persona_id_argument(
        add_command(
            subparsers,
            "export",
            "Convert a trained persona model to GGUF and register it with Ollama",
            cmd_export,
        )
    )
