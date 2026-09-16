"""Command-line interface for GarnetRapture_evai: env, data, model, check, build-dataset."""

import argparse
import io
import json
import sys
from pathlib import Path

from .dataset import (
    build_greeting_record,
    build_persona_dataset,
    build_speech_pattern_records,
)
from .dialogue import extract_dialogue_exchanges
from .environment import inspect_environment, validate_environment
from .errors import FFFError
from .loader import discover_persona_files, inspect_persona_files, load_persona_file
from .manifest import build_dataset_manifest
from .model import inspect_local_model
from .paths import (
    ARTIFACT_DIR,
    CONFIG_DIR,
    DATA_DIR,
    DATASETS_DIR,
    MODEL_DIR,
    PROJECT_ROOT,
    ensure_artifact_directories,
)
from .split import SplitConfig, leakage_safe_split
from .train import load_training_config

DATASET_VERSION = "0.1.0"
DATASET_LANGUAGE = "ko"


def cmd_env(args: argparse.Namespace) -> int:
    """Inspect and report concrete runtime environment and GPU availability."""
    print("=" * 60)
    print(" [env] Runtime and system environment check")
    print("=" * 60)

    report = inspect_environment()
    print(f"* Python version: {report.python_version}")
    print(f"* Python executable: {report.python_executable}")
    print(f"* Platform: {report.platform_name}")
    print(f"* PyTorch version: {report.torch_version or 'not installed'}")
    print(f"* PyTorch CUDA runtime: {report.torch_cuda_version or 'N/A'}")
    print(f"* PyTorch CUDA available: {report.cuda_available}")
    print(f"* Detected GPU count: {report.gpu_count}")

    if report.gpus:
        for gpu in report.gpus:
            print(
                f"  - [GPU {gpu.index}] {gpu.name} "
                f"(VRAM: {gpu.total_memory_gb} GB / {gpu.total_memory_bytes:,} bytes)"
            )
    else:
        print("  - No CUDA GPU detected.")

    print("\n* Required training package status:")
    for name, status in report.libraries.items():
        if status.installed:
            print(f"  - {name:<14}: {status.version}")
        else:
            print(f"  - {name:<14}: [not installed] ({status.error})")

    is_valid, defects = validate_environment(report)
    print("-" * 60)
    if is_valid:
        print("Result: all required environment and CUDA requirements are satisfied.")
        return 0
    else:
        print("Result: environment check failed (see defects below):")
        for defect in defects:
            print(f"  ! {defect}")
        return 1


def cmd_data(args: argparse.Namespace) -> int:
    """Inspect actual data/*.json persona files non-recursively."""
    print("=" * 60)
    print(" [data] Full inspection of persona data (data/*.json)")
    print("=" * 60)

    all_files = discover_persona_files()
    total_count = len(all_files)
    print(f"* Target directory: {DATA_DIR}")
    print(f"* Total JSON files found: {total_count}")

    if total_count == 0:
        print("! Warning: no JSON files found under data/.")
        return 1

    valid_list, error_list = inspect_persona_files()
    print(f"* Valid persona files: {len(valid_list)}")
    print(f"* Files with errors: {len(error_list)}")
    print("-" * 60)

    # Print summary table of valid personas
    print(f"{'file':<26} {'id':<8} {'name':<12} {'speech':<6} {'evertalk':<10} {'story':<6}")
    print("-" * 72)
    for file_path, persona in valid_list:
        fname = file_path.name
        pid = persona.id
        pname = persona.name
        speech_cnt = len(persona.speech_patterns)
        evertalk_cnt = len(persona.dialogues.evertalk)
        story_cnt = len(persona.dialogues.story)
        print(f"{fname:<26} {pid:<8} {pname:<12} {speech_cnt:<6} {evertalk_cnt:<10} {story_cnt:<6}")

    if error_list:
        print("\n! [error details]")
        for file_path, err in error_list:
            print(f"  - file: {file_path.name}")
            print(f"    cause: {err}")
        return 1

    print("\nResult: all persona JSON data validated successfully.")
    return 0


def cmd_model(args: argparse.Namespace) -> int:
    """Inspect local base model assets without VRAM weight allocation."""
    print("=" * 60)
    print(f" [model] Local base model asset check ({MODEL_DIR})")
    print("=" * 60)

    status = inspect_local_model()
    print(f"* Model directory: {status.model_path}")
    print(f"* Directory exists: {status.directory_exists}")
    print(f"* AutoConfig load status: {status.config_valid}")
    if status.config_valid:
        print(f"  - Architecture/type: {status.model_type}")
        print(f"  - Vocab size: {status.vocab_size}")
        print(f"  - Hidden size: {status.hidden_size}")
        print(f"  - Layers: {status.num_layers}")

    print(f"* AutoTokenizer load status: {status.tokenizer_valid}")
    weight_list = ", ".join(status.weight_files) if status.weight_files else "none"
    print(f"* Safetensors weight files: {len(status.weight_files)} ({weight_list})")
    print(f"* Total weight size: {status.total_weight_gb} GB ({status.total_weight_bytes:,} bytes)")

    print("-" * 60)
    if status.is_valid:
        print("Result: all offline asset checks for the local base model passed.")
        return 0
    else:
        print("Result: local base model check failed:")
        for err in status.errors:
            print(f"  ! {err}")
        return 1


def cmd_check(args: argparse.Namespace) -> int:
    """Execute comprehensive foundational validation."""
    print("=" * 70)
    print(" [check] Comprehensive baseline validation")
    print("=" * 70)

    checks_passed = True

    # 1. Project paths
    print("\n1. Project paths:")
    print(f"  * PROJECT_ROOT: {PROJECT_ROOT} (exists: {PROJECT_ROOT.exists()})")
    print(f"  * DATA_DIR: {DATA_DIR} (exists: {DATA_DIR.exists()})")
    print(f"  * MODEL_DIR: {MODEL_DIR} (exists: {MODEL_DIR.exists()})")
    print(f"  * CONFIG_DIR: {CONFIG_DIR} (exists: {CONFIG_DIR.exists()})")
    print(f"  * ARTIFACT_DIR: {ARTIFACT_DIR} (exists: {ARTIFACT_DIR.exists()})")
    if not (
        PROJECT_ROOT.exists() and DATA_DIR.exists() and MODEL_DIR.exists() and CONFIG_DIR.exists()
    ):
        print("  ! One or more core directory paths are missing.")
        checks_passed = False
    else:
        print("  -> Path structure OK.")

    # 2. Artifact directories ensure
    print("\n2. Artifact directories:")
    try:
        ensure_artifact_directories()
        print("  -> artifacts/{datasets,adapters,merged,gguf,reports} directories ready.")
    except OSError as err:
        print(f"  ! Failed to create artifact directories: {err}")
        checks_passed = False

    # 3. Configurations
    print("\n3. Project configuration files:")
    model_cfg_path = CONFIG_DIR / "model.yaml"
    training_cfg_path = CONFIG_DIR / "training.yaml"

    if not model_cfg_path.exists():
        print(f"  ! model.yaml not found: {model_cfg_path}")
        checks_passed = False
    else:
        print(f"  * model.yaml: OK ({model_cfg_path.name})")

    try:
        t_cfg = load_training_config(training_cfg_path)
        print(
            f"  * training.yaml: OK (mode={t_cfg.training_mode}, "
            f"dtype={t_cfg.precision.dtype}, lr={t_cfg.optimizer.learning_rate}, "
            f"epochs={t_cfg.training.epochs})"
        )
    except FFFError as err:
        print(f"  ! training.yaml validation failed: {err}")
        checks_passed = False

    # 4. Environment & GPU
    print("\n4. Runtime environment and GPU:")
    env_report = inspect_environment()
    env_valid, env_defects = validate_environment(env_report)
    if env_valid:
        gpu_summary = ", ".join(f"{g.name} ({g.total_memory_gb}GB)" for g in env_report.gpus)
        print(f"  -> OK (CUDA available, GPU: {gpu_summary})")
    else:
        print("  ! Environment check failed:")
        for defect in env_defects:
            print(f"    - {defect}")
        checks_passed = False

    # 5. Persona Data
    print("\n5. Persona data integrity:")
    all_persona_files = discover_persona_files()
    if not all_persona_files:
        print("  ! No persona JSON files found.")
        checks_passed = False
    else:
        valid_personas, persona_errors = inspect_persona_files()
        if persona_errors:
            print(f"  ! {len(persona_errors)} persona file(s) have errors:")
            for p_file, p_err in persona_errors:
                print(f"    - {p_file.name}: {p_err}")
            checks_passed = False
        else:
            print(f"  -> OK ({len(valid_personas)} persona files fully validated)")

    # 6. Local Base Model
    print("\n6. Local base model:")
    model_status = inspect_local_model()
    if model_status.is_valid:
        print(
            f"  -> OK (model_type={model_status.model_type}, "
            f"weights={model_status.total_weight_gb}GB, tokenizer validated)"
        )
    else:
        print("  ! Local base model check failed:")
        for m_err in model_status.errors:
            print(f"    - {m_err}")
        checks_passed = False

    print("\n" + "=" * 70)
    if checks_passed:
        print(" [PASS] Comprehensive baseline validation passed.")
        print("=" * 70)
        return 0
    else:
        print(" [FAIL] One or more required checks did not pass.")
        print("=" * 70)
        return 1


def _write_jsonl(path: Path, rows: list) -> None:
    """Write a list of to_dict()-capable records as deterministic JSONL."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row.to_dict(), ensure_ascii=False))
            f.write("\n")


def cmd_build_dataset(args: argparse.Namespace) -> int:
    """Build deterministic, classified SFT datasets for every canonical persona.

    Each persona JSON file is treated as one fully independent persona_id:
    variant files (e.g. garnet vs garnet_rapture) are never merged, matching
    their distinct in-game id/name/class (plan.md SS9, SS12).
    """
    print("=" * 70)
    print(" [build-dataset] Deterministic SFT dataset generation for every persona")
    print("=" * 70)

    ensure_artifact_directories()
    all_files = discover_persona_files()
    if not all_files:
        print("! No persona JSON files found under data/.")
        return 1

    split_config = SplitConfig()
    total_accepted = 0
    total_excluded = 0
    load_failures: list[str] = []
    no_data_personas: list[str] = []

    for file_path in all_files:
        persona_id = file_path.stem
        try:
            persona = load_persona_file(file_path)
        except (FFFError, OSError) as err:
            load_failures.append(f"{file_path.name}: load failed ({err})")
            continue

        extraction = extract_dialogue_exchanges(persona, persona_id, str(file_path))

        dialogue_records, dialogue_exclusions = build_persona_dataset(
            extraction, persona_id, DATASET_LANGUAGE, str(file_path)
        )
        speech_records, speech_exclusions = build_speech_pattern_records(
            persona.speech_patterns, persona_id, persona.name, DATASET_LANGUAGE, str(file_path)
        )
        greeting_record, greeting_exclusion = build_greeting_record(
            persona.personality.greeting, persona_id, persona.name, DATASET_LANGUAGE, str(file_path)
        )

        records = [*dialogue_records, *speech_records]
        if greeting_record is not None:
            records.append(greeting_record)

        exclusions = [*dialogue_exclusions, *speech_exclusions]
        if greeting_exclusion is not None:
            exclusions.append(greeting_exclusion)

        if not records:
            no_data_personas.append(
                f"{file_path.name}: evertalk/speech_patterns/greeting are all empty; "
                "no source material to learn from."
            )
            continue

        split = leakage_safe_split(records, split_config)
        manifest = build_dataset_manifest(
            persona_id, persona.name, file_path, DATASET_VERSION, split
        )

        persona_dir = DATASETS_DIR / persona_id
        _write_jsonl(persona_dir / "train.jsonl", split.train)
        _write_jsonl(persona_dir / "validation.jsonl", split.validation)
        _write_jsonl(persona_dir / "test.jsonl", split.test)
        _write_jsonl(persona_dir / "exclusions.jsonl", exclusions)
        (persona_dir / "manifest.json").write_text(
            json.dumps(manifest.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
        )

        total_accepted += len(records)
        total_excluded += len(exclusions)
        print(
            f"  * {persona_id:<24} accepted={len(records):<5} excluded={len(exclusions):<5} "
            f"(train={len(split.train)}/val={len(split.validation)}/test={len(split.test)})"
        )

    print("-" * 70)
    print(f"* Persona files processed: {len(all_files)}")
    print(f"* Total accepted SFT records: {total_accepted}")
    print(f"* Total excluded (traceable) records: {total_excluded}")

    if load_failures:
        print("\n! [load failures]")
        for failure in load_failures:
            print(f"  - {failure}")

    if no_data_personas:
        print("\n! [no source material — a data limitation, nothing invented]")
        for entry in no_data_personas:
            print(f"  - {entry}")

    if load_failures or no_data_personas:
        return 1

    print("\nResult: deterministic SFT dataset generation completed for every persona.")
    return 0


def cmd_train(args: argparse.Namespace) -> int:
    """Run full-parameter SFT for one persona's spirit identity on the base model."""
    from .train import train_persona_model

    persona_id: str = args.persona_id
    dataset_path = DATASETS_DIR / persona_id / "train.jsonl"
    if not dataset_path.exists():
        print(f"! No SFT dataset found for persona '{persona_id}': {dataset_path}")
        print("  Run `garnet-evai build-dataset` first.")
        return 1

    print("=" * 70)
    print(f" [train] Full-parameter fine-tuning: persona '{persona_id}'")
    print("=" * 70)

    output_dir = train_persona_model(persona_id)

    print("-" * 70)
    print(f"Result: training complete. Model saved to {output_dir}")
    return 0


def cmd_evaluate(args: argparse.Namespace) -> int:
    """Run the fixed identity-regression evaluation for one trained persona model."""
    from .evaluate import build_fixed_regression_prompts, score_response_against_prompt
    from .paths import MERGED_DIR, REPORTS_DIR
    from .train import load_base_model_and_tokenizer

    persona_id: str = args.persona_id
    model_dir = MERGED_DIR / persona_id
    if not model_dir.exists():
        print(f"! No trained model found for persona '{persona_id}': {model_dir}")
        print("  Run `garnet-evai train {persona_id}` first.")
        return 1

    print("=" * 70)
    print(f" [evaluate] Fixed identity-regression evaluation: persona '{persona_id}'")
    print("=" * 70)

    all_files = discover_persona_files()
    persona = load_persona_file(DATA_DIR / f"{persona_id}.json")
    other_names = [
        load_persona_file(fp).name
        for fp in all_files
        if fp.stem != persona_id
    ][:1]

    model, tokenizer = load_base_model_and_tokenizer(model_dir)

    def generate_fn(prompt: str) -> str:
        # Training goes through the chat template (trl.SFTTrainer applies it
        # to the `messages` column); evaluation must use the exact same
        # template, or the model never recognizes the turn boundary it was
        # trained on and degenerates into repetition.
        inputs = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
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
        return tokenizer.decode(new_tokens, skip_special_tokens=True)

    prompts = build_fixed_regression_prompts(persona.name, other_names)
    metrics = [score_response_against_prompt(generate_fn(p.prompt), p) for p in prompts]

    passed = sum(1 for m in metrics if m.score == 1.0)
    for metric in metrics:
        status = "PASS" if metric.score == 1.0 else "FAIL"
        print(f"  [{status}] {metric.category.value}: {metric.details}")

    print("-" * 70)
    print(f"Result: {passed}/{len(metrics)} regression categories passed.")

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report_path = REPORTS_DIR / f"{persona_id}_evaluation.json"
    report_path.write_text(
        json.dumps(
            {
                "persona_id": persona_id,
                "persona_name": persona.name,
                "passed": passed,
                "total": len(metrics),
                "metrics": [
                    {"category": m.category.value, "score": m.score, "details": m.details}
                    for m in metrics
                ],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Report written to {report_path}")

    return 0 if passed == len(metrics) else 1


def cmd_export(args: argparse.Namespace) -> int:
    """Convert a trained persona model to GGUF and register it with Ollama."""
    from .export import GGUFExportConfig, OllamaModelConfig, export_gguf, register_ollama_model
    from .paths import GGUF_DIR, MERGED_DIR

    persona_id: str = args.persona_id
    model_dir = MERGED_DIR / persona_id
    if not model_dir.exists():
        print(f"! No trained model found for persona '{persona_id}': {model_dir}")
        print(f"  Run `garnet-evai train {persona_id}` first.")
        return 1

    print("=" * 70)
    print(f" [export] GGUF conversion + Ollama registration: persona '{persona_id}'")
    print("=" * 70)

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
    except FFFError as err:
        print(f"! Export failed: {err}")
        return 1

    print("-" * 70)
    print("Result: export complete.")
    return 0


def cmd_adapter_fidelity(args: argparse.Namespace) -> int:
    from .paths import REPORTS_DIR
    from .spirit_adapter import measure_rank_fidelity

    persona_id: str = args.persona_id
    ranks: list[int] = args.ranks
    print("=" * 70)
    print(f" [adapter-fidelity] Full-FT vs base+LoRA logits: persona '{persona_id}'")
    print("=" * 70)

    report = measure_rank_fidelity(persona_id, ranks, max_records=args.max_records)
    print(f"* evaluated tokens: {report.token_count}")
    print(
        f"* base vs full-FT: mean KL={report.base_mean_kl_to_full:.5f}, "
        f"top-1 agreement={report.base_top1_agreement_with_full:.4f}"
    )
    for row in report.ranks:
        print(
            f"  rank {row.rank:>4}: mean KL={row.mean_kl_to_full:.5f}, "
            f"top-1 agreement={row.top1_agreement_with_full:.4f}, "
            f"retained effect={row.retained_effect:.4f}"
        )

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = REPORTS_DIR / f"{persona_id}_adapter_fidelity.json"
    out_path.write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")
    print(f"Report written to {out_path}")
    return 0


def cmd_extract_adapter(args: argparse.Namespace) -> int:
    from .spirit_adapter import extract_spirit_adapter

    persona_id: str = args.persona_id
    print("=" * 70)
    print(f" [extract-adapter] persona '{persona_id}', rank {args.rank}")
    print("=" * 70)
    out_dir = extract_spirit_adapter(persona_id, args.rank)
    total = sum(p.stat().st_size for p in out_dir.iterdir() if p.is_file())
    print(f"Result: adapter written to {out_dir} ({total / 1024 / 1024:.1f} MB)")
    return 0


def cmd_adapter_runtime_check(args: argparse.Namespace) -> int:
    from .spirit_adapter import verify_runtime_fidelity

    persona_ids: list[str] = args.persona_ids
    print("=" * 70)
    print(f" [adapter-runtime-check] single backbone + {len(persona_ids)} spirit adapters")
    print("=" * 70)
    for row in verify_runtime_fidelity(persona_ids, max_records=args.max_records):
        print(
            f"  {row.persona_id:<24} tokens={row.token_count:<5} "
            f"mean KL to full-FT={row.mean_kl_to_full:.5f} "
            f"top-1 agreement={row.top1_agreement_with_full:.4f}"
        )
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    """Launch the local EVAI web chat UI preview HTTP server."""
    from .web.server import run_server

    host = getattr(args, "host", "127.0.0.1")
    port = getattr(args, "port", 8000)
    open_browser = not getattr(args, "no_browser", False)

    try:
        run_server(host=host, port=port, open_browser=open_browser)
        return 0
    except Exception as err:
        print(f"\n[error] Failed to start web chat server: {err}", file=sys.stderr)
        return 1


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line argument parser."""
    parser = argparse.ArgumentParser(
        prog="garnet-evai",
        description="GarnetRapture_evai persona fine-tuning pipeline CLI",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # env
    subparsers.add_parser("env", help="Inspect Python runtime, PyTorch, CUDA, and GPU status")

    # data
    subparsers.add_parser(
        "data", help="Inspect and validate canonical persona JSON files non-recursively"
    )

    # model
    subparsers.add_parser("model", help="Inspect local base model assets without VRAM loading")

    # check
    subparsers.add_parser("check", help="Run comprehensive project baseline checks")

    # build-dataset
    subparsers.add_parser(
        "build-dataset",
        help="Build deterministic, classified, leakage-safe SFT datasets for every persona",
    )

    # train
    train_parser = subparsers.add_parser(
        "train",
        help="Run full-parameter SFT for one persona's spirit identity",
    )
    train_parser.add_argument("persona_id", help="Persona id (data JSON filename stem)")

    # evaluate
    evaluate_parser = subparsers.add_parser(
        "evaluate",
        help="Run the fixed identity-regression evaluation for one trained persona model",
    )
    evaluate_parser.add_argument("persona_id", help="Persona id (data JSON filename stem)")

    # export
    export_parser = subparsers.add_parser(
        "export",
        help="Convert a trained persona model to GGUF and register it with Ollama",
    )
    export_parser.add_argument("persona_id", help="Persona id (data JSON filename stem)")

    fidelity_parser = subparsers.add_parser(
        "adapter-fidelity",
        help="Measure full-FT vs base+LoRA logit fidelity across candidate ranks",
    )
    fidelity_parser.add_argument("persona_id", help="Persona id (data JSON filename stem)")
    fidelity_parser.add_argument(
        "--ranks", type=int, nargs="+", default=[16, 32, 64, 128, 256], help="Candidate ranks"
    )
    fidelity_parser.add_argument(
        "--max-records", type=int, default=16, help="Held-out SFT records to score"
    )

    extract_parser = subparsers.add_parser(
        "extract-adapter",
        help="Extract a PEFT spirit adapter from a full fine-tuned persona checkpoint",
    )
    extract_parser.add_argument("persona_id", help="Persona id (data JSON filename stem)")
    extract_parser.add_argument("--rank", type=int, required=True, help="LoRA rank")

    runtime_check_parser = subparsers.add_parser(
        "adapter-runtime-check",
        help="Load spirit adapters into one backbone and compare each against its full-FT",
    )
    runtime_check_parser.add_argument("persona_ids", nargs="+", help="Persona ids to load")
    runtime_check_parser.add_argument(
        "--max-records", type=int, default=16, help="Held-out SFT records to score"
    )

    # serve
    serve_parser = subparsers.add_parser(
        "serve",
        help="Launch the local EVAI web chat UI preview HTTP server",
    )
    serve_parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Host interface to bind the server to (default: 127.0.0.1)",
    )
    serve_parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to listen on (default: 8000)",
    )
    serve_parser.add_argument(
        "--no-browser",
        action="store_true",
        help="Do not automatically open the default web browser",
    )

    return parser


def main() -> None:
    """Main entrypoint for CLI execution."""
    # Windows consoles default stdout/stderr to the system ANSI codepage
    # (e.g. cp949), which raises UnicodeEncodeError on non-ASCII characters
    # such as an em dash. Force UTF-8 so output never crashes the process.
    for stream in (sys.stdout, sys.stderr):
        if isinstance(stream, io.TextIOWrapper):
            stream.reconfigure(encoding="utf-8")

    parser = build_parser()
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(0)

    commands = {
        "env": cmd_env,
        "data": cmd_data,
        "model": cmd_model,
        "check": cmd_check,
        "build-dataset": cmd_build_dataset,
        "train": cmd_train,
        "evaluate": cmd_evaluate,
        "export": cmd_export,
        "adapter-fidelity": cmd_adapter_fidelity,
        "extract-adapter": cmd_extract_adapter,
        "adapter-runtime-check": cmd_adapter_runtime_check,
        "serve": cmd_serve,
    }

    handler = commands.get(args.command)
    if handler is None:
        parser.print_help()
        sys.exit(1)

    try:
        exit_code = handler(args)
        sys.exit(exit_code)
    except FFFError as err:
        print(f"\n[error] {err}", file=sys.stderr)
        sys.exit(1)
