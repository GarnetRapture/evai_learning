import argparse

from cli.command_registry import SubParsers, add_command, print_banner
from common.errors import EvaiError
from common.paths import (
    ARTIFACT_DIR,
    CONFIG_DIR,
    DATA_DIR,
    MODEL_DIR,
    PROJECT_ROOT,
    ensure_artifact_directories,
)
from inspection.base_model_assets import inspect_local_model
from inspection.runtime_environment import inspect_environment, validate_environment
from persona.loader import discover_persona_files, inspect_persona_files
from training.config import load_training_config


def cmd_env(args: argparse.Namespace) -> int:
    print_banner("[env] Runtime and system environment check", 60)

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
    print("Result: environment check failed (see defects below):")
    for defect in defects:
        print(f"  ! {defect}")
    return 1


def cmd_data(args: argparse.Namespace) -> int:
    print_banner("[data] Full inspection of persona data (data/*.json)", 60)

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

    print(f"{'file':<26} {'id':<8} {'name':<12} {'speech':<6} {'evertalk':<10} {'story':<6}")
    print("-" * 72)
    for file_path, persona in valid_list:
        print(
            f"{file_path.name:<26} {persona.id:<8} {persona.name:<12} "
            f"{len(persona.speech_patterns):<6} {len(persona.dialogues.evertalk):<10} "
            f"{len(persona.dialogues.story):<6}"
        )

    if error_list:
        print("\n! [error details]")
        for file_path, err in error_list:
            print(f"  - file: {file_path.name}")
            print(f"    cause: {err}")
        return 1

    print("\nResult: all persona JSON data validated successfully.")
    return 0


def cmd_model(args: argparse.Namespace) -> int:
    print_banner(f"[model] Local base model asset check ({MODEL_DIR})", 60)

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
    print("Result: local base model check failed:")
    for err in status.errors:
        print(f"  ! {err}")
    return 1


def cmd_check(args: argparse.Namespace) -> int:
    print_banner("[check] Comprehensive baseline validation")

    checks_passed = True

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

    print("\n2. Artifact directories:")
    try:
        ensure_artifact_directories()
        print("  -> artifacts/{datasets,reports} directories ready.")
    except OSError as err:
        print(f"  ! Failed to create artifact directories: {err}")
        checks_passed = False

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
            f"  * training.yaml: OK (mode=full_sft, "
            f"dtype={t_cfg.training.base_dtype}, lr={t_cfg.optimizer.learning_rate}, "
            f"epochs={t_cfg.training.epochs})"
        )
    except EvaiError as err:
        print(f"  ! training.yaml validation failed: {err}")
        checks_passed = False

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

    print("\n5. Persona data integrity:")
    valid_personas, persona_errors = inspect_persona_files()
    if not valid_personas and not persona_errors:
        print("  ! No persona JSON files found.")
        checks_passed = False
    elif persona_errors:
        print(f"  ! {len(persona_errors)} persona file(s) have errors:")
        for p_file, p_err in persona_errors:
            print(f"    - {p_file.name}: {p_err}")
        checks_passed = False
    else:
        print(f"  -> OK ({len(valid_personas)} persona files fully validated)")

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
    print(" [FAIL] One or more required checks did not pass.")
    print("=" * 70)
    return 1


def register(subparsers: SubParsers) -> None:
    add_command(subparsers, "env", "Inspect Python runtime, PyTorch, CUDA, and GPU status", cmd_env)
    add_command(
        subparsers,
        "data",
        "Inspect and validate canonical persona JSON files non-recursively",
        cmd_data,
    )
    add_command(
        subparsers, "model", "Inspect local base model assets without VRAM loading", cmd_model
    )
    add_command(subparsers, "check", "Run comprehensive project baseline checks", cmd_check)
