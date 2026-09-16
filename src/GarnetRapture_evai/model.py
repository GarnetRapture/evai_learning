"""Inspection and validation of the local base model assets without full VRAM instantiation."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .paths import MODEL_DIR


@dataclass(frozen=True)
class ModelAssetStatus:
    """Inspection status of local base model assets."""

    model_path: Path
    directory_exists: bool
    config_valid: bool
    model_type: str | None
    vocab_size: int | None
    hidden_size: int | None
    num_layers: int | None
    tokenizer_valid: bool
    weight_files: list[str]
    total_weight_bytes: int
    is_valid: bool
    errors: list[str]

    @property
    def total_weight_gb(self) -> float:
        return round(self.total_weight_bytes / (1024**3), 2)

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_path": str(self.model_path),
            "directory_exists": self.directory_exists,
            "config_valid": self.config_valid,
            "model_type": self.model_type,
            "vocab_size": self.vocab_size,
            "hidden_size": self.hidden_size,
            "num_layers": self.num_layers,
            "tokenizer_valid": self.tokenizer_valid,
            "weight_files": self.weight_files,
            "total_weight_bytes": self.total_weight_bytes,
            "total_weight_gb": self.total_weight_gb,
            "is_valid": self.is_valid,
            "errors": self.errors,
        }


def inspect_local_model(model_dir: Path | None = None) -> ModelAssetStatus:
    """Inspect local base model directory without instantiating model weights in VRAM.

    Validates:
    - Directory existence
    - Configuration loading with local_files_only=True
    - Tokenizer loading with local_files_only=True
    - Presence of safetensors weight files
    """
    target_dir = model_dir if model_dir is not None else MODEL_DIR
    errors: list[str] = []

    if not target_dir.exists() or not target_dir.is_dir():
        return ModelAssetStatus(
            model_path=target_dir,
            directory_exists=False,
            config_valid=False,
            model_type=None,
            vocab_size=None,
            hidden_size=None,
            num_layers=None,
            tokenizer_valid=False,
            weight_files=[],
            total_weight_bytes=0,
            is_valid=False,
            errors=[f"Model directory does not exist: {target_dir}"],
        )

    # 1. Inspect weight files
    weight_files = sorted(
        [p.name for p in target_dir.glob("*.safetensors") if p.is_file()]
    )
    total_weight_bytes = sum(
        (target_dir / name).stat().st_size for name in weight_files
    )

    if not weight_files:
        errors.append("No *.safetensors weight files found in model directory.")

    # 2. Inspect config.json using AutoConfig (offline)
    config_valid = False
    model_type: str | None = None
    vocab_size: int | None = None
    hidden_size: int | None = None
    num_layers: int | None = None

    try:
        from transformers import AutoConfig

        cfg = AutoConfig.from_pretrained(str(target_dir), local_files_only=True)
        config_valid = True
        model_type = getattr(cfg, "model_type", None)
        vocab_size = getattr(cfg, "vocab_size", None)
        hidden_size = getattr(cfg, "hidden_size", None)
        num_layers = getattr(cfg, "num_hidden_layers", None)
    except Exception as err:
        errors.append(f"Failed to load AutoConfig locally: {err}")

    # 3. Inspect tokenizer using AutoTokenizer (offline)
    tokenizer_valid = False
    try:
        from transformers import AutoTokenizer

        _ = AutoTokenizer.from_pretrained(str(target_dir), local_files_only=True)
        tokenizer_valid = True
    except Exception as err:
        errors.append(f"Failed to load AutoTokenizer locally: {err}")

    is_valid = len(errors) == 0 and config_valid and tokenizer_valid and len(weight_files) > 0

    return ModelAssetStatus(
        model_path=target_dir,
        directory_exists=True,
        config_valid=config_valid,
        model_type=model_type,
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        tokenizer_valid=tokenizer_valid,
        weight_files=weight_files,
        total_weight_bytes=total_weight_bytes,
        is_valid=is_valid,
        errors=errors,
    )


def validate_local_model(model_dir: Path | None = None) -> tuple[bool, list[str]]:
    """Validate local model assets and return boolean pass status and diagnostic errors."""
    status = inspect_local_model(model_dir)
    return status.is_valid, status.errors
