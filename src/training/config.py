"""Validated settings for the fixed per-spirit LoRA curriculum."""

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import yaml

from common.errors import ConfigurationError
from common.model_contract import CONTEXT_LENGTH
from common.paths import CONFIG_DIR

BASE_DTYPE_NAMES = {"bfloat16", "float16", "float32"}


@dataclass(frozen=True)
class LoraSettings:
    rank: int
    alpha: int
    dropout: float
    rank_candidates: tuple[int, ...] = (8, 16, 32)


@dataclass(frozen=True)
class OptimizerSettings:
    learning_rate: float
    weight_decay: float
    warmup_ratio: float
    max_grad_norm: float


@dataclass(frozen=True)
class TrainingSettings:
    max_length: int
    batch_size: int
    epochs: int
    seed: int
    base_dtype: str
    micro_batch_size: int


@dataclass(frozen=True)
class EvaluationSettings:
    sample_generations: int
    max_new_tokens: int


@dataclass(frozen=True)
class SpiritLoraConfig:
    lora: LoraSettings
    optimizer: OptimizerSettings
    training: TrainingSettings
    evaluation: EvaluationSettings


def load_spirit_lora_config(
    config_path: Path | None = None, rank_override: int | None = None
) -> SpiritLoraConfig:
    target_path = config_path if config_path is not None else CONFIG_DIR / "training.yaml"
    if not target_path.exists():
        raise ConfigurationError(f"Spirit LoRA config not found: {target_path}", target_path)
    try:
        raw: dict[str, Any] = yaml.safe_load(target_path.read_text(encoding="utf-8"))
    except yaml.YAMLError as err:
        raise ConfigurationError(f"Failed to parse spirit LoRA YAML: {err}", target_path) from err
    if "extends" in raw:
        return load_spirit_lora_config(target_path.parent / raw["extends"], rank_override)
    if raw.get("training_mode") != "lora_sft":
        raise ConfigurationError("training_mode must be lora_sft", target_path)
    try:
        config = SpiritLoraConfig(
            lora=LoraSettings(
                rank=int(raw["lora"]["rank"]),
                alpha=int(raw["lora"]["alpha"]),
                dropout=float(raw["lora"]["dropout"]),
                rank_candidates=tuple(int(r) for r in raw["lora"]["rank_candidates"]),
            ),
            optimizer=OptimizerSettings(
                learning_rate=float(raw["optimizer"]["learning_rate"]),
                weight_decay=float(raw["optimizer"]["weight_decay"]),
                warmup_ratio=float(raw["optimizer"]["warmup_ratio"]),
                max_grad_norm=float(raw["optimizer"]["max_grad_norm"]),
            ),
            training=TrainingSettings(
                max_length=int(raw["training"]["max_length"]),
                batch_size=int(raw["training"]["batch_size"]),
                epochs=int(raw["training"]["epochs"]),
                seed=int(raw["training"]["seed"]),
                base_dtype=str(raw["training"]["base_dtype"]),
                micro_batch_size=int(raw["training"]["micro_batch_size"]),
            ),
            evaluation=EvaluationSettings(
                sample_generations=int(raw["evaluation"]["sample_generations"]),
                max_new_tokens=int(raw["evaluation"]["max_new_tokens"]),
            ),
        )
    except KeyError as err:
        raise ConfigurationError(f"Missing spirit LoRA config key: {err}", target_path) from err
    except (TypeError, ValueError) as err:
        raise ConfigurationError(f"Invalid spirit LoRA config value: {err}", target_path) from err
    if config.training.base_dtype not in BASE_DTYPE_NAMES:
        raise ConfigurationError(
            f"Unsupported base_dtype: {config.training.base_dtype}", target_path
        )
    if rank_override is not None:
        config = replace(
            config,
            lora=replace(
                config.lora,
                rank=rank_override,
                alpha=rank_override * 2,
                rank_candidates=(rank_override,),
            ),
        )
    if config.lora.rank <= 0 or config.training.batch_size <= 0 or config.training.epochs <= 0:
        raise ConfigurationError("rank, batch_size and epochs must be positive", target_path)
    if not 0 < config.training.micro_batch_size <= config.training.batch_size:
        raise ConfigurationError("micro_batch_size must be in 1..batch_size", target_path)
    if not 0 < config.training.max_length <= CONTEXT_LENGTH:
        raise ConfigurationError("max_length exceeds the fixed model context", target_path)
    if not config.lora.rank_candidates or any(r <= 0 for r in config.lora.rank_candidates):
        raise ConfigurationError("rank_candidates must contain positive ranks", target_path)
    if not 0 <= config.lora.dropout < 1 or not 0 <= config.optimizer.warmup_ratio <= 1:
        raise ConfigurationError("Invalid dropout or warmup ratio", target_path)
    if config.optimizer.learning_rate <= 0 or config.optimizer.max_grad_norm <= 0:
        raise ConfigurationError("learning_rate and max_grad_norm must be positive", target_path)
    return config
