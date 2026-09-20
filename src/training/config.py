"""Settings for joint training of the single fixed backbone."""

from dataclasses import dataclass
from pathlib import Path

import yaml

from common.errors import ConfigurationError
from common.model_contract import CONTEXT_LENGTH
from common.paths import CONFIG_DIR, GENERAL_CORPUS_FILE


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
    micro_batch_size: int
    micro_batch_tokens: int
    epochs: int
    seed: int
    base_dtype: str
    gpu_memory_fraction: float
    token_memory_limit_mib: int
    preparation_workers: int
    curriculum: str
    replay_ratio: float
    replay_floor: int
    general_corpus: bool


@dataclass(frozen=True)
class EvaluationSettings:
    sample_generations: int
    max_new_tokens: int


@dataclass(frozen=True)
class TrainingConfig:
    optimizer: OptimizerSettings
    training: TrainingSettings
    evaluation: EvaluationSettings


def load_training_config(config_path: Path | None = None) -> TrainingConfig:
    path = config_path if config_path is not None else CONFIG_DIR / "training.yaml"
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or raw.get("training_mode") != "full_sft":
        raise ConfigurationError("training_mode must be full_sft for the single model", path)
    try:
        config = TrainingConfig(
            optimizer=OptimizerSettings(**raw["optimizer"]),
            training=TrainingSettings(**raw["training"]),
            evaluation=EvaluationSettings(**raw["evaluation"]),
        )
    except (KeyError, TypeError) as err:
        raise ConfigurationError(f"Invalid joint training settings: {err}", path) from err
    training, optimizer = config.training, config.optimizer
    if training.base_dtype != "bfloat16":
        raise ConfigurationError(
            "Training keeps one bfloat16 weight copy updated by stochastic-rounding AdamW", path
        )
    if not 0 < training.micro_batch_size <= training.batch_size or training.epochs <= 0:
        raise ConfigurationError("Invalid batch size or epoch count", path)
    if not 0 < training.max_length <= CONTEXT_LENGTH:
        raise ConfigurationError("Training length exceeds the fixed context", path)
    if training.micro_batch_tokens < training.max_length:
        raise ConfigurationError(
            "A micro batch must hold at least one maximum-length record", path
        )
    if not 0 < training.gpu_memory_fraction <= 1:
        raise ConfigurationError("GPU memory fraction must be in (0, 1]", path)
    if training.token_memory_limit_mib <= 0:
        raise ConfigurationError("Token storage requires a positive CPU memory limit", path)
    if training.preparation_workers <= 0:
        raise ConfigurationError("CPU preparation requires a positive worker count", path)
    if training.curriculum not in {
        "full", "dialogue_alignment", "knowledge_completion", "dialogue_extension",
        "dialogue_context",
    }:
        raise ConfigurationError("Unknown single-model curriculum", path)
    if not 0 < training.replay_ratio <= 1:
        raise ConfigurationError("Dialogue replay ratio must be in (0, 1]", path)
    if training.replay_floor < 0:
        raise ConfigurationError("Per spirit-language replay floor must not be negative", path)
    if training.general_corpus and not GENERAL_CORPUS_FILE.is_file():
        raise ConfigurationError(
            f"General corpus is enabled but not built: {GENERAL_CORPUS_FILE}", path
        )
    if not 0 <= optimizer.warmup_ratio <= 1 or optimizer.learning_rate <= 0:
        raise ConfigurationError("Invalid optimizer schedule", path)
    if optimizer.max_grad_norm <= 0 or optimizer.weight_decay < 0:
        raise ConfigurationError("Invalid optimizer regularization", path)
    return config
