from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from common.errors import ConfigurationError
from common.paths import CONFIG_DIR, MERGED_DIR
from inference.model_loader import load_model_and_tokenizer
from sft_dataset.storage import read_split_conversations, sft_split_path


@dataclass(frozen=True)
class PrecisionConfig:
    dtype: str
    gradient_checkpointing: bool


@dataclass(frozen=True)
class OptimizerConfig:
    name: str
    learning_rate: float
    weight_decay: float
    warmup_ratio: float
    lr_scheduler_type: str


@dataclass(frozen=True)
class TrainingHyperparameters:
    max_length: int
    batch_size: int
    gradient_accumulation_steps: int
    epochs: int
    seed: int


@dataclass(frozen=True)
class TrainingPipelineConfig:
    training_mode: str
    precision: PrecisionConfig
    optimizer: OptimizerConfig
    training: TrainingHyperparameters


def load_training_config(config_path: Path | None = None) -> TrainingPipelineConfig:
    target_path = config_path if config_path is not None else (CONFIG_DIR / "training.yaml")
    if not target_path.exists():
        raise ConfigurationError(
            f"Training configuration file not found: {target_path}", target_path
        )

    try:
        raw: dict[str, Any] = yaml.safe_load(target_path.read_text(encoding="utf-8"))
    except yaml.YAMLError as err:
        raise ConfigurationError(f"Failed to parse training YAML: {err}", target_path) from err

    try:
        p_raw = raw["precision"]
        o_raw = raw["optimizer"]
        t_raw = raw["training"]

        return TrainingPipelineConfig(
            training_mode=str(raw["training_mode"]),
            precision=PrecisionConfig(
                dtype=str(p_raw["dtype"]),
                gradient_checkpointing=bool(p_raw["gradient_checkpointing"]),
            ),
            optimizer=OptimizerConfig(
                name=str(o_raw["name"]),
                learning_rate=float(o_raw["learning_rate"]),
                weight_decay=float(o_raw["weight_decay"]),
                warmup_ratio=float(o_raw["warmup_ratio"]),
                lr_scheduler_type=str(o_raw["lr_scheduler_type"]),
            ),
            training=TrainingHyperparameters(
                max_length=int(t_raw["max_length"]),
                batch_size=int(t_raw["batch_size"]),
                gradient_accumulation_steps=int(t_raw["gradient_accumulation_steps"]),
                epochs=int(t_raw["epochs"]),
                seed=int(t_raw["seed"]),
            ),
        )
    except KeyError as err:
        raise ConfigurationError(
            f"Missing required training config key: {err}", target_path
        ) from err
    except (ValueError, TypeError) as err:
        raise ConfigurationError(f"Invalid training config value: {err}", target_path) from err


def build_persona_dataset_for_training(persona_id: str, split: str = "train") -> Any:
    from datasets import Dataset

    jsonl_path = sft_split_path(persona_id, split)
    if not jsonl_path.exists():
        raise ConfigurationError(
            f"SFT dataset split not found for persona '{persona_id}': {jsonl_path}. "
            "Run `build-dataset` first.",
            jsonl_path,
        )
    return Dataset.from_list(
        [{"messages": messages} for messages in read_split_conversations(jsonl_path)]
    )


def train_persona_model(
    persona_id: str,
    config: TrainingPipelineConfig | None = None,
    output_dir: Path | None = None,
) -> Path:
    from trl.trainer.sft_config import SFTConfig
    from trl.trainer.sft_trainer import SFTTrainer

    cfg = config if config is not None else load_training_config()
    target_output_dir = output_dir if output_dir is not None else (MERGED_DIR / persona_id)
    target_output_dir.mkdir(parents=True, exist_ok=True)

    model, tokenizer = load_model_and_tokenizer()
    train_dataset = build_persona_dataset_for_training(persona_id, split="train")
    eval_dataset = build_persona_dataset_for_training(persona_id, split="validation")
    has_eval_data = len(eval_dataset) > 0

    steps_per_epoch = -(
        -len(train_dataset) // (cfg.training.batch_size * cfg.training.gradient_accumulation_steps)
    )
    total_steps = steps_per_epoch * cfg.training.epochs
    warmup_steps = round(total_steps * cfg.optimizer.warmup_ratio)

    sft_config = SFTConfig(
        output_dir=str(target_output_dir),
        per_device_train_batch_size=cfg.training.batch_size,
        gradient_accumulation_steps=cfg.training.gradient_accumulation_steps,
        num_train_epochs=cfg.training.epochs,
        max_length=cfg.training.max_length,
        learning_rate=cfg.optimizer.learning_rate,
        weight_decay=cfg.optimizer.weight_decay,
        warmup_steps=warmup_steps,
        lr_scheduler_type=cfg.optimizer.lr_scheduler_type,
        optim=cfg.optimizer.name,
        bf16=cfg.precision.dtype == "bfloat16",
        fp16=cfg.precision.dtype == "float16",
        gradient_checkpointing=cfg.precision.gradient_checkpointing,
        seed=cfg.training.seed,
        data_seed=cfg.training.seed,
        logging_steps=10,
        save_strategy="no",
        eval_strategy="epoch" if has_eval_data else "no",
        report_to=[],
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset if has_eval_data else None,
        processing_class=tokenizer,
    )
    trainer.train()
    trainer.save_model(str(target_output_dir))
    tokenizer.save_pretrained(str(target_output_dir))

    return target_output_dir
