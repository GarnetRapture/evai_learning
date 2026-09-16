"""Full-parameter fine-tuning configuration and execution for the persona spirit model.

Persona identity is trained directly into the base model's own weights
(plan.md SS1, SS15: "the Persona is the conversational identity of the
resulting model"), not layered on through a frozen-base LoRA adapter. LFM2.5-
230M-Base is small enough that every parameter, across every LIV convolution
and GQA layer, is updated during training on an 8GB consumer GPU.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from .errors import ConfigurationError
from .paths import CONFIG_DIR, DATASETS_DIR, MERGED_DIR, MODEL_DIR


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
    """Load and validate training.yaml configuration."""
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


def load_base_model_and_tokenizer(model_dir: Path | None = None) -> tuple[Any, Any]:
    """Load the full-precision base model and tokenizer for direct full-parameter training.

    Uses local_files_only=True: the base model must already be present at
    `model_dir` (see `fff model` for offline asset verification).
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    target_dir = model_dir if model_dir is not None else MODEL_DIR
    tokenizer = AutoTokenizer.from_pretrained(str(target_dir), local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(str(target_dir), local_files_only=True)
    return model, tokenizer


def load_persona_sft_messages(persona_id: str, split: str = "train") -> list[dict[str, Any]]:
    """Load one persona's SFT JSONL split and convert each record to a chat-message list.

    Reads `artifacts/datasets/<persona_id>/<split>.jsonl` (produced by
    `cli.py build-dataset`). Each row's prompt/completion turns are
    concatenated in order into a single `messages` list, exactly preserving
    role and content — no rewriting.
    """
    jsonl_path = DATASETS_DIR / persona_id / f"{split}.jsonl"
    if not jsonl_path.exists():
        raise ConfigurationError(
            f"SFT dataset split not found for persona '{persona_id}': {jsonl_path}. "
            "Run `build-dataset` first.",
            jsonl_path,
        )

    rows: list[dict[str, Any]] = []
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            turns = [*record["prompt"], *record["completion"]]
            messages = [{"role": t["role"], "content": t["content"]} for t in turns]
            rows.append({"messages": messages})
    return rows


def build_persona_dataset_for_training(persona_id: str, split: str = "train") -> Any:
    """Build a `datasets.Dataset` of chat messages for one persona's SFT split.

    Deferred import: `datasets` is only required at actual training time.
    """
    from datasets import Dataset

    return Dataset.from_list(load_persona_sft_messages(persona_id, split))


def train_persona_model(
    persona_id: str,
    config: TrainingPipelineConfig | None = None,
    output_dir: Path | None = None,
) -> Path:
    """Run full-parameter SFT for one persona's spirit identity on the base model.

    Every base model parameter is updated directly (plan.md SS1, SS15): this
    is not a frozen-base LoRA adapter run. The trained model is saved under
    `output_dir` (default: `artifacts/merged/<persona_id>`, since there is no
    separate adapter-merge step in full-parameter training) and that path is
    returned. Deferred import: trl/transformers are only required here, at
    actual training time.
    """
    from trl.trainer.sft_config import SFTConfig
    from trl.trainer.sft_trainer import SFTTrainer

    cfg = config if config is not None else load_training_config()
    target_output_dir = output_dir if output_dir is not None else (MERGED_DIR / persona_id)
    target_output_dir.mkdir(parents=True, exist_ok=True)

    model, tokenizer = load_base_model_and_tokenizer()
    train_dataset = build_persona_dataset_for_training(persona_id, split="train")
    eval_dataset = build_persona_dataset_for_training(persona_id, split="validation")
    # A persona with very little canonical dialogue can produce an empty
    # validation split (leakage_safe_split has nothing left to allocate).
    # SFTTrainer cannot build an eval dataloader from zero examples, so
    # evaluation is skipped rather than invented for personas this sparse.
    has_eval_data = len(eval_dataset) > 0

    # transformers>=5.2 TrainingArguments dropped warmup_ratio in favor of
    # warmup_steps; convert the configured ratio ourselves so training.yaml
    # can keep expressing warmup as a ratio of total optimizer steps.
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
