import hashlib
import json
import math
import random
import time
from collections import Counter
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any

import torch
import yaml

from adapter.spirit_adapter import measured_lora_targets
from common.device import select_torch_device
from common.errors import ConfigurationError, EvaiError
from common.paths import (
    ADAPTERS_DIR,
    CONFIG_DIR,
    DATASETS_DIR,
    ensure_artifact_directories,
    spirit_adapter_dir,
    spirit_training_report_path,
)
from inference.generation import GenerationSettings, generate_from_messages
from inference.model_loader import load_causal_lm, load_tokenizer
from sft_dataset.storage import message_list, persona_dataset_dir, sft_split_path
from spirit_dataset.builder import ROSTER_FILE_NAME, SPIRIT_FILE_NAME
from spirit_dataset.records import TrainingTask

IGNORE_INDEX = -100
TRAIN_SPLIT = "train"
VALIDATION_SPLIT = "validation"
TEST_SPLIT = "test"
SHUFFLE_POOL_BATCHES = 16
BASE_DTYPES: dict[str, torch.dtype] = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}


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


@dataclass(frozen=True)
class TokenizedExample:
    record_id: str
    prompt: list[dict[str, str]]
    reference: str
    input_ids: list[int]
    labels: list[int]
    task: str = TrainingTask.SPEECH.value


@dataclass(frozen=True)
class PreparedSpirit:
    train: list[TokenizedExample]
    validation: list[TokenizedExample]
    test: list[TokenizedExample]
    over_length: list[str]
    source_analysis_examples: int
    dataset_provenance: dict[str, Any]
    training_tasks: dict[str, int] = field(default_factory=dict)
    supervised_tokens_by_task: dict[str, int] = field(default_factory=dict)


@dataclass(frozen=True)
class EpochResult:
    epoch: int
    train_loss: float
    validation_loss: float | None


@dataclass(frozen=True)
class SampleGeneration:
    record_id: str
    user: str
    reference: str
    generated: str


@dataclass
class SpiritLoraReport:
    slug: str
    config: dict[str, Any]
    adapter_dir: str
    train_examples: int
    validation_examples: int
    over_length_excluded: list[str]
    trainable_parameters: int
    optimizer_steps: int
    base_validation_loss: float | None
    epochs: list[EpochResult] = field(default_factory=list)
    samples: list[SampleGeneration] = field(default_factory=list)
    seconds: float = 0.0
    peak_memory_gib: float = 0.0
    rank_trials: list[dict[str, Any]] = field(default_factory=list)
    rank_selection_metric: str = ""
    test_loss: float | None = None
    source_analysis_examples: int = 0
    target_contract: str = "speech_with_auxiliary_self_memory_and_self_judgment"
    training_tasks: dict[str, int] = field(default_factory=dict)
    supervised_tokens_by_task: dict[str, int] = field(default_factory=dict)
    dataset_provenance: dict[str, Any] = field(default_factory=dict)
    phase_seconds: dict[str, float] = field(
        default_factory=lambda: {
            "prepare": 0.0,
            "train": 0.0,
            "validation": 0.0,
            "samples": 0.0,
            "save": 0.0,
        }
    )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


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
    if config.training.base_dtype not in BASE_DTYPES:
        raise ConfigurationError(
            f"Unsupported base_dtype: {config.training.base_dtype}", target_path
        )
    if rank_override is not None:
        config = replace(
            config, lora=replace(
                config.lora, rank=rank_override, alpha=rank_override * 2,
                rank_candidates=(rank_override,),
            )
        )
    if config.lora.rank <= 0 or config.training.batch_size <= 0 or config.training.epochs <= 0:
        raise ConfigurationError("rank, batch_size and epochs must be positive", target_path)
    if not 0 < config.training.micro_batch_size <= config.training.batch_size:
        raise ConfigurationError("micro_batch_size must be in 1..batch_size", target_path)
    return config


def roster_slugs() -> list[str]:
    roster_path = DATASETS_DIR / ROSTER_FILE_NAME
    if not roster_path.exists():
        raise EvaiError(f"Spirit roster not found: {roster_path}. Run `build-dataset` first.")
    roster = json.loads(roster_path.read_text(encoding="utf-8"))
    return [str(spirit["slug"]) for spirit in roster["spirits"]]


def read_split_records(slug: str, split: str) -> tuple[list[dict[str, Any]], str | None]:
    path = sft_split_path(slug, split)
    if not path.exists():
        return [], None
    raw = path.read_bytes()
    records = [json.loads(line) for line in raw.decode("utf-8").splitlines() if line.strip()]
    return records, hashlib.sha256(raw).hexdigest()


def template_ids(
    tokenizer: Any, messages: list[dict[str, str]], generation_prompt: bool
) -> list[int]:
    encoded = tokenizer.apply_chat_template(
        messages, add_generation_prompt=generation_prompt, tokenize=True, return_dict=True
    )
    return list(encoded["input_ids"])


def tokenize_records(
    tokenizer: Any, records: list[dict[str, Any]], max_length: int
) -> tuple[list[TokenizedExample], list[str]]:
    examples: list[TokenizedExample] = []
    over_length: list[str] = []
    for record in records:
        prompt = message_list(record["prompt"])
        completion = message_list(record["completion"])
        if not prompt or prompt[-1]["role"] != "user":
            raise EvaiError(f"Record requires a final user context: {record['id']}")
        if len(completion) != 1 or completion[0]["role"] != "assistant":
            raise EvaiError(f"Record requires one assistant completion: {record['id']}")
        prompt_ids = template_ids(tokenizer, prompt, generation_prompt=True)
        full_ids = template_ids(tokenizer, [*prompt, *completion], generation_prompt=False)
        if full_ids[: len(prompt_ids)] != prompt_ids or len(full_ids) <= len(prompt_ids):
            raise EvaiError(
                f"Chat template prompt is not a prefix of the full sequence: {record['id']}"
            )
        if len(full_ids) > max_length:
            over_length.append(str(record["id"]))
            continue
        labels = [IGNORE_INDEX] * len(prompt_ids) + full_ids[len(prompt_ids) :]
        examples.append(
            TokenizedExample(
                record_id=str(record["id"]),
                prompt=prompt,
                reference="\n".join(turn["content"] for turn in completion),
                input_ids=full_ids,
                labels=labels,
                task=str(record.get("task", TrainingTask.SPEECH.value)),
            )
        )
    return examples, over_length


def length_grouped_batches(
    examples: list[TokenizedExample], batch_size: int, rng: random.Random | None
) -> list[list[TokenizedExample]]:
    ordered = list(examples)
    if rng is not None:
        rng.shuffle(ordered)
    pool_size = batch_size * SHUFFLE_POOL_BATCHES
    batches: list[list[TokenizedExample]] = []
    for start in range(0, len(ordered), pool_size):
        pool = sorted(ordered[start : start + pool_size], key=lambda item: len(item.input_ids))
        batches.extend(pool[i : i + batch_size] for i in range(0, len(pool), batch_size))
    if rng is not None:
        rng.shuffle(batches)
    return batches


def collate(
    batch: list[TokenizedExample], pad_token_id: int, device: str
) -> dict[str, torch.Tensor]:
    width = max(len(item.input_ids) for item in batch)
    input_ids = torch.full((len(batch), width), pad_token_id, dtype=torch.long)
    labels = torch.full((len(batch), width), IGNORE_INDEX, dtype=torch.long)
    attention_mask = torch.zeros((len(batch), width), dtype=torch.long)
    for row, item in enumerate(batch):
        length = len(item.input_ids)
        input_ids[row, :length] = torch.tensor(item.input_ids, dtype=torch.long)
        labels[row, :length] = torch.tensor(item.labels, dtype=torch.long)
        attention_mask[row, :length] = 1
    return {
        "input_ids": input_ids.to(device),
        "labels": labels.to(device),
        "attention_mask": attention_mask.to(device),
    }


def supervised_token_count(batch: dict[str, torch.Tensor]) -> int:
    return int((batch["labels"][:, 1:] != IGNORE_INDEX).sum().item())


def completion_token_loss(
    model: Any, batch: dict[str, torch.Tensor]
) -> tuple[torch.Tensor, torch.Tensor]:
    causal_lm = model.get_base_model()
    hidden = causal_lm.model(
        input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], use_cache=False
    ).last_hidden_state
    shifted_labels = batch["labels"][:, 1:]
    supervised = shifted_labels != IGNORE_INDEX
    logits = causal_lm.lm_head(hidden[:, :-1][supervised]).float()
    loss_sum = torch.nn.functional.cross_entropy(
        logits, shifted_labels[supervised], reduction="sum"
    )
    return loss_sum, supervised.sum()


class SpiritLoraTrainer:
    def __init__(self, config: SpiritLoraConfig) -> None:
        self.config = config
        self.device = select_torch_device()
        self.tokenizer = load_tokenizer()
        if self.tokenizer.pad_token_id is None:
            raise EvaiError("Tokenizer has no pad token")
        self.base_model = load_causal_lm(
            device=self.device, dtype=BASE_DTYPES[config.training.base_dtype]
        )
        self.base_model.requires_grad_(False)
        self.target_modules = measured_lora_targets(self.base_model)
        self.autocast_dtype = (
            BASE_DTYPES[config.training.base_dtype] if self.device == "cuda" else None
        )

    def _autocast(self) -> Any:
        if self.autocast_dtype is None or self.autocast_dtype == torch.float32:
            return torch.autocast(device_type=self.device, enabled=False)
        return torch.autocast(device_type=self.device, dtype=self.autocast_dtype)

    def _lora_config(self, config: SpiritLoraConfig | None = None) -> Any:
        from peft import LoraConfig

        cfg = config if config is not None else self.config
        return LoraConfig(
            r=cfg.lora.rank,
            lora_alpha=cfg.lora.alpha,
            lora_dropout=cfg.lora.dropout,
            bias="none",
            target_modules=self.target_modules,
            task_type="CAUSAL_LM",
        )

    def _evaluate_loss(self, model: Any, examples: list[TokenizedExample]) -> float | None:
        if not examples:
            return None
        model.eval()
        total = torch.zeros((), device=self.device)
        tokens = torch.zeros((), device=self.device, dtype=torch.long)
        with torch.no_grad():
            for batch_items in length_grouped_batches(
                examples, self.config.training.micro_batch_size, None
            ):
                batch = collate(batch_items, self.tokenizer.pad_token_id, self.device)
                with self._autocast():
                    loss_sum, count = completion_token_loss(model, batch)
                total += loss_sum
                tokens += count
        token_count = int(tokens.item())
        return float(total.item()) / token_count if token_count else None

    def _samples(self, model: Any, examples: list[TokenizedExample]) -> list[SampleGeneration]:
        model.eval()
        settings = GenerationSettings(max_new_tokens=self.config.evaluation.max_new_tokens)
        samples: list[SampleGeneration] = []
        speech_examples = [item for item in examples if item.task == TrainingTask.SPEECH.value]
        for example in speech_examples[: self.config.evaluation.sample_generations]:
            with self._autocast():
                generated = generate_from_messages(
                    model, self.tokenizer, example.prompt, settings,
                )
            samples.append(
                SampleGeneration(
                    record_id=example.record_id,
                    user=example.prompt[-1]["content"],
                    reference=example.reference,
                    generated=generated,
                )
            )
        return samples

    def _backward_batch(
        self, model: Any, examples: list[TokenizedExample],
    ) -> tuple[torch.Tensor, int]:
        token_count = sum(
            label != IGNORE_INDEX for item in examples for label in item.labels[1:]
        )
        total_loss = torch.zeros((), device=self.device)
        width = self.config.training.micro_batch_size
        for start in range(0, len(examples), width):
            batch = collate(
                examples[start:start + width], self.tokenizer.pad_token_id, self.device,
            )
            with self._autocast():
                loss_sum, _ = completion_token_loss(model, batch)
            (loss_sum / token_count).backward()
            total_loss += loss_sum.detach()
        return total_loss, token_count

    def _train_candidate(
        self, slug: str, cfg: SpiritLoraConfig, prepared: PreparedSpirit,
        base_validation_loss: float | None, measure_base: bool,
    ) -> tuple[SpiritLoraReport, dict[str, torch.Tensor]]:
        from peft import get_peft_model
        from peft.utils import get_peft_model_state_dict
        from transformers import get_cosine_schedule_with_warmup

        started = time.perf_counter()
        train_examples = prepared.train
        validation_examples = prepared.validation

        torch.manual_seed(cfg.training.seed)
        rng = random.Random(cfg.training.seed)
        model = get_peft_model(self.base_model, self._lora_config(cfg), adapter_name=slug)
        try:
            trainable = [p for p in model.parameters() if p.requires_grad]
            steps_per_epoch = math.ceil(len(train_examples) / cfg.training.batch_size)
            total_steps = steps_per_epoch * cfg.training.epochs
            optimizer = torch.optim.AdamW(
                trainable,
                lr=cfg.optimizer.learning_rate,
                weight_decay=cfg.optimizer.weight_decay,
                fused=self.device == "cuda",
            )
            scheduler = get_cosine_schedule_with_warmup(
                optimizer, round(total_steps * cfg.optimizer.warmup_ratio), total_steps
            )
            report = SpiritLoraReport(
                slug=slug,
                config=asdict(cfg),
                adapter_dir=str(spirit_adapter_dir(slug)),
                train_examples=len(train_examples),
                validation_examples=len(validation_examples),
                over_length_excluded=prepared.over_length,
                trainable_parameters=sum(p.numel() for p in trainable),
                optimizer_steps=total_steps,
                base_validation_loss=(
                    self._evaluate_loss(model, validation_examples)
                    if measure_base else base_validation_loss
                ),
                source_analysis_examples=prepared.source_analysis_examples,
                dataset_provenance=prepared.dataset_provenance,
                training_tasks=prepared.training_tasks,
                supervised_tokens_by_task=prepared.supervised_tokens_by_task,
            )
            report.phase_seconds["prepare"] = time.perf_counter() - started
            for epoch in range(1, cfg.training.epochs + 1):
                phase = time.perf_counter()
                model.train()
                epoch_loss = torch.zeros((), device=self.device)
                epoch_tokens = torch.zeros((), device=self.device, dtype=torch.long)
                for batch_items in length_grouped_batches(
                    train_examples, cfg.training.batch_size, rng
                ):
                    loss_sum, token_count = self._backward_batch(model, batch_items)
                    epoch_loss += loss_sum
                    epoch_tokens += token_count
                    torch.nn.utils.clip_grad_norm_(trainable, cfg.optimizer.max_grad_norm)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
                train_loss = float(epoch_loss.item()) / max(int(epoch_tokens.item()), 1)
                if not math.isfinite(train_loss):
                    raise EvaiError(f"Non-finite loss while training '{slug}' (epoch {epoch})")
                report.phase_seconds["train"] += time.perf_counter() - phase
                phase = time.perf_counter()
                validation_loss = self._evaluate_loss(model, validation_examples)
                report.phase_seconds["validation"] += time.perf_counter() - phase
                report.epochs.append(
                    EpochResult(epoch=epoch, train_loss=train_loss, validation_loss=validation_loss)
                )
            state = {
                key: value.detach().cpu().clone()
                for key, value in get_peft_model_state_dict(model, adapter_name=slug).items()
            }
        finally:
            self.base_model = model.unload()
            self.base_model.requires_grad_(False)
            if any("lora_" in name for name, _ in self.base_model.named_parameters()):
                raise EvaiError(f"LoRA layers remained on the base model after '{slug}'")
        report.seconds = time.perf_counter() - started
        if self.device == "cuda":
            report.peak_memory_gib = torch.cuda.max_memory_allocated() / 2**30
        return report, state

    def _prepare_spirit(self, slug: str) -> PreparedSpirit:
        examples: dict[str, list[TokenizedExample]] = {}
        over_length: list[str] = []
        source_analysis_examples = 0
        split_hashes: dict[str, str | None] = {}
        profile = (persona_dataset_dir(slug) / SPIRIT_FILE_NAME).read_bytes()
        manifest = json.loads(profile)["manifest"]
        for split in (TRAIN_SPLIT, VALIDATION_SPLIT, TEST_SPLIT):
            records, split_hashes[split] = read_split_records(slug, split)
            examples[split], excluded = tokenize_records(
                self.tokenizer, records, self.config.training.max_length
            )
            over_length.extend(excluded)
            if split == TRAIN_SPLIT:
                retained_ids = {example.record_id for example in examples[split]}
                source_analysis_examples = sum(
                    str(record["id"]) in retained_ids
                    and bool(record.get("judgment", {}).get("complete"))
                    and record.get("task") == TrainingTask.SPEECH.value
                    for record in records
                )
        if not examples[TRAIN_SPLIT]:
            raise EvaiError(f"No trainable records for spirit '{slug}'")
        supervised_tokens: Counter[str] = Counter()
        for example in examples[TRAIN_SPLIT]:
            supervised_tokens[example.task] += sum(
                label != IGNORE_INDEX for label in example.labels
            )
        return PreparedSpirit(
            train=examples[TRAIN_SPLIT], validation=examples[VALIDATION_SPLIT],
            test=examples[TEST_SPLIT], over_length=over_length,
            source_analysis_examples=source_analysis_examples,
            training_tasks=dict(Counter(example.task for example in examples[TRAIN_SPLIT])),
            supervised_tokens_by_task=dict(supervised_tokens),
            dataset_provenance={
                "dataset_version": manifest["dataset_version"],
                "records_by_source_class": manifest["records_by_source_class"],
                "records_by_task": manifest["records_by_task"],
                "profile_sha256": hashlib.sha256(profile).hexdigest(),
                "splits_sha256": split_hashes,
                "source_databases_sha256": manifest["source_databases_sha256"],
                "episodic_memory_sha256": manifest["episodic_memory_sha256"],
                "judgment_sha256": manifest["judgment_sha256"],
            },
        )

    def train_spirit(self, slug: str) -> SpiritLoraReport:
        from peft import get_peft_model
        from peft.utils import set_peft_model_state_dict

        started = time.perf_counter()
        if self.device == "cuda":
            torch.cuda.reset_peak_memory_stats()
        prepared = self._prepare_spirit(slug)
        prepare_seconds = time.perf_counter() - started
        base_validation_loss: float | None = None
        best: tuple[SpiritLoraReport, dict[str, torch.Tensor], SpiritLoraConfig] | None = None
        best_score = math.inf
        trials: list[dict[str, Any]] = []
        total_optimizer_steps = 0
        phase_seconds = {"prepare": prepare_seconds, "train": 0.0, "validation": 0.0,
                         "samples": 0.0, "save": 0.0}
        for rank in sorted(set(self.config.lora.rank_candidates)):
            cfg = replace(self.config, lora=replace(self.config.lora, rank=rank, alpha=rank * 2))
            candidate, state = self._train_candidate(
                slug, cfg, prepared, base_validation_loss, measure_base=not trials
            )
            base_validation_loss = candidate.base_validation_loss
            total_optimizer_steps += candidate.optimizer_steps
            for phase_name, duration in candidate.phase_seconds.items():
                phase_seconds[phase_name] += duration
            last = candidate.epochs[-1]
            score = last.validation_loss if last.validation_loss is not None else last.train_loss
            trials.append({
                "rank": rank, "validation_loss": last.validation_loss,
                "train_loss": last.train_loss, "seconds": candidate.seconds,
                "trainable_parameters": candidate.trainable_parameters,
                "optimizer_steps": candidate.optimizer_steps,
            })
            print(f"  * {slug}: rank={rank} selection_loss={score:.6f}", flush=True)
            if score < best_score:
                best = candidate, state, cfg
                best_score = score
        if best is None:
            raise EvaiError(f"No finite rank selection result for '{slug}'")
        report, state, cfg = best
        report.phase_seconds = phase_seconds
        report.optimizer_steps = total_optimizer_steps
        report.rank_trials = trials
        report.rank_selection_metric = (
            "validation_loss" if report.epochs[-1].validation_loss is not None
            else "training_loss_no_holdout_available"
        )
        model = get_peft_model(self.base_model, self._lora_config(cfg), adapter_name=slug)
        try:
            set_peft_model_state_dict(model, state, adapter_name=slug)
            phase = time.perf_counter()
            report.test_loss = self._evaluate_loss(model, prepared.test)
            report.phase_seconds["validation"] += time.perf_counter() - phase
            phase = time.perf_counter()
            report.samples = self._samples(model, prepared.test)
            report.phase_seconds["samples"] += time.perf_counter() - phase
            phase = time.perf_counter()
            ensure_artifact_directories()
            model.save_pretrained(
                str(ADAPTERS_DIR), selected_adapters=[slug], save_embedding_layers=False
            )
            report.phase_seconds["save"] += time.perf_counter() - phase
        finally:
            self.base_model = model.unload()
            self.base_model.requires_grad_(False)
        report.seconds = time.perf_counter() - started
        if self.device == "cuda":
            report.peak_memory_gib = torch.cuda.max_memory_allocated() / 2**30
        report_path = spirit_training_report_path(slug)
        report_path.write_text(
            json.dumps(report.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return report
