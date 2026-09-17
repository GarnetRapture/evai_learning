import hashlib
import json
import math
import random
import time
from collections import Counter
from dataclasses import asdict, replace
from typing import Any

import torch

from adapter.spirit_adapter import measured_lora_targets
from common.device import select_torch_device
from common.errors import EvaiError
from common.model_contract import (
    ADAPTER_FORMAT,
    DATASET_VERSION,
    MODEL_ID,
    WEIGHTS_SHA256,
)
from common.paths import (
    ADAPTERS_DIR,
    SPIRIT_FILE_NAME,
    ensure_artifact_directories,
    spirit_adapter_dir,
    spirit_training_report_path,
)
from inference.generation import GenerationSettings, generate_from_messages
from inference.model_loader import load_causal_lm, load_tokenizer
from sft_dataset.storage import persona_dataset_dir
from spirit_dataset.records import TrainingTask
from training.config import SpiritLoraConfig
from training.data import collate, length_grouped_batches, read_split_records, tokenize_records
from training.loss import completion_token_loss
from training.records import (
    IGNORE_INDEX,
    EpochResult,
    PreparedSpirit,
    SampleGeneration,
    SpiritLoraReport,
    TokenizedExample,
)

TRAIN_SPLIT = "train"
VALIDATION_SPLIT = "validation"
TEST_SPLIT = "test"
BASE_DTYPES: dict[str, torch.dtype] = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}


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
        speech_examples = [
            item for item in examples if item.task == TrainingTask.PERSONA_SPEECH.value
        ]
        language_groups = {
            language: [item for item in speech_examples if item.language == language]
            for language in ("ko", "en", "zh_tw")
        }
        selected = [
            group[index]
            for index in range(self.config.evaluation.sample_generations)
            for group in language_groups.values()
            if index < len(group)
        ]
        for example in selected[: self.config.evaluation.sample_generations]:
            with self._autocast():
                generated = generate_from_messages(
                    model,
                    self.tokenizer,
                    example.prompt,
                    settings,
                )
            samples.append(
                SampleGeneration(
                    record_id=example.record_id,
                    user=example.prompt[-1]["content"],
                    reference=example.reference,
                    generated=generated,
                    language=example.language,
                )
            )
        return samples

    def _backward_batch(
        self,
        model: Any,
        examples: list[TokenizedExample],
    ) -> tuple[torch.Tensor, int]:
        token_count = sum(label != IGNORE_INDEX for item in examples for label in item.labels[1:])
        total_loss = torch.zeros((), device=self.device)
        width = self.config.training.micro_batch_size
        for start in range(0, len(examples), width):
            batch = collate(
                examples[start : start + width],
                self.tokenizer.pad_token_id,
                self.device,
            )
            with self._autocast():
                loss_sum, _ = completion_token_loss(model, batch)
            (loss_sum / token_count).backward()
            total_loss += loss_sum.detach()
        return total_loss, token_count

    def _train_candidate(
        self,
        slug: str,
        cfg: SpiritLoraConfig,
        prepared: PreparedSpirit,
        base_validation_loss: float | None,
        measure_base: bool,
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
                    if measure_base
                    else base_validation_loss
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
                    torch.nn.utils.clip_grad_norm_(
                        trainable,
                        cfg.optimizer.max_grad_norm,
                        error_if_nonfinite=True,
                    )
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
            report.samples = self._samples(model, validation_examples)
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
        if manifest["dataset_version"] != DATASET_VERSION:
            raise EvaiError(f"Stale curriculum for {slug}; rebuild from canonical source")
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
                    and record.get("task") == TrainingTask.PERSONA_SPEECH.value
                    for record in records
                )
        if not examples[TRAIN_SPLIT]:
            raise EvaiError(f"No trainable records for spirit '{slug}'")
        if not examples[VALIDATION_SPLIT] or not examples[TEST_SPLIT]:
            raise EvaiError(f"No independent held-out data for '{slug}' after tokenization")
        missing = {task.value for task in TrainingTask} - {
            example.task for example in examples[TRAIN_SPLIT]
        }
        if missing:
            raise EvaiError(f"Incomplete grounded curriculum for {slug}: {sorted(missing)}")
        languages = {example.language for example in examples[TRAIN_SPLIT]}
        if languages != {"ko", "en", "zh_tw"}:
            raise EvaiError(f"Incomplete simultaneous language data for {slug}: {languages}")
        supervised_tokens: Counter[str] = Counter()
        for example in examples[TRAIN_SPLIT]:
            supervised_tokens[example.task] += sum(
                label != IGNORE_INDEX for label in example.labels
            )
        return PreparedSpirit(
            train=examples[TRAIN_SPLIT],
            validation=examples[VALIDATION_SPLIT],
            test=examples[TEST_SPLIT],
            over_length=over_length,
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
        phase_seconds = {
            "prepare": prepare_seconds,
            "train": 0.0,
            "validation": 0.0,
            "samples": 0.0,
            "save": 0.0,
        }
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
            if last.validation_loss is None:
                raise EvaiError(f"Rank measurement requires held-out loss for '{slug}'")
            score = last.validation_loss
            trials.append(
                {
                    "rank": rank,
                    "validation_loss": last.validation_loss,
                    "train_loss": last.train_loss,
                    "seconds": candidate.seconds,
                    "trainable_parameters": candidate.trainable_parameters,
                    "optimizer_steps": candidate.optimizer_steps,
                    "samples": [asdict(sample) for sample in candidate.samples],
                    "semantic_review": "required",
                }
            )
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
        report.rank_selection_metric = "validation_loss"
        report.rank_selection_metric += "_provisional_pending_semantic_review"
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
            (spirit_adapter_dir(slug) / "evai_adapter.json").write_text(
                json.dumps(
                    {
                        "format": ADAPTER_FORMAT,
                        "spirit": slug,
                        "base_model": MODEL_ID,
                        "base_weights_sha256": WEIGHTS_SHA256,
                        "dataset": prepared.dataset_provenance,
                        "rank": cfg.lora.rank,
                        "rank_selection": report.rank_selection_metric,
                        "target_modules": self.target_modules,
                        "quality_approved": False,
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
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
