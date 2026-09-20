"""Joint full-parameter SFT that writes one BF16 model at the end of the run."""

import json
import math
import os
import random
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
from transformers import get_cosine_schedule_with_warmup

from common.errors import EvaiError
from common.hashing import compute_file_sha256
from common.model_contract import (
    MAX_MODEL_BYTES,
    MODEL_ID,
    TRAINING_CONTRACT_FILE,
    WEIGHTS_SHA256,
    read_training_contract,
    verify_backbone,
)
from common.model_storage import MODEL_STAGE, commit_pending_model, model_storage_lock
from common.paths import MODEL_DIR, REPORTS_DIR
from evai_kernels.optimizer import StochasticRoundingAdamW
from inference.model_loader import load_causal_lm, load_tokenizer
from spirit_dataset.roster import roster_slugs
from training.batching import prefetched_steps
from training.config import TrainingConfig, TrainingSettings
from training.corpus import TrainingCorpus
from training.curriculum import select_correction_curriculum, training_signature
from training.data import move_batch
from training.loss import completion_token_loss
from training.records import EpochResult, TrainingReport

REPORT_INTERVAL_STEPS = 25


def _evaluate(
    model: Any, corpus: TrainingCorpus, split: str, pad: int, settings: TrainingSettings
) -> float:
    print(f"Evaluating {split}: {len(corpus.splits[split])} records", flush=True)
    model.eval()
    total = torch.zeros((), device="cuda")
    tokens = 0
    with (
        torch.inference_mode(),
        prefetched_steps(
            corpus,
            split,
            settings.batch_size,
            settings.micro_batch_size,
            settings.micro_batch_tokens,
            pad,
        ) as steps,
    ):
        for step in steps:
            for cpu_batch in step.micro_batches:
                batch = move_batch(cpu_batch, "cuda")
                loss = completion_token_loss(model, batch)
                total += loss
                del batch, loss
            tokens += step.target_count
    mean_loss = float(total.item()) / tokens
    if not math.isfinite(mean_loss):
        raise EvaiError(f"Non-finite {split} loss in joint model training")
    return mean_loss


def _save_model(model: Any, report: TrainingReport) -> Path:
    started = time.perf_counter()
    with model_storage_lock():
        commit_pending_model()
        expected_sha = report.output_weights_sha256 or report.input_weights_sha256
        if verify_backbone(MODEL_DIR) != expected_sha:
            raise EvaiError("Another model update was committed after this training run loaded")
        _write_model(model, report)
    report.model_save_seconds += time.perf_counter() - started
    return MODEL_DIR


def _write_model(model: Any, report: TrainingReport) -> None:
    stage = MODEL_STAGE
    stage.mkdir(parents=True, exist_ok=True)
    shared: dict[tuple[int, int, tuple[int, ...], tuple[int, ...]], torch.Tensor] = {}
    state = {}
    for name, tensor in model.state_dict().items():
        key = (tensor.data_ptr(), tensor.numel(), tuple(tensor.shape), tuple(tensor.stride()))
        if key not in shared:
            shared[key] = tensor.detach().to(device="cpu", dtype=torch.bfloat16)
        state[name] = shared[key]
    model.save_pretrained(
        str(stage), state_dict=state, safe_serialization=True, max_shard_size="700MB"
    )
    del state, shared
    weights = stage / "model.safetensors"
    with weights.open("r+b") as handle:
        os.fsync(handle.fileno())
    contract = {
        "training_mode": "full_sft",
        "base_model": MODEL_ID,
        "origin_sha256": WEIGHTS_SHA256,
        "input_weights_sha256": report.input_weights_sha256,
        "weights_sha256": compute_file_sha256(weights),
        "spirits": sorted(report.dataset_provenance),
        "profile_sha256": {
            slug: entry["profile_sha256"] for slug, entry in report.dataset_provenance.items()
        },
        "parameters": report.trainable_parameters,
        "storage_dtype": "bfloat16",
        "quality_approved": False,
        "completed_training_signature": report.training_signature,
    }
    marker = stage / TRAINING_CONTRACT_FILE
    pending_marker = stage / f"{TRAINING_CONTRACT_FILE}.tmp"
    with pending_marker.open("w", encoding="utf-8") as handle:
        json.dump(contract, handle, ensure_ascii=False, indent=2)
        handle.flush()
        os.fsync(handle.fileno())
    total_size = (
        weights.stat().st_size
        + pending_marker.stat().st_size
        + sum(
            path.stat().st_size
            for path in MODEL_DIR.iterdir()
            if path.is_file() and path.name not in {"model.safetensors", TRAINING_CONTRACT_FILE}
        )
    )
    if total_size >= MAX_MODEL_BYTES:
        raise EvaiError(f"Final model exceeds the 700MB contract: {total_size} bytes")
    os.replace(pending_marker, marker)
    commit_pending_model(contract["weights_sha256"])
    report.output_weights_sha256 = contract["weights_sha256"]
    for path in stage.iterdir():
        path.unlink()
    stage.rmdir()


def train_model(config: TrainingConfig) -> Path:
    if not torch.cuda.is_available() or "RTX 3070" not in torch.cuda.get_device_name(0):
        raise EvaiError("This training pipeline requires the fixed RTX 3070")
    torch.cuda.set_per_process_memory_fraction(config.training.gpu_memory_fraction, 0)
    started = time.perf_counter()
    torch.manual_seed(config.training.seed)
    rng = random.Random(config.training.seed)
    with model_storage_lock():
        input_sha = verify_backbone(MODEL_DIR)
        prior_contract = read_training_contract(MODEL_DIR)
    if (
        config.training.curriculum != "full"
        and prior_contract is None
    ):
        raise EvaiError("Dialogue correction requires the existing jointly trained model")
    tokenizer = load_tokenizer()
    if tokenizer.pad_token_id is None:
        raise EvaiError("The fixed tokenizer must define a pad token")
    slugs = roster_slugs()
    corpus = TrainingCorpus.prepare(
        slugs,
        tokenizer,
        config.training.max_length,
        config.training.token_memory_limit_mib * 2**20,
        config.training.preparation_workers,
        config.training.general_corpus,
    )
    preparation_seconds = time.perf_counter() - started
    signature = training_signature(asdict(config), corpus.provenance, corpus.partition_moves)
    if (
        prior_contract is not None
        and prior_contract.get("completed_training_signature") == signature
    ):
        print(
            "This exact curriculum is already complete on the current model; "
            "retaining weights without repeated optimizer steps or model writes. "
            "Curriculum completion does not imply response-quality approval.",
            flush=True,
        )
        return MODEL_DIR
    selection = select_correction_curriculum(
        corpus, config.training.curriculum, config.training.replay_ratio, rng,
        replay_floor=config.training.replay_floor,
    )
    if not corpus.splits["train"]:
        print(
            "No eligible training examples; retaining the current learned model.",
            flush=True,
        )
        return MODEL_DIR
    pad_token_id = tokenizer.pad_token_id
    del tokenizer
    model = load_causal_lm(device="cuda", dtype=torch.bfloat16, expected_sha=input_sha)
    model.requires_grad_(True)
    model.config.use_cache = False
    shapes = {name: tuple(value.shape) for name, value in model.named_parameters()}
    parameters = list(model.parameters())
    report = TrainingReport(
        config=asdict(config),
        model_dir=str(MODEL_DIR),
        input_weights_sha256=input_sha,
        trainable_parameters=sum(parameter.numel() for parameter in parameters),
        train_examples=len(corpus.splits["train"]),
        validation_examples=len(corpus.splits["validation"]),
        test_examples=len(corpus.splits["test"]),
        dataset_provenance=corpus.provenance,
        over_length_excluded=corpus.excluded,
        partition_moves=corpus.partition_moves,
        training_signature=signature,
        token_storage_bytes=corpus.token_ids.numel() * corpus.token_ids.element_size(),
        preparation_seconds=preparation_seconds,
        curriculum_selection=selection,
    )
    print(
        f"One model / {len(slugs)} spirits / "
        f"{report.trainable_parameters:,} trainable parameters / "
        f"{report.train_examples} train records",
        flush=True,
    )
    print(f"Over-length exclusions by split: {corpus.excluded}", flush=True)
    print(f"Curriculum selection: {selection}", flush=True)
    optimizer = StochasticRoundingAdamW(
        model,
        lr=config.optimizer.learning_rate,
        weight_decay=config.optimizer.weight_decay,
        max_grad_norm=config.optimizer.max_grad_norm,
        seed=config.training.seed,
    )
    per_epoch = math.ceil(report.train_examples / config.training.batch_size)
    total_steps = per_epoch * config.training.epochs
    for group in optimizer.param_groups:
        group["initial_lr"] = config.optimizer.learning_rate
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        round(total_steps * config.optimizer.warmup_ratio),
        total_steps,
    )
    torch.cuda.reset_peak_memory_stats()
    for epoch in range(1, config.training.epochs + 1):
        model.train()
        total_loss = torch.tensor(0.0, device="cuda")
        total_tokens = 0
        total_examples = 0
        interval_started = time.perf_counter()
        interval_step = report.optimizer_steps
        interval_tokens = total_tokens
        interval_wait = report.data_wait_seconds
        interval_micro_batches = report.train_micro_batches
        with prefetched_steps(
            corpus,
            "train",
            config.training.batch_size,
            config.training.micro_batch_size,
            config.training.micro_batch_tokens,
            pad_token_id,
            rng,
        ) as steps:
            while True:
                wait_started = time.perf_counter()
                step = next(steps, None)
                report.data_wait_seconds += time.perf_counter() - wait_started
                if step is None:
                    break
                count = step.target_count
                example_count = step.example_count
                report.train_micro_batches += len(step.micro_batches)
                optimizer.zero_grad()
                for cpu_batch in step.micro_batches:
                    batch = move_batch(cpu_batch, "cuda")
                    loss = completion_token_loss(model, batch, per_example=True)
                    (loss / example_count).backward()
                    total_loss += loss.detach()
                    del loss, batch
                optimizer.step()
                scheduler.step()
                optimizer.raise_if_nonfinite()
                total_tokens += count
                total_examples += example_count
                report.optimizer_steps += 1
                if report.optimizer_steps % REPORT_INTERVAL_STEPS == 0:
                    mean_loss = float(total_loss.item()) / total_examples
                    elapsed = time.perf_counter() - interval_started
                    completed = report.optimizer_steps - interval_step
                    steps_per_second = completed / elapsed
                    remaining = (total_steps - report.optimizer_steps) / steps_per_second
                    wait_ms = (report.data_wait_seconds - interval_wait) * 1000 / completed
                    token_rate = (total_tokens - interval_tokens) / elapsed
                    peak_allocated = torch.cuda.max_memory_allocated() / 2**30
                    micro_count = (report.train_micro_batches - interval_micro_batches) / completed
                    print(
                        f"epoch={epoch}/{config.training.epochs} "
                        f"step={report.optimizer_steps}/{total_steps} "
                        f"({report.optimizer_steps / total_steps:.1%}) "
                        f"eta={remaining / 60:.1f}min "
                        f"loss={mean_loss:.6f} "
                        f"steps_per_second={steps_per_second:.2f} "
                        f"target_tokens_per_second={token_rate:.1f} "
                        f"data_wait_ms={wait_ms:.3f} "
                        f"micro_batches_per_step={micro_count:.2f} "
                        f"pytorch_peak_allocated={peak_allocated:.2f}GiB",
                        flush=True,
                    )
                    interval_started = time.perf_counter()
                    interval_step = report.optimizer_steps
                    interval_tokens = total_tokens
                    interval_wait = report.data_wait_seconds
                    interval_micro_batches = report.train_micro_batches
        train_loss = float(total_loss.item()) / total_examples
        if not math.isfinite(train_loss):
            raise EvaiError("Non-finite loss in joint model training")
        validation_loss = _evaluate(model, corpus, "validation", pad_token_id, config.training)
        report.epochs.append(EpochResult(epoch, train_loss, validation_loss))
        print(f"epoch={epoch} validation_loss={validation_loss:.6f}", flush=True)
    optimizer.raise_if_nonfinite()
    del scheduler, optimizer, parameters
    model.zero_grad(set_to_none=True)
    torch.cuda.empty_cache()
    report.test_loss = _evaluate(model, corpus, "test", pad_token_id, config.training)
    if {name: tuple(value.shape) for name, value in model.named_parameters()} != shapes:
        raise EvaiError("Training changed the fixed model's parameter layout")
    report.pytorch_peak_allocated_gib = torch.cuda.max_memory_allocated() / 2**30
    result = _save_model(model, report)
    report.seconds = time.perf_counter() - started
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (REPORTS_DIR / "model_training.json").write_text(
        json.dumps(report.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        f"Final single model saved: {result}; validation={report.epochs[-1].validation_loss:.6f}; "
        f"test={report.test_loss:.6f}; actual response review remains required",
        flush=True,
    )
    return result
