"""Joint full-parameter SFT of one model; one final BF16 save, no checkpoints."""

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
    verify_backbone,
)
from common.paths import MODEL_DIR, REPORTS_DIR, SCRATCH_DIR
from inference.model_loader import load_causal_lm, load_tokenizer
from spirit_dataset.roster import roster_slugs
from training.batching import prefetched_steps
from training.config import TrainingConfig
from training.corpus import TrainingCorpus
from training.data import move_batch
from training.loss import completion_token_loss
from training.records import EpochResult, TrainingReport


def _evaluate(model: Any, corpus: TrainingCorpus, split: str, pad: int) -> float:
    print(f"Evaluating {split}: {len(corpus.splits[split])} records", flush=True)
    model.eval()
    total = torch.zeros((), device="cuda")
    tokens = 0
    with torch.inference_mode(), prefetched_steps(corpus, split, 1, 1, pad) as steps:
        for step in steps:
            batch = move_batch(step.micro_batches[0], "cuda")
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = completion_token_loss(model, batch)
            total += loss
            tokens += step.target_count
            del batch, loss
    mean_loss = float(total.item()) / tokens
    if not math.isfinite(mean_loss):
        raise EvaiError(f"Non-finite {split} loss in joint model training")
    return mean_loss


def _save_final(model: Any, report: TrainingReport) -> Path:
    stage = SCRATCH_DIR / "final-model"
    stage.mkdir(parents=True, exist_ok=True)
    model.to(device="cpu", dtype=torch.bfloat16)
    model.save_pretrained(str(stage), safe_serialization=True, max_shard_size="700MB")
    weights = stage / "model.safetensors"
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
    }
    marker = stage / TRAINING_CONTRACT_FILE
    marker.write_text(json.dumps(contract, ensure_ascii=False, indent=2), encoding="utf-8")
    total_size = (
        weights.stat().st_size
        + marker.stat().st_size
        + sum(
            path.stat().st_size
            for path in MODEL_DIR.iterdir()
            if path.is_file() and path.name not in {"model.safetensors", TRAINING_CONTRACT_FILE}
        )
    )
    if total_size >= MAX_MODEL_BYTES:
        raise EvaiError(f"Final model exceeds the 700MB contract: {total_size} bytes")
    os.replace(weights, MODEL_DIR / weights.name)
    os.replace(marker, MODEL_DIR / marker.name)
    for path in stage.iterdir():
        path.unlink()
    stage.rmdir()
    return MODEL_DIR


def train_model(config: TrainingConfig) -> Path:
    if not torch.cuda.is_available() or "RTX 3070" not in torch.cuda.get_device_name(0):
        raise EvaiError("This training pipeline requires the fixed RTX 3070")
    started = time.perf_counter()
    torch.manual_seed(config.training.seed)
    rng = random.Random(config.training.seed)
    input_sha = verify_backbone(MODEL_DIR)
    tokenizer = load_tokenizer()
    if tokenizer.pad_token_id is None:
        raise EvaiError("The fixed tokenizer must define a pad token")
    slugs = roster_slugs()
    corpus = TrainingCorpus.prepare(
        slugs,
        tokenizer,
        config.training.max_length,
        config.training.token_memory_limit_mib * 2**20,
    )
    preparation_seconds = time.perf_counter() - started
    pad_token_id = tokenizer.pad_token_id
    del tokenizer
    model = load_causal_lm(device="cuda", dtype=torch.float32)
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
        token_storage_bytes=corpus.token_ids.numel() * corpus.token_ids.element_size(),
        preparation_seconds=preparation_seconds,
    )
    print(
        f"One model / {len(slugs)} spirits / "
        f"{report.trainable_parameters:,} trainable parameters / "
        f"{report.train_examples} train records",
        flush=True,
    )
    print(f"Over-length exclusions by split: {corpus.excluded}", flush=True)
    optimizer = torch.optim.AdamW(
        parameters,
        lr=config.optimizer.learning_rate,
        weight_decay=config.optimizer.weight_decay,
        fused=True,
    )
    per_epoch = math.ceil(report.train_examples / config.training.batch_size)
    total_steps = per_epoch * config.training.epochs
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, round(total_steps * config.optimizer.warmup_ratio), total_steps
    )
    torch.cuda.reset_peak_memory_stats()
    for epoch in range(1, config.training.epochs + 1):
        model.train()
        total_loss = torch.zeros((), device="cuda")
        total_tokens = 0
        interval_started = time.perf_counter()
        interval_step = report.optimizer_steps
        interval_tokens = 0
        interval_wait = report.data_wait_seconds
        interval_micro_batches = report.train_micro_batches
        with prefetched_steps(
            corpus,
            "train",
            config.training.batch_size,
            config.training.micro_batch_size,
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
                report.train_micro_batches += len(step.micro_batches)
                optimizer.zero_grad(set_to_none=True)
                for cpu_batch in step.micro_batches:
                    batch = move_batch(cpu_batch, "cuda")
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        loss = completion_token_loss(model, batch)
                        normalized = loss / count
                    normalized.backward()
                    total_loss += loss.detach()
                    del loss, normalized, batch
                torch.nn.utils.clip_grad_norm_(
                    parameters, config.optimizer.max_grad_norm, error_if_nonfinite=True
                )
                optimizer.step()
                scheduler.step()
                total_tokens += count
                report.optimizer_steps += 1
                if report.optimizer_steps % 25 == 0 or report.optimizer_steps % per_epoch == 0:
                    mean_loss = float(total_loss.item()) / total_tokens
                    elapsed = time.perf_counter() - interval_started
                    completed = report.optimizer_steps - interval_step
                    wait_ms = (report.data_wait_seconds - interval_wait) * 1000 / completed
                    token_rate = (total_tokens - interval_tokens) / elapsed
                    peak_allocated = torch.cuda.max_memory_allocated() / 2**30
                    micro_count = (report.train_micro_batches - interval_micro_batches) / completed
                    print(
                        f"epoch={epoch}/{config.training.epochs} "
                        f"step={report.optimizer_steps}/{total_steps} "
                        f"loss={mean_loss:.6f} "
                        f"steps_per_second={completed / elapsed:.2f} "
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
        train_loss = float(total_loss.item()) / total_tokens
        if not math.isfinite(train_loss):
            raise EvaiError("Non-finite loss in joint model training")
        validation_loss = _evaluate(model, corpus, "validation", pad_token_id)
        report.epochs.append(EpochResult(epoch, train_loss, validation_loss))
        print(f"epoch={epoch} validation_loss={validation_loss:.6f}", flush=True)
    optimizer.zero_grad(set_to_none=True)
    del scheduler, optimizer, parameters
    torch.cuda.empty_cache()
    report.test_loss = _evaluate(model, corpus, "test", pad_token_id)
    if {name: tuple(value.shape) for name, value in model.named_parameters()} != shapes:
        raise EvaiError("Training changed the fixed model's parameter layout")
    report.pytorch_peak_allocated_gib = torch.cuda.max_memory_allocated() / 2**30
    result = _save_final(model, report)
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
