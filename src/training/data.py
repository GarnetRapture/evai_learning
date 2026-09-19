"""Load, tokenize and batch the canonical multilingual curriculum."""

import hashlib
import json
from typing import Any

import torch

from common.errors import EvaiError
from sft_dataset.storage import message_list
from spirit_dataset.records import TrainingTask
from spirit_dataset.runtime_prompt import bind_training_context
from training.records import (
    IGNORE_INDEX,
    EncodedRecord,
    PreparedRecord,
    TokenizedExample,
    TrainingBatch,
)


def example_fingerprint(input_ids: list[int], prompt_tokens: int) -> str:
    material = json.dumps((prompt_tokens, input_ids), separators=(",", ":")).encode("ascii")
    return hashlib.sha256(material).hexdigest()


def template_ids(
    tokenizer: Any, messages: list[dict[str, str]], generation_prompt: bool
) -> list[int]:
    encoded = tokenizer.apply_chat_template(
        messages, add_generation_prompt=generation_prompt, tokenize=True, return_dict=True
    )
    return list(encoded["input_ids"])


def tokenize_records(
    tokenizer: Any, records: list[dict[str, Any]], max_length: int, *, spirit_id: str
) -> tuple[list[TokenizedExample], list[str]]:
    prepared = prepare_records(tokenizer, records, max_length, spirit_id=spirit_id)
    return (
        [item.example for item in prepared if item.example is not None],
        [str(item.record["id"]) for item in prepared if item.example is None],
    )


def prepare_records(
    tokenizer: Any, records: list[dict[str, Any]], max_length: int, *, spirit_id: str
) -> list[PreparedRecord]:
    prepared: list[PreparedRecord] = []
    prompts = []
    completions = []
    for record in records:
        prompt = bind_training_context(message_list(record["prompt"]), spirit_id)
        completion = message_list(record["completion"])
        if not prompt or prompt[-1]["role"] != "user":
            raise EvaiError(f"Record requires a final user context: {record['id']}")
        if len(completion) != 1 or completion[0]["role"] != "assistant":
            raise EvaiError(f"Record requires one assistant completion: {record['id']}")
        prompts.append(prompt)
        completions.append(completion)
    prompt_batch = tokenizer.apply_chat_template(
        prompts, add_generation_prompt=True, tokenize=True, return_dict=True
    )["input_ids"]
    full_batch = tokenizer.apply_chat_template(
        [[*prompt, *completion] for prompt, completion in zip(prompts, completions, strict=True)],
        add_generation_prompt=False,
        tokenize=True,
        return_dict=True,
    )["input_ids"]
    for record, prompt, completion, prompt_ids, full_ids in zip(
        records, prompts, completions, prompt_batch, full_batch, strict=True
    ):
        task = TrainingTask(record.get("task", TrainingTask.PERSONA_SPEECH.value))
        if full_ids[: len(prompt_ids)] != prompt_ids or len(full_ids) <= len(prompt_ids):
            raise EvaiError(
                f"Chat template prompt is not a prefix of the full sequence: {record['id']}"
            )
        if len(full_ids) > max_length:
            prepared.append(PreparedRecord(record, None))
            continue
        labels = [IGNORE_INDEX] * len(prompt_ids) + full_ids[len(prompt_ids) :]
        prepared.append(
            PreparedRecord(
                record,
                TokenizedExample(
                    record_id=str(record["id"]),
                    prompt=prompt,
                    reference="\n".join(turn["content"] for turn in completion),
                    input_ids=full_ids,
                    labels=labels,
                    task=task.value,
                    language=str(record.get("language", "ko")),
                ),
            )
        )
    return prepared


def collate(
    batch: list[EncodedRecord], token_ids: torch.Tensor, pad_token_id: int
) -> TrainingBatch:
    """Prepare pinned CPU tensors and causal target positions before GPU submission."""
    width = max(item.token_count for item in batch)
    shape = (len(batch), width)
    target_count = sum(item.token_count - item.prompt_tokens for item in batch)
    input_ids = torch.full(shape, pad_token_id, dtype=torch.long, pin_memory=True)
    attention_mask = torch.zeros(shape, dtype=torch.long, pin_memory=True)
    target_positions = torch.empty(target_count, dtype=torch.long, pin_memory=True)
    target_ids = torch.empty(target_count, dtype=torch.long, pin_memory=True)
    target_weights = torch.empty(target_count, dtype=torch.float32, pin_memory=True)
    target_offset = 0
    for row, item in enumerate(batch):
        tokens = token_ids.narrow(0, item.token_offset, item.token_count)
        input_ids[row, : item.token_count] = tokens
        attention_mask[row, : item.token_count] = 1
        count = item.token_count - item.prompt_tokens
        target_ids[target_offset : target_offset + count] = tokens[item.prompt_tokens :]
        target_weights[target_offset : target_offset + count] = 1.0 / count
        target_positions[target_offset : target_offset + count] = torch.arange(
            row * width + item.prompt_tokens - 1, row * width + item.token_count - 1
        )
        target_offset += count
    return TrainingBatch(input_ids, attention_mask, target_positions, target_ids, target_weights)


def move_batch(batch: TrainingBatch, device: str) -> TrainingBatch:
    return TrainingBatch(
        batch.input_ids.to(device, non_blocking=True),
        batch.attention_mask.to(device, non_blocking=True),
        batch.target_positions.to(device, non_blocking=True),
        batch.target_ids.to(device, non_blocking=True),
        batch.target_weights.to(device, non_blocking=True),
    )
