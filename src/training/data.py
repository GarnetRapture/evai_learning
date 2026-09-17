"""Load, tokenize and batch the canonical multilingual curriculum."""

from typing import Any

import torch

from common.errors import EvaiError
from sft_dataset.storage import message_list
from spirit_dataset.records import TrainingTask
from spirit_dataset.runtime_prompt import bind_spirit_identity
from training.records import IGNORE_INDEX, EncodedRecord, TokenizedExample, TrainingBatch


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
    examples: list[TokenizedExample] = []
    over_length: list[str] = []
    for record in records:
        task = TrainingTask(record.get("task", TrainingTask.PERSONA_SPEECH.value))
        prompt = bind_spirit_identity(message_list(record["prompt"]), spirit_id)
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
                task=task.value,
                language=str(record.get("language", "ko")),
            )
        )
    return examples, over_length


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
    target_offset = 0
    for row, item in enumerate(batch):
        tokens = token_ids.narrow(0, item.token_offset, item.token_count)
        input_ids[row, : item.token_count] = tokens
        attention_mask[row, : item.token_count] = 1
        count = item.token_count - item.prompt_tokens
        target_ids[target_offset : target_offset + count] = tokens[item.prompt_tokens :]
        target_positions[target_offset : target_offset + count] = torch.arange(
            row * width + item.prompt_tokens - 1, row * width + item.token_count - 1
        )
        target_offset += count
    return TrainingBatch(input_ids, attention_mask, target_positions, target_ids)


def move_batch(batch: TrainingBatch, device: str) -> TrainingBatch:
    return TrainingBatch(
        batch.input_ids.to(device, non_blocking=True),
        batch.attention_mask.to(device, non_blocking=True),
        batch.target_positions.to(device, non_blocking=True),
        batch.target_ids.to(device, non_blocking=True),
    )
