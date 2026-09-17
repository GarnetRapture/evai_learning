"""Load, tokenize and batch the canonical multilingual curriculum."""

import hashlib
import json
import random
from typing import Any

import torch

from common.errors import EvaiError
from common.paths import DATASETS_DIR, ROSTER_FILE_NAME
from sft_dataset.storage import message_list, sft_split_path
from spirit_dataset.records import TrainingTask
from training.records import IGNORE_INDEX, SHUFFLE_POOL_BATCHES, TokenizedExample


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
        task = TrainingTask(record.get("task", TrainingTask.PERSONA_SPEECH.value))
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
                task=task.value,
                language=str(record.get("language", "ko")),
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
