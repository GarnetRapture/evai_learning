from dataclasses import dataclass
from typing import Any

import torch


@dataclass(frozen=True)
class GenerationSettings:
    max_new_tokens: int = 128
    do_sample: bool = False
    repetition_penalty: float = 1.3
    no_repeat_ngram_size: int = 3


DEFAULT_GENERATION_SETTINGS = GenerationSettings()


def generate_reply(
    model: Any,
    tokenizer: Any,
    user_message: str,
    settings: GenerationSettings = DEFAULT_GENERATION_SETTINGS,
) -> str:
    inputs = tokenizer.apply_chat_template(
        [{"role": "user", "content": user_message}],
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    ).to(model.device)
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=settings.max_new_tokens,
            do_sample=settings.do_sample,
            repetition_penalty=settings.repetition_penalty,
            no_repeat_ngram_size=settings.no_repeat_ngram_size,
        )
    new_tokens = output_ids[0][inputs["input_ids"].shape[1] :]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
