"""Assistant-only causal loss without projecting context tokens into the vocabulary."""

from typing import Any

import torch

from training.records import IGNORE_INDEX


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
        logits,
        shifted_labels[supervised],
        reduction="sum",
    )
    return loss_sum, supervised.sum()
