"""Assistant-only causal loss without projecting context tokens into the vocabulary."""

from typing import Any

import torch

from training.records import TrainingBatch


def completion_token_loss(
    model: Any, batch: TrainingBatch, *, per_example: bool = False
) -> torch.Tensor:
    hidden = model.model(
        input_ids=batch.input_ids, attention_mask=batch.attention_mask, use_cache=False
    ).last_hidden_state
    selected = hidden.flatten(0, 1).index_select(0, batch.target_positions)
    logits = model.lm_head(selected).float()
    token_losses = torch.nn.functional.cross_entropy(
        logits,
        batch.target_ids,
        reduction="none",
    )
    return (token_losses * batch.target_weights).sum() if per_example else token_losses.sum()
