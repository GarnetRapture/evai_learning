"""Native PyTorch execution for the locked length-three LIV convolution.

Weights, padding, gates and cache layout match LFM2.5 exactly. Single-token
decode uses a three-element reduction instead of launching grouped conv1d.
No optional kernel download or missing-package fallback is involved.
"""

from types import MethodType
from typing import Any

import torch
from torch.nn import functional as F


def short_conv_forward(
    self: Any,
    hidden_states: torch.Tensor,
    past_key_values: Any = None,
    attention_mask: torch.Tensor | None = None,
    seq_idx: Any = None,
) -> torch.Tensor:
    if seq_idx is not None:
        raise ValueError("Packed independent sequences require separate recurrent states")
    length = hidden_states.shape[1]
    if attention_mask is not None:
        hidden_states = hidden_states * attention_mask[:, :, None].to(hidden_states.dtype)
    b, c, x = self.in_proj(hidden_states).transpose(1, 2).chunk(3, dim=1)
    gated = b * x
    cached = past_key_values is not None and past_key_values.has_previous_state(self.layer_idx)
    if (
        past_key_values is not None
        and cached
        and length == 1
        and not past_key_values.layers[self.layer_idx].record_past
    ):
        state = past_key_values.layers[self.layer_idx].conv_states[0]
        updated = torch.cat((state[:, :, 1:], gated), dim=-1)
        state.copy_(updated)
        convolved = (
            (updated.float() * self.conv.weight[:, 0].float())
            .sum(
                dim=-1,
                keepdim=True,
            )
            .to(gated.dtype)
        )
        if self.conv.bias is not None:
            convolved = convolved + self.conv.bias[None, :, None]
    else:
        if past_key_values is not None:
            gated = past_key_values.update_conv_state(
                gated,
                self.layer_idx,
                conv_kernel_size=self.conv_kernel_size,
            )
        convolved = F.conv1d(
            gated,
            self.conv.weight,
            self.conv.bias,
            padding=self.conv_kernel_size - 1,
            groups=gated.shape[1],
        )[:, :, : gated.shape[2]]
        convolved = convolved[:, :, -length:]
    return self.out_proj((c * convolved.to(c.dtype)).transpose(1, 2).contiguous())


def bind_native_liv(model: Any) -> None:
    for layer in model.model.layers:
        if not layer.is_attention_layer:
            layer.conv.forward = MethodType(short_conv_forward, layer.conv)
