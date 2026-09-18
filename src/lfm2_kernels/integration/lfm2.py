"""Bind the fused short-convolution into transformers' LFM2 layers.

Uncached full-sequence forwards (training, evaluation, uncached scoring) run the fused
Triton path. Cached generation keeps the recurrent-state contract of transformers:
multi-token prefill advances the convolution state through ``update_conv_state`` and
single-token decode reduces the three taps directly against the rolled state.
"""

from types import MethodType
from typing import Any

import torch
from torch.nn import functional as F
from transformers.models.lfm2.modeling_lfm2 import Lfm2ShortConv

from shortconv_triton.exceptions import ShortConvShapeError
from shortconv_triton.functional import gated_short_conv


def _mask_padding(hidden_states: torch.Tensor, attention_mask: torch.Tensor | None) -> torch.Tensor:
    if attention_mask is None:
        return hidden_states
    return (hidden_states * attention_mask[:, :, None]).to(hidden_states.dtype)


def _cached_short_conv(
    layer: Any, projection: torch.Tensor, past_key_values: Any, length: int
) -> torch.Tensor:
    gate, carrier, value = projection.transpose(1, 2).chunk(3, dim=1)
    gated = gate * value
    layer_cache = past_key_values.layers[layer.layer_idx]
    if (
        length == 1
        and past_key_values.has_previous_state(layer.layer_idx)
        and not layer_cache.record_past
    ):
        state = layer_cache.conv_states[0]
        rolled = torch.cat((state[:, :, 1:], gated), dim=-1)
        state.copy_(rolled)
        convolved = (
            (rolled.float() * layer.conv.weight[:, 0].float())
            .sum(dim=-1, keepdim=True)
            .to(gated.dtype)
        )
        if layer.conv.bias is not None:
            convolved = convolved + layer.conv.bias[None, :, None]
    else:
        extended = past_key_values.update_conv_state(
            gated, layer.layer_idx, conv_kernel_size=layer.conv_kernel_size
        )
        convolved = F.conv1d(
            extended,
            layer.conv.weight,
            layer.conv.bias,
            padding=layer.conv_kernel_size - 1,
            groups=extended.shape[1],
        )[:, :, : extended.shape[2]][:, :, -length:]
    return (carrier * convolved.to(carrier.dtype)).transpose(1, 2).contiguous()


def lfm2_short_conv_forward(
    self: Any,
    hidden_states: torch.Tensor,
    past_key_values: Any = None,
    attention_mask: torch.Tensor | None = None,
    seq_idx: torch.Tensor | None = None,
) -> torch.Tensor:
    if seq_idx is not None:
        raise ShortConvShapeError("Packed independent sequences require separate recurrent states")
    length = hidden_states.shape[1]
    projection = self.in_proj(_mask_padding(hidden_states, attention_mask))
    if past_key_values is None:
        mixed = gated_short_conv(projection, self.conv.weight.squeeze(1), self.conv.bias)
    else:
        mixed = _cached_short_conv(self, projection, past_key_values, length)
    return self.out_proj(mixed)


def bind_fused_short_conv(model: torch.nn.Module) -> int:
    """Route every ``Lfm2ShortConv`` in ``model`` through the fused path; return the count."""
    bound = 0
    for module in model.modules():
        if isinstance(module, Lfm2ShortConv):
            module.forward = MethodType(lfm2_short_conv_forward, module)
            bound += 1
    return bound
