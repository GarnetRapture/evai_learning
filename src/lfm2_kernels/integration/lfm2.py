"""Bind the CUDA short-convolution operators into transformers' LFM2 layers.

Uncached full-sequence forwards run the differentiable operator. Cached generation
keeps transformers' recurrent-state contract on the same operators: the layer cache
holds ``gate * value`` for the last ``taps`` tokens, the first prefill ignores prior
state, and every later call convolves against that history and rolls it in place.
"""

from types import MethodType
from typing import Any

import torch
from transformers.models.lfm2.modeling_lfm2 import Lfm2RMSNorm, Lfm2ShortConv

from lfm2_kernels.exceptions import Lfm2KernelsShapeError, Lfm2KernelsUnsupportedError
from lfm2_kernels.rms_norm import rms_norm
from lfm2_kernels.shortconv import gated_short_conv, gated_short_conv_cached


def _mask_padding(hidden_states: torch.Tensor, attention_mask: torch.Tensor | None) -> torch.Tensor:
    if attention_mask is None:
        return hidden_states
    return (hidden_states * attention_mask[:, :, None]).to(hidden_states.dtype)


def _cached_short_conv(layer: Any, projection: torch.Tensor, past_key_values: Any) -> torch.Tensor:
    layer_cache = past_key_values.layers[layer.layer_idx]
    if layer_cache.record_past:
        raise Lfm2KernelsUnsupportedError(
            "Rollback-recording caches keep unbounded history; the operator rolls a fixed window"
        )
    if not layer_cache.is_conv_states_initialized[0]:
        layer_cache.lazy_initialization(
            conv_states=projection.new_empty(
                (projection.shape[0], layer.conv.weight.shape[0], layer.conv_kernel_size)
            ),
            state_idx=0,
            conv_kernel_size=layer.conv_kernel_size,
        )
    has_history = bool(layer_cache.has_previous_state[0])
    mixed = gated_short_conv_cached(
        projection,
        layer.conv.weight.squeeze(1),
        layer.conv.bias,
        layer_cache.conv_states[0],
        has_history,
    )
    layer_cache.has_previous_state[0] = True
    return mixed


def lfm2_short_conv_forward(
    self: Any,
    hidden_states: torch.Tensor,
    past_key_values: Any = None,
    attention_mask: torch.Tensor | None = None,
    seq_idx: torch.Tensor | None = None,
) -> torch.Tensor:
    if seq_idx is not None:
        raise Lfm2KernelsShapeError(
            "Packed independent sequences require separate recurrent states"
        )
    projection = self.in_proj(_mask_padding(hidden_states, attention_mask))
    if past_key_values is None:
        mixed = gated_short_conv(projection, self.conv.weight.squeeze(1), self.conv.bias)
    else:
        mixed = _cached_short_conv(self, projection, past_key_values)
    return self.out_proj(mixed)


def lfm2_rms_norm_forward(self: Any, hidden_states: torch.Tensor) -> torch.Tensor:
    return rms_norm(hidden_states, self.weight, self.variance_epsilon)


def bind_fused_short_conv(model: torch.nn.Module) -> int:
    """Route every ``Lfm2ShortConv`` in ``model`` through the CUDA operators; return the count."""
    bound = 0
    for module in model.modules():
        if isinstance(module, Lfm2ShortConv):
            module.forward = MethodType(lfm2_short_conv_forward, module)
            bound += 1
    return bound


def bind_fused_rms_norm(model: torch.nn.Module) -> int:
    """Route every ``Lfm2RMSNorm`` in ``model`` through the fused CUDA operator."""
    bound = 0
    for module in model.modules():
        if isinstance(module, Lfm2RMSNorm):
            module.forward = MethodType(lfm2_rms_norm_forward, module)
            bound += 1
    return bound
