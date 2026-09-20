"""Bind the fused RMSNorm, SwiGLU, and RoPE operators into transformers' Qwen3 layers."""

from collections.abc import Callable
from types import MethodType
from typing import Any

import torch
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.models.qwen3.modeling_qwen3 import (
    Qwen3Attention,
    Qwen3MLP,
    Qwen3RMSNorm,
    eager_attention_forward,
)

from evai_kernels.rms_norm import rms_norm
from evai_kernels.rope import apply_rotary_pos_emb
from evai_kernels.swiglu import swiglu

NORMS_PER_LAYER = 4
FINAL_NORMS = 1
MLPS_PER_LAYER = 1
ATTENTIONS_PER_LAYER = 1


def qwen3_rms_norm_forward(self: Any, hidden_states: torch.Tensor) -> torch.Tensor:
    return rms_norm(hidden_states, self.weight, self.variance_epsilon)


def qwen3_mlp_forward(self: Any, hidden_states: torch.Tensor) -> torch.Tensor:
    gate = self.gate_proj(hidden_states)
    up = self.up_proj(hidden_states)
    return self.down_proj(swiglu(gate, up))


def qwen3_attention_forward(
    self: Any,
    hidden_states: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
    attention_mask: torch.Tensor | None,
    past_key_values: Any = None,
    **kwargs: Any,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, self.head_dim)

    query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
    key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
    value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

    cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

    if past_key_values is not None:
        key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx)

    attention_interface: Callable = ALL_ATTENTION_FUNCTIONS.get_interface(
        self.config._attn_implementation, eager_attention_forward
    )

    attn_output, attn_weights = attention_interface(
        self,
        query_states,
        key_states,
        value_states,
        attention_mask,
        dropout=0.0 if not self.training else self.attention_dropout,
        scaling=self.scaling,
        sliding_window=self.sliding_window,
        **kwargs,
    )

    attn_output = attn_output.reshape(*input_shape, -1).contiguous()
    attn_output = self.o_proj(attn_output)
    return attn_output, attn_weights


def expected_rms_norms(layers: int) -> int:
    return NORMS_PER_LAYER * layers + FINAL_NORMS


def expected_mlps(layers: int) -> int:
    return MLPS_PER_LAYER * layers


def expected_attentions(layers: int) -> int:
    return ATTENTIONS_PER_LAYER * layers


def bind_fused_rms_norm(model: torch.nn.Module) -> int:
    bound = 0
    for module in model.modules():
        if isinstance(module, Qwen3RMSNorm):
            module.forward = MethodType(qwen3_rms_norm_forward, module)
            bound += 1
    return bound


def bind_fused_swiglu(model: torch.nn.Module) -> int:
    bound = 0
    for module in model.modules():
        if isinstance(module, Qwen3MLP):
            module.forward = MethodType(qwen3_mlp_forward, module)
            bound += 1
    return bound


def bind_fused_rope(model: torch.nn.Module) -> int:
    bound = 0
    for module in model.modules():
        if isinstance(module, Qwen3Attention):
            module.forward = MethodType(qwen3_attention_forward, module)
            bound += 1
    return bound
