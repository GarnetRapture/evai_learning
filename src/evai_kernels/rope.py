"""Rotary position embedding applied to query and key in one CUDA launch per tensor.

Matches ``q*cos + rotate_half(q)*sin`` (and the same for ``k``) exactly. The backward pass
reuses the forward kernel with the sine term negated, since RoPE's Jacobian is its own
transpose up to that sign: ``d/dq (q*cos + rotate_half(q)*sin) = grad*cos - rotate_half(grad)*sin``.
"""

from typing import Any

import torch

from lfm2_kernels.device import require_same_cuda_device
from lfm2_kernels.exceptions import Lfm2KernelsShapeError
from lfm2_kernels.extension import operators


def _apply(
    flat: torch.Tensor,
    cos_flat: torch.Tensor,
    sin_flat: torch.Tensor,
    heads: int,
    seq_len: int,
    negate_sin: bool,
) -> torch.Tensor:
    return operators().rope_apply(flat, cos_flat, sin_flat, heads, seq_len, negate_sin)


class RopeFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any,
        q: torch.Tensor,
        k: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch, heads_q, seq_len, head_dim = q.shape
        heads_k = k.shape[1]
        if cos.shape[0] != batch:
            cos = cos.expand(batch, -1, -1)
            sin = sin.expand(batch, -1, -1)
        cos_flat = cos.contiguous().view(batch * seq_len, head_dim)
        sin_flat = sin.contiguous().view(batch * seq_len, head_dim)
        q_flat = q.contiguous().view(-1, head_dim)
        k_flat = k.contiguous().view(-1, head_dim)
        q_embed = _apply(q_flat, cos_flat, sin_flat, heads_q, seq_len, False).view(q.shape)
        k_embed = _apply(k_flat, cos_flat, sin_flat, heads_k, seq_len, False).view(k.shape)
        ctx.save_for_backward(cos_flat, sin_flat)
        ctx.heads_q = heads_q
        ctx.heads_k = heads_k
        ctx.seq_len = seq_len
        ctx.q_shape = q.shape
        ctx.k_shape = k.shape
        return q_embed, k_embed

    @staticmethod
    def backward(
        ctx: Any, grad_q: torch.Tensor, grad_k: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, None, None]:
        cos_flat, sin_flat = ctx.saved_tensors
        head_dim = cos_flat.shape[-1]
        grad_q_flat = grad_q.contiguous().view(-1, head_dim)
        grad_k_flat = grad_k.contiguous().view(-1, head_dim)
        d_q_flat = _apply(grad_q_flat, cos_flat, sin_flat, ctx.heads_q, ctx.seq_len, True)
        d_k_flat = _apply(grad_k_flat, cos_flat, sin_flat, ctx.heads_k, ctx.seq_len, True)
        d_q = d_q_flat.view(ctx.q_shape)
        d_k = d_k_flat.view(ctx.k_shape)
        return d_q, d_k, None, None


def apply_rotary_pos_emb(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, unsqueeze_dim: int = 1
) -> tuple[torch.Tensor, torch.Tensor]:
    del unsqueeze_dim
    if q.shape[-1] != cos.shape[-1] or q.shape[-1] != k.shape[-1]:
        raise Lfm2KernelsShapeError(
            f"RoPE q {tuple(q.shape)}, k {tuple(k.shape)}, "
            f"cos {tuple(cos.shape)} must share head_dim"
        )
    if (cos.shape[0] not in (1, q.shape[0])) or q.shape[2] != cos.shape[1]:
        raise Lfm2KernelsShapeError(
            f"RoPE q {tuple(q.shape)} batch/seq must match cos {tuple(cos.shape)}"
        )
    require_same_cuda_device(q, k, cos, sin)
    q_embed, k_embed = RopeFunction.apply(q, k, cos, sin)
    return q_embed, k_embed
