"""Rotary position embedding applied to query and key in one CUDA launch per tensor.

Matches ``q*cos + rotate_half(q)*sin`` (and the same for ``k``) exactly. The backward pass
reuses the forward kernel with the sine term negated, since RoPE's Jacobian is its own
transpose up to that sign: ``d/dq (q*cos + rotate_half(q)*sin) = grad*cos - rotate_half(grad)*sin``.

``q``/``k`` are consumed with their native ``[batch, heads, seq, head_dim]`` layout — no
``.contiguous()`` copy. Qwen3 attention hands them in straight from
``q_proj(...).view(...).transpose(1, 2)``, which is non-contiguous on every axis but the
last; the CUDA ABI carries that axis's strides instead of materialising a copy first.
``cos``/``sin`` are still flattened because the rotary embedding module already produces
them contiguous, so that call is a no-op in practice.
"""

from typing import Any

import torch

from evai_kernels.device import require_same_cuda_device
from evai_kernels.exceptions import EvaiKernelsShapeError
from evai_kernels.extension import operators


def _apply(
    tensor: torch.Tensor, cos_flat: torch.Tensor, sin_flat: torch.Tensor, negate_sin: bool
) -> torch.Tensor:
    return operators().rope_apply(tensor, cos_flat, sin_flat, negate_sin)


class RopeFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any,
        q: torch.Tensor,
        k: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch, _, seq_len, head_dim = q.shape
        if cos.shape[0] != batch:
            cos = cos.expand(batch, -1, -1)
            sin = sin.expand(batch, -1, -1)
        cos_flat = cos.contiguous().view(batch * seq_len, head_dim)
        sin_flat = sin.contiguous().view(batch * seq_len, head_dim)
        q_embed = _apply(q, cos_flat, sin_flat, False)
        k_embed = _apply(k, cos_flat, sin_flat, False)
        ctx.save_for_backward(cos_flat, sin_flat)
        return q_embed, k_embed

    @staticmethod
    def backward(
        ctx: Any, grad_q: torch.Tensor, grad_k: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, None, None]:
        cos_flat, sin_flat = ctx.saved_tensors
        d_q = _apply(grad_q, cos_flat, sin_flat, True)
        d_k = _apply(grad_k, cos_flat, sin_flat, True)
        return d_q, d_k, None, None


def apply_rotary_pos_emb(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, unsqueeze_dim: int = 1
) -> tuple[torch.Tensor, torch.Tensor]:
    del unsqueeze_dim
    if q.shape[-1] != cos.shape[-1] or q.shape[-1] != k.shape[-1]:
        raise EvaiKernelsShapeError(
            f"RoPE q {tuple(q.shape)}, k {tuple(k.shape)}, "
            f"cos {tuple(cos.shape)} must share head_dim"
        )
    if (cos.shape[0] not in (1, q.shape[0])) or q.shape[2] != cos.shape[1]:
        raise EvaiKernelsShapeError(
            f"RoPE q {tuple(q.shape)} batch/seq must match cos {tuple(cos.shape)}"
        )
    require_same_cuda_device(q, k, cos, sin)
    q_embed, k_embed = RopeFunction.apply(q, k, cos, sin)
    return q_embed, k_embed
