"""Composed PyTorch definition of the gated short-convolution.

This is the mathematical contract the fused kernels are verified against. It mirrors
the ``chunk -> gate * value -> depthwise causal conv -> carrier *`` chain exactly and
is never dispatched at runtime.
"""

import torch
from torch.nn import functional as F


def gated_short_conv_reference(
    projection: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    seq_len = projection.shape[1]
    channels = projection.shape[-1] // 3
    gate, carrier, value = projection.transpose(-1, -2).chunk(3, dim=-2)
    gated = gate * value
    convolved = F.conv1d(
        gated.to(weight.dtype),
        weight.unsqueeze(1),
        bias,
        padding=weight.shape[-1] - 1,
        groups=channels,
    )[:, :, :seq_len]
    return (carrier * convolved.to(carrier.dtype)).transpose(-1, -2).contiguous()
