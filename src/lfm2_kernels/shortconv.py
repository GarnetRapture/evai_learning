"""Gated short-convolution on the CUDA operators: training autograd and cached generation."""

from typing import Any

import torch

from lfm2_kernels.device import require_same_cuda_device
from lfm2_kernels.exceptions import Lfm2KernelsShapeError, Lfm2KernelsUnsupportedError
from lfm2_kernels.extension import operators


def _validate(
    projection: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None
) -> None:
    if projection.ndim != 3 or projection.shape[-1] % 3 != 0:
        raise Lfm2KernelsShapeError(
            f"projection must be [batch, tokens, 3 * channels], got {tuple(projection.shape)}"
        )
    channels = projection.shape[-1] // 3
    if weight.ndim != 2 or weight.shape[0] != channels:
        raise Lfm2KernelsShapeError(
            f"weight must be [channels, taps] with channels={channels}, got {tuple(weight.shape)}"
        )
    if bias is not None and (bias.ndim != 1 or bias.shape[0] != channels):
        raise Lfm2KernelsShapeError(
            f"bias must be [channels] with channels={channels}, got {tuple(bias.shape)}"
        )


def _kernel_parameters(
    weight: torch.Tensor, bias: torch.Tensor | None
) -> tuple[torch.Tensor, torch.Tensor | None]:
    kernel_weight = weight.detach().to(torch.float32).contiguous()
    kernel_bias = bias.detach().to(torch.float32).contiguous() if bias is not None else None
    return kernel_weight, kernel_bias


class GatedShortConvFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any, projection: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None
    ) -> torch.Tensor:
        kernel_weight, kernel_bias = _kernel_parameters(weight, bias)
        output = operators().shortconv_forward(projection, kernel_weight, kernel_bias, None)
        ctx.save_for_backward(projection, kernel_weight, kernel_bias)
        ctx.weight_dtype = weight.dtype
        ctx.bias_dtype = bias.dtype if bias is not None else None
        return output

    @staticmethod
    def backward(
        ctx: Any, grad_output: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        projection, kernel_weight, kernel_bias = ctx.saved_tensors
        grad_projection, grad_weight, grad_bias = operators().shortconv_backward(
            projection, kernel_weight, kernel_bias, grad_output.contiguous()
        )
        return (
            grad_projection,
            grad_weight.to(ctx.weight_dtype),
            grad_bias.to(ctx.bias_dtype) if kernel_bias is not None else None,
        )


def gated_short_conv(
    projection: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None = None
) -> torch.Tensor:
    """``carrier * causal_conv(gate * value)`` over ``[batch, tokens, gate|carrier|value]``."""
    _validate(projection, weight, bias)
    require_same_cuda_device(projection, weight, bias)
    result: torch.Tensor = GatedShortConvFunction.apply(projection, weight, bias)
    return result


def gated_short_conv_cached(
    projection: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    state: torch.Tensor,
    has_history: bool,
) -> torch.Tensor:
    """One generation step: convolve against ``state`` history, then roll ``state`` in place."""
    _validate(projection, weight, bias)
    require_same_cuda_device(projection, weight, bias, state)
    if torch.is_grad_enabled() and (projection.requires_grad or weight.requires_grad):
        raise Lfm2KernelsUnsupportedError("The cached generation step is inference-only")
    kernel_weight, kernel_bias = _kernel_parameters(weight, bias)
    library = operators()
    output = library.shortconv_forward(
        projection, kernel_weight, kernel_bias, state if has_history else None
    )
    library.shortconv_state_update(projection, state, has_history)
    return output
