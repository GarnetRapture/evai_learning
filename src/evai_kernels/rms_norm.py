"""Llama-style RMSNorm fused into one CUDA launch per direction.

Matches ``weight * (x * rsqrt(mean(x^2) + eps)).to(input dtype)``: the normalised value
is rounded to the input dtype before the weight multiply, exactly as the composed
PyTorch module does. Only one float per row is saved for the backward pass.
"""

from typing import Any

import torch

from lfm2_kernels.device import require_same_cuda_device
from lfm2_kernels.exceptions import Lfm2KernelsShapeError
from lfm2_kernels.extension import operators


class RmsNormFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any, input_tensor: torch.Tensor, weight: torch.Tensor, epsilon: float
    ) -> torch.Tensor:
        contiguous = input_tensor.contiguous()
        output, inverse_rms = operators().rms_norm_forward(contiguous, weight, epsilon)
        ctx.save_for_backward(contiguous, weight, inverse_rms)
        return output

    @staticmethod
    def backward(
        ctx: Any, grad_output: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, None]:
        contiguous, weight, inverse_rms = ctx.saved_tensors
        grad_input, grad_weight = operators().rms_norm_backward(
            grad_output.contiguous(), contiguous, weight, inverse_rms
        )
        return grad_input, grad_weight, None


def rms_norm(input_tensor: torch.Tensor, weight: torch.Tensor, epsilon: float) -> torch.Tensor:
    if weight.ndim != 1 or input_tensor.shape[-1] != weight.shape[0]:
        raise Lfm2KernelsShapeError(
            f"RMSNorm weight {tuple(weight.shape)} must match the last axis of "
            f"{tuple(input_tensor.shape)}"
        )
    if input_tensor.dtype != weight.dtype:
        raise Lfm2KernelsShapeError(
            f"RMSNorm input {input_tensor.dtype} and weight {weight.dtype} must share a dtype"
        )
    require_same_cuda_device(input_tensor, weight)
    result: torch.Tensor = RmsNormFunction.apply(input_tensor, weight, epsilon)
    return result
