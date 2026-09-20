"""SwiGLU activation (silu(gate) * up) fused into one CUDA launch per direction.

Matches ``F.silu(gate) * up`` exactly: the two elementwise passes (silu, then multiply)
that a composed PyTorch MLP performs as separate kernel launches are fused into one.
"""

from typing import Any

import torch

from evai_kernels.device import require_same_cuda_device
from evai_kernels.exceptions import EvaiKernelsShapeError
from evai_kernels.extension import operators


class SwiGluFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
        gate_contiguous = gate.contiguous()
        up_contiguous = up.contiguous()
        output = operators().swiglu_forward(gate_contiguous, up_contiguous)
        ctx.save_for_backward(gate_contiguous, up_contiguous)
        return output

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        gate, up = ctx.saved_tensors
        grad_gate, grad_up = operators().swiglu_backward(grad_output.contiguous(), gate, up)
        return grad_gate, grad_up


def swiglu(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    if gate.shape != up.shape:
        raise EvaiKernelsShapeError(
            f"SwiGLU gate {tuple(gate.shape)} and up {tuple(up.shape)} must match"
        )
    if gate.dtype != up.dtype:
        raise EvaiKernelsShapeError(
            f"SwiGLU gate {gate.dtype} and up {up.dtype} must share a dtype"
        )
    require_same_cuda_device(gate, up)
    result: torch.Tensor = SwiGluFunction.apply(gate, up)
    return result
