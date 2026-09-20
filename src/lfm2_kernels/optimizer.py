"""AdamW over one flat bfloat16 buffer, executed by three CUDA launches per step.

Weights, gradients and both moments live in single contiguous buffers; every model
parameter and its ``.grad`` are views into them. Each step computes the global gradient
norm, applies the clip coefficient and performs the AdamW update with stochastic
rounding on the device, so there is no host synchronisation and no FP32 master copy.
A non-finite norm withholds the update and is reported at the next explicit check.
"""

import math
from collections.abc import Callable

import torch

from lfm2_kernels.device import require_same_cuda_device
from lfm2_kernels.exceptions import (
    Lfm2KernelsNonFiniteError,
    Lfm2KernelsShapeError,
    Lfm2KernelsUnsupportedError,
)
from lfm2_kernels.extension import operators

FLAT_ALIGNMENT = 8


class StochasticRoundingAdamW(torch.optim.Optimizer):
    def __init__(
        self,
        module: torch.nn.Module,
        *,
        lr: float,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        max_grad_norm: float,
        seed: int,
    ) -> None:
        parameters = [parameter for parameter in module.parameters() if parameter.requires_grad]
        if not parameters:
            raise Lfm2KernelsShapeError("The optimizer requires trainable parameters")
        foreign = sorted({str(parameter.dtype) for parameter in parameters} - {"torch.bfloat16"})
        if foreign:
            raise Lfm2KernelsShapeError(f"All trainable parameters must be bfloat16: {foreign}")
        device = require_same_cuda_device(*parameters)
        super().__init__(
            parameters, {"lr": lr, "betas": betas, "eps": eps, "weight_decay": weight_decay}
        )
        offsets: list[int] = []
        total = 0
        for parameter in parameters:
            offsets.append(total)
            total += -(-parameter.numel() // FLAT_ALIGNMENT) * FLAT_ALIGNMENT
        self._weights = torch.zeros(total, dtype=torch.bfloat16, device=device)
        with torch.no_grad():
            for parameter, offset in zip(parameters, offsets, strict=True):
                count = parameter.numel()
                self._weights[offset : offset + count].copy_(parameter.reshape(-1))
                parameter.data = self._weights[offset : offset + count].view_as(parameter)
        self._grads = torch.zeros_like(self._weights)
        self._exp_avg = torch.zeros_like(self._weights)
        self._exp_avg_sq = torch.zeros_like(self._weights)
        self._clip_state = torch.zeros(3, dtype=torch.float32, device=device)
        with torch.no_grad():
            for parameter, offset in zip(parameters, offsets, strict=True):
                count = parameter.numel()
                parameter.grad = self._grads[offset : offset + count].view_as(parameter)
        self._max_grad_norm = max_grad_norm
        self._seed = seed
        self._steps = 0

    @property
    def flat_numel(self) -> int:
        return self._weights.numel()

    @property
    def gradient_norm(self) -> torch.Tensor:
        return self._clip_state[1]

    def zero_grad(self, set_to_none: bool = True) -> None:
        """The fused step clears the flat gradient buffer as it consumes it."""

    @torch.no_grad()
    def step(self, closure: Callable[[], float] | None = None) -> float | None:
        if closure is not None:
            raise Lfm2KernelsUnsupportedError("Closure re-evaluation is not supported")
        self._steps += 1
        group = self.param_groups[0]
        beta1, beta2 = group["betas"]
        operators().sr_adamw_step(
            self._weights,
            self._grads,
            self._exp_avg,
            self._exp_avg_sq,
            self._clip_state,
            self._max_grad_norm,
            group["lr"],
            beta1,
            beta2,
            group["eps"],
            group["weight_decay"],
            1.0 - beta1**self._steps,
            math.sqrt(1.0 - beta2**self._steps),
            self._seed,
            self._steps,
        )
        return None

    def raise_if_nonfinite(self) -> None:
        if bool(self._clip_state[2].item()):
            raise Lfm2KernelsNonFiniteError(
                "A gradient norm became non-finite; that update and later ones were withheld"
            )
