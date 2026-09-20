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

from evai_kernels.device import require_same_cuda_device
from evai_kernels.exceptions import (
    EvaiKernelsNonFiniteError,
    EvaiKernelsShapeError,
    EvaiKernelsUnsupportedError,
)
from evai_kernels.extension import operators

FLAT_ALIGNMENT = 8
QUANTIZED_MOMENT_BLOCK_SIZE = 256
QUANTIZED_MOMENT_CODEBOOK_SIZE = 256


def _dynamic_quantization_codebook(signed: bool) -> torch.Tensor:
    """The 256-entry dynamic exponent/fraction codebook from Dettmers et al.,
    "8-Bit Approximations for Parallelism in Deep Learning" (arXiv:1511.04561),
    as specialised for optimizer moments in "8-bit Optimizers via Block-wise
    Quantization" (arXiv:2110.02861): signed for the first moment, unsigned
    (strictly non-negative) for the second.
    """
    total_bits = 8
    max_exponent_bits = 7
    non_sign_bits = total_bits - 1
    additional_items = (2 ** (non_sign_bits - max_exponent_bits)) - 1
    values: list[float] = []
    exponent = 0
    for exponent in range(max_exponent_bits):
        fraction_items = (
            (2 ** (exponent + non_sign_bits - max_exponent_bits)) + 1
            if signed
            else (2 ** (exponent + non_sign_bits - max_exponent_bits + 1)) + 1
        )
        boundaries = torch.linspace(0.1, 1.0, fraction_items, dtype=torch.float32)
        means = (boundaries[:-1] + boundaries[1:]) / 2.0
        scale = 10.0 ** (-(max_exponent_bits - 1) + exponent)
        values.extend((scale * means).tolist())
        if signed:
            values.extend((-scale * means).tolist())
    if additional_items > 0:
        boundaries = torch.linspace(0.1, 1.0, additional_items + 1, dtype=torch.float32)
        means = (boundaries[:-1] + boundaries[1:]) / 2.0
        scale = 10.0 ** (-(max_exponent_bits - 1) + exponent)
        values.extend((scale * means).tolist())
        if signed:
            values.extend((-scale * means).tolist())
    values.append(0.0)
    values.append(1.0)
    if len(values) != QUANTIZED_MOMENT_CODEBOOK_SIZE:
        raise EvaiKernelsShapeError(
            f"Dynamic quantization codebook must hold {QUANTIZED_MOMENT_CODEBOOK_SIZE} entries, "
            f"got {len(values)}"
        )
    values.sort()
    return torch.tensor(values, dtype=torch.float32)


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
            raise EvaiKernelsShapeError("The optimizer requires trainable parameters")
        foreign = sorted({str(parameter.dtype) for parameter in parameters} - {"torch.bfloat16"})
        if foreign:
            raise EvaiKernelsShapeError(f"All trainable parameters must be bfloat16: {foreign}")
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
            raise EvaiKernelsUnsupportedError("Closure re-evaluation is not supported")
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
            raise EvaiKernelsNonFiniteError(
                "A gradient norm became non-finite; that update and later ones were withheld"
            )


class QuantizedMomentAdamW(torch.optim.Optimizer):
    """AdamW whose first and second moments live in an 8-bit block-wise dynamic
    quantized buffer on the GPU (Dettmers et al., "8-bit Optimizers via
    Block-wise Quantization", ICLR 2021) instead of full-precision moments.

    Weights and gradients stay bfloat16 exactly as in ``StochasticRoundingAdamW``;
    only ``exp_avg``/``exp_avg_sq`` are compressed to one byte plus a per-256-element
    absmax scale, cutting fixed optimizer memory from 8 to roughly 6 bytes/parameter.
    Dequantize -> update -> requantize all happen inside one CUDA launch, so there
    is no host transfer and no PCIe traffic.
    """

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
            raise EvaiKernelsShapeError("The optimizer requires trainable parameters")
        foreign = sorted({str(parameter.dtype) for parameter in parameters} - {"torch.bfloat16"})
        if foreign:
            raise EvaiKernelsShapeError(f"All trainable parameters must be bfloat16: {foreign}")
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
        blocks = -(-total // QUANTIZED_MOMENT_BLOCK_SIZE)
        self._state1 = torch.zeros(total, dtype=torch.uint8, device=device)
        self._state2 = torch.zeros(total, dtype=torch.uint8, device=device)
        self._absmax1 = torch.zeros(blocks, dtype=torch.float32, device=device)
        self._absmax2 = torch.zeros(blocks, dtype=torch.float32, device=device)
        self._quantiles1 = _dynamic_quantization_codebook(signed=True).to(device)
        self._quantiles2 = _dynamic_quantization_codebook(signed=False).to(device)
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
            raise EvaiKernelsUnsupportedError("Closure re-evaluation is not supported")
        self._steps += 1
        group = self.param_groups[0]
        beta1, beta2 = group["betas"]
        operators().sr_adamw_8bit_step(
            self._weights,
            self._grads,
            self._state1,
            self._state2,
            self._absmax1,
            self._absmax2,
            self._quantiles1,
            self._quantiles2,
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
            raise EvaiKernelsNonFiniteError(
                "A gradient norm became non-finite; that update and later ones were withheld"
            )
