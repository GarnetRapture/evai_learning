"""CUDA operators for the trained transformer backbone, built with the installed CUDA toolkit.

``csrc`` is compiled by nvcc through xmake into one operator library that registers
``torch.ops.lfm2_kernels``. The core depends only on PyTorch; ``integration.qwen3``
additionally imports transformers.
"""

from lfm2_kernels.device import DeviceProfile, device_profile, require_same_cuda_device
from lfm2_kernels.exceptions import (
    Lfm2KernelsDeviceError,
    Lfm2KernelsError,
    Lfm2KernelsNonFiniteError,
    Lfm2KernelsShapeError,
    Lfm2KernelsUnsupportedError,
)
from lfm2_kernels.extension import operators
from lfm2_kernels.optimizer import StochasticRoundingAdamW
from lfm2_kernels.rms_norm import RmsNormFunction, rms_norm
from lfm2_kernels.rope import RopeFunction, apply_rotary_pos_emb
from lfm2_kernels.swiglu import SwiGluFunction, swiglu

__version__ = "0.1.0"

__all__ = [
    "DeviceProfile",
    "Lfm2KernelsDeviceError",
    "Lfm2KernelsError",
    "Lfm2KernelsNonFiniteError",
    "Lfm2KernelsShapeError",
    "Lfm2KernelsUnsupportedError",
    "RmsNormFunction",
    "RopeFunction",
    "StochasticRoundingAdamW",
    "SwiGluFunction",
    "__version__",
    "apply_rotary_pos_emb",
    "device_profile",
    "operators",
    "require_same_cuda_device",
    "rms_norm",
    "swiglu",
]
