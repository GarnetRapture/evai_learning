"""CUDA operators for the trained transformer backbone, built with the installed CUDA toolkit.

``csrc`` is compiled by nvcc through xmake into one operator library that registers
``torch.ops.evai_kernels``. The core depends only on PyTorch; ``integration.qwen3``
additionally imports transformers.
"""

from evai_kernels.device import DeviceProfile, device_profile, require_same_cuda_device
from evai_kernels.exceptions import (
    EvaiKernelsDeviceError,
    EvaiKernelsError,
    EvaiKernelsNonFiniteError,
    EvaiKernelsShapeError,
    EvaiKernelsUnsupportedError,
)
from evai_kernels.extension import operators
from evai_kernels.optimizer import QuantizedMomentAdamW, StochasticRoundingAdamW
from evai_kernels.rms_norm import RmsNormFunction, rms_norm
from evai_kernels.rope import RopeFunction, apply_rotary_pos_emb
from evai_kernels.swiglu import SwiGluFunction, swiglu

__version__ = "0.1.0"

__all__ = [
    "DeviceProfile",
    "EvaiKernelsDeviceError",
    "EvaiKernelsError",
    "EvaiKernelsNonFiniteError",
    "EvaiKernelsShapeError",
    "EvaiKernelsUnsupportedError",
    "QuantizedMomentAdamW",
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
