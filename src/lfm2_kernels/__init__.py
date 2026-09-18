"""CUDA operators for LFM2-style hybrid models, built with the installed CUDA toolkit.

``csrc`` is compiled by nvcc through xmake into one operator library that registers
``torch.ops.lfm2_kernels``. The core depends only on PyTorch; ``integration.lfm2``
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
from lfm2_kernels.reference import gated_short_conv_reference
from lfm2_kernels.rms_norm import RmsNormFunction, rms_norm
from lfm2_kernels.shortconv import (
    GatedShortConvFunction,
    gated_short_conv,
    gated_short_conv_cached,
)

__version__ = "0.1.0"

__all__ = [
    "DeviceProfile",
    "GatedShortConvFunction",
    "Lfm2KernelsDeviceError",
    "Lfm2KernelsError",
    "Lfm2KernelsNonFiniteError",
    "Lfm2KernelsShapeError",
    "Lfm2KernelsUnsupportedError",
    "RmsNormFunction",
    "StochasticRoundingAdamW",
    "__version__",
    "device_profile",
    "gated_short_conv",
    "gated_short_conv_cached",
    "gated_short_conv_reference",
    "operators",
    "require_same_cuda_device",
    "rms_norm",
]
