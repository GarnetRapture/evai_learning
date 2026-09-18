"""Fused Triton gated short-convolution for LFM2-style hybrid models.

The core (``functional``, ``kernels``, ``device``, ``reference``) depends only on
PyTorch and Triton. ``integration.lfm2`` additionally imports transformers.
"""

from shortconv_triton.device import (
    DeviceProfile,
    TileConfig,
    device_profile,
    fused_path_is_available,
    select_tile,
)
from shortconv_triton.exceptions import (
    ShortConvDeviceError,
    ShortConvError,
    ShortConvShapeError,
    ShortConvUnsupportedError,
)
from shortconv_triton.functional import GatedShortConvFunction, gated_short_conv
from shortconv_triton.reference import gated_short_conv_reference

__version__ = "0.1.0"

__all__ = [
    "DeviceProfile",
    "GatedShortConvFunction",
    "ShortConvDeviceError",
    "ShortConvError",
    "ShortConvShapeError",
    "ShortConvUnsupportedError",
    "TileConfig",
    "__version__",
    "device_profile",
    "fused_path_is_available",
    "gated_short_conv",
    "gated_short_conv_reference",
    "select_tile",
]
