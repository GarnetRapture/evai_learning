"""Route the fixed LFM2.5 LIV short-convolutions through the project's fused library.

Uncached full-sequence forwards run the ``shortconv_triton`` Triton kernels; cached
prefill and single-token decode keep the library's recurrent-state path.
"""

from typing import Any

from common.errors import EvaiError
from shortconv_triton.integration.lfm2 import bind_fused_short_conv


def bind_native_liv(model: Any) -> None:
    expected = sum(layer_type == "conv" for layer_type in model.config.layer_types)
    bound = bind_fused_short_conv(model)
    if bound != expected:
        raise EvaiError(f"Fused LIV binding covered {bound} of {expected} convolution layers")
