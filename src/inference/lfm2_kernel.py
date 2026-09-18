"""Route the fixed LFM2.5 LIV short-convolutions and RMSNorms through the CUDA operators.

Training, evaluation, cached prefill and single-token decode all run the
``lfm2_kernels`` operator library built with the installed CUDA toolkit.
"""

from typing import Any

from common.errors import EvaiError
from lfm2_kernels.integration.lfm2 import bind_fused_rms_norm, bind_fused_short_conv


def bind_cuda_operators(model: Any) -> None:
    layer_types = list(model.config.layer_types)
    convolutions = sum(layer_type == "conv" for layer_type in layer_types)
    attention = sum(layer_type == "full_attention" for layer_type in layer_types)
    norms = 2 * len(layer_types) + 1 + 2 * attention
    bound_convolutions = bind_fused_short_conv(model)
    if bound_convolutions != convolutions:
        raise EvaiError(
            f"Fused LIV binding covered {bound_convolutions} of {convolutions} convolution layers"
        )
    bound_norms = bind_fused_rms_norm(model)
    if bound_norms != norms:
        raise EvaiError(f"Fused RMSNorm binding covered {bound_norms} of {norms} norms")
