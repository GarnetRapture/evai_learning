"""Route the trained Qwen3 backbone's RMSNorm, SwiGLU, and RoPE math through fused CUDA
operators."""

from typing import Any

from common.errors import EvaiError
from lfm2_kernels.integration.qwen3 import (
    bind_fused_rms_norm,
    bind_fused_rope,
    bind_fused_swiglu,
    expected_attentions,
    expected_mlps,
    expected_rms_norms,
)


def bind_cuda_operators(model: Any) -> None:
    layers = model.config.num_hidden_layers
    expected_norms = expected_rms_norms(layers)
    bound_norms = bind_fused_rms_norm(model)
    if bound_norms != expected_norms:
        raise EvaiError(f"Fused RMSNorm binding covered {bound_norms} of {expected_norms} norms")
    expected_swiglu = expected_mlps(layers)
    bound_swiglu = bind_fused_swiglu(model)
    if bound_swiglu != expected_swiglu:
        raise EvaiError(f"Fused SwiGLU binding covered {bound_swiglu} of {expected_swiglu} MLPs")
    expected_rope = expected_attentions(layers)
    bound_rope = bind_fused_rope(model)
    if bound_rope != expected_rope:
        raise EvaiError(
            f"Fused RoPE binding covered {bound_rope} of {expected_rope} attention layers"
        )
