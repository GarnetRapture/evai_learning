#include "ops.cuh"

#include <torch/library.h>

TORCH_LIBRARY(lfm2_kernels, library)
{
    library.def(
        "shortconv_forward(Tensor projection, Tensor weight, Tensor? bias, Tensor? state) -> Tensor");
    library.def(
        "shortconv_backward(Tensor projection, Tensor weight, Tensor? bias, Tensor grad_output)"
        " -> (Tensor, Tensor, Tensor)");
    library.def("shortconv_state_update(Tensor projection, Tensor(a!) state, bool has_history) -> ()");
    library.def(
        "sr_adamw_step(Tensor(a!) param, Tensor grad, Tensor(b!) exp_avg, Tensor(c!) exp_avg_sq,"
        " Tensor(d!) clip_state, float max_norm, float lr, float beta1, float beta2, float eps,"
        " float weight_decay, float bias_correction1, float bias_correction2_sqrt, int seed, int step) -> ()");
}

TORCH_LIBRARY_IMPL(lfm2_kernels, CUDA, library)
{
    library.impl("shortconv_forward", &lfm2_kernels::shortconv_forward);
    library.impl("shortconv_backward", &lfm2_kernels::shortconv_backward);
    library.impl("shortconv_state_update", &lfm2_kernels::shortconv_state_update);
    library.impl("sr_adamw_step", &lfm2_kernels::sr_adamw_step);
}
