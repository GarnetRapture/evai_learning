#pragma once

#include <ATen/core/Tensor.h>

#include <cstdint>
#include <optional>
#include <tuple>

namespace lfm2_kernels {

at::Tensor shortconv_forward(
    const at::Tensor& projection,
    const at::Tensor& weight,
    const std::optional<at::Tensor>& bias,
    const std::optional<at::Tensor>& state);

std::tuple<at::Tensor, at::Tensor, at::Tensor> shortconv_backward(
    const at::Tensor& projection,
    const at::Tensor& weight,
    const std::optional<at::Tensor>& bias,
    const at::Tensor& grad_output);

void shortconv_state_update(const at::Tensor& projection, at::Tensor& state, bool has_history);

void sr_adamw_step(
    at::Tensor& param,
    const at::Tensor& grad,
    at::Tensor& exp_avg,
    at::Tensor& exp_avg_sq,
    at::Tensor& clip_state,
    double max_norm,
    double lr,
    double beta1,
    double beta2,
    double eps,
    double weight_decay,
    double bias_correction1,
    double bias_correction2_sqrt,
    std::int64_t seed,
    std::int64_t step);

}
