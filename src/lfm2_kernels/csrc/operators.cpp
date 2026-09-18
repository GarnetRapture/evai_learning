#include "kernels.cuh"

#include <ATen/core/Tensor.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/ops/empty.h>
#include <ATen/ops/zeros.h>
#include <c10/core/ScalarType.h>
#include <c10/core/TensorOptions.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/util/Exception.h>
#include <torch/library.h>

#include <cuda_runtime_api.h>
#include <driver_types.h>

#include <algorithm>
#include <cstdint>
#include <optional>
#include <tuple>

namespace lfm2_kernels {
namespace {

StorageType storage_of(const at::Tensor& tensor)
{
    if (tensor.scalar_type() == at::kBFloat16) {
        return StorageType::bfloat16;
    }
    TORCH_CHECK(
        tensor.scalar_type() == at::kFloat,
        "lfm2_kernels short-convolution supports bfloat16 and float32 activations, got ",
        tensor.scalar_type());
    return StorageType::float32;
}

void require_positive_strides(const at::Tensor& tensor, const char* name)
{
    for (std::int64_t axis = 0; axis < tensor.dim(); ++axis) {
        TORCH_CHECK(tensor.stride(axis) > 0, name, " must have positive strides");
    }
}

ConstTensor3 const_view(const at::Tensor& tensor, const char* name)
{
    require_positive_strides(tensor, name);
    return ConstTensor3{
        tensor.const_data_ptr(),
        Shape3{tensor.size(0), tensor.size(1), tensor.size(2)},
        Shape3{tensor.stride(0), tensor.stride(1), tensor.stride(2)}};
}

MutableTensor3 mutable_view(at::Tensor& tensor, const char* name)
{
    require_positive_strides(tensor, name);
    return MutableTensor3{
        tensor.mutable_data_ptr(),
        Shape3{tensor.size(0), tensor.size(1), tensor.size(2)},
        Shape3{tensor.stride(0), tensor.stride(1), tensor.stride(2)}};
}

ConstTensor3 absent_view()
{
    return ConstTensor3{nullptr, Shape3{0, 0, 0}, Shape3{1, 1, 1}};
}

LaunchContext launch_context()
{
    return LaunchContext{
        at::cuda::getCurrentCUDAStream().stream(), at::cuda::getCurrentDeviceProperties()->multiProcessorCount};
}

void require_launched(cudaError_t status, const char* operation)
{
    TORCH_CHECK(status == cudaSuccess, operation, " failed to launch: ", describe_cuda_error(status));
}

void check_projection(const at::Tensor& projection)
{
    TORCH_CHECK(projection.is_cuda(), "projection must be a CUDA tensor");
    TORCH_CHECK(projection.dim() == 3, "projection must be [batch, tokens, 3 * channels]");
    TORCH_CHECK(projection.size(2) % 3 == 0, "projection channel axis must split into gate|carrier|value");
    TORCH_CHECK(projection.stride(2) == 1, "projection must be contiguous along its channel axis");
}

ShortConvParameters check_parameters(
    const at::Tensor& projection, const at::Tensor& weight, const std::optional<at::Tensor>& bias)
{
    const std::int64_t channels = projection.size(2) / 3;
    TORCH_CHECK(weight.device() == projection.device(), "weight must share the projection device");
    TORCH_CHECK(weight.scalar_type() == at::kFloat, "weight must be float32");
    TORCH_CHECK(weight.dim() == 2 && weight.size(0) == channels, "weight must be [channels, taps]");
    TORCH_CHECK(weight.is_contiguous(), "weight must be contiguous");
    TORCH_CHECK(shortconv_supports_taps(weight.size(1)), "short-convolution supports 2 to 4 taps");
    if (bias.has_value()) {
        TORCH_CHECK(bias->device() == projection.device(), "bias must share the projection device");
        TORCH_CHECK(bias->scalar_type() == at::kFloat, "bias must be float32");
        TORCH_CHECK(bias->dim() == 1 && bias->size(0) == channels, "bias must be [channels]");
        TORCH_CHECK(bias->is_contiguous(), "bias must be contiguous");
    }
    return ShortConvParameters{
        weight.const_data_ptr<float>(), bias.has_value() ? bias->const_data_ptr<float>() : nullptr, weight.size(1)};
}

void check_companion(const at::Tensor& tensor, const at::Tensor& projection, const char* name)
{
    TORCH_CHECK(tensor.device() == projection.device(), name, " must share the projection device");
    TORCH_CHECK(tensor.scalar_type() == projection.scalar_type(), name, " dtype must match projection");
    TORCH_CHECK(tensor.dim() == 3, name, " must be three-dimensional");
}

std::int64_t check_rms_norm(const at::Tensor& input, const at::Tensor& weight)
{
    TORCH_CHECK(input.is_cuda(), "RMSNorm input must be a CUDA tensor");
    TORCH_CHECK(input.dim() >= 1 && input.is_contiguous(), "RMSNorm input must be contiguous");
    TORCH_CHECK(weight.device() == input.device(), "RMSNorm weight must share the input device");
    TORCH_CHECK(weight.scalar_type() == input.scalar_type(), "RMSNorm weight dtype must match input");
    TORCH_CHECK(weight.dim() == 1 && weight.is_contiguous(), "RMSNorm weight must be one contiguous row");
    const std::int64_t columns = input.size(-1);
    TORCH_CHECK(weight.size(0) == columns, "RMSNorm weight length must match the normalized axis");
    TORCH_CHECK(rms_norm_supports_columns(columns), "RMSNorm supports widths 32 to 1024 in powers of two");
    storage_of(input);
    return columns;
}

void check_flat(const at::Tensor& tensor, const at::Tensor& reference, const char* name)
{
    TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(tensor.device() == reference.device(), name, " must share the parameter device");
    TORCH_CHECK(tensor.scalar_type() == at::kBFloat16, name, " must be bfloat16");
    TORCH_CHECK(tensor.dim() == 1 && tensor.is_contiguous(), name, " must be one contiguous flat buffer");
    TORCH_CHECK(tensor.numel() == reference.numel(), name, " must match the parameter buffer length");
}

}

at::Tensor shortconv_forward(
    const at::Tensor& projection,
    const at::Tensor& weight,
    const std::optional<at::Tensor>& bias,
    const std::optional<at::Tensor>& state)
{
    check_projection(projection);
    const ShortConvParameters parameters = check_parameters(projection, weight, bias);
    const c10::cuda::CUDAGuard guard(projection.device());
    const std::int64_t batch = projection.size(0);
    const std::int64_t seq_len = projection.size(1);
    const std::int64_t channels = projection.size(2) / 3;
    if (state.has_value()) {
        check_companion(*state, projection, "state");
        TORCH_CHECK(
            state->size(0) == batch && state->size(1) == channels && state->size(2) >= parameters.taps - 1,
            "state must be [batch, channels, history >= taps - 1]");
    }
    at::Tensor output = at::empty({batch, seq_len, channels}, projection.options());
    if (batch == 0 || seq_len == 0) {
        return output;
    }
    const LaunchContext context = launch_context();
    const ShortConvForward request{
        storage_of(projection),
        const_view(projection, "projection"),
        parameters,
        state.has_value() ? const_view(*state, "state") : absent_view(),
        mutable_view(output, "output"),
        shortconv_chunk_tokens(batch, seq_len, channels, context.multiprocessors)};
    require_launched(launch_shortconv_forward(request, context), "shortconv_forward");
    return output;
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> shortconv_backward(
    const at::Tensor& projection,
    const at::Tensor& weight,
    const std::optional<at::Tensor>& bias,
    const at::Tensor& grad_output)
{
    check_projection(projection);
    const ShortConvParameters parameters = check_parameters(projection, weight, bias);
    const c10::cuda::CUDAGuard guard(projection.device());
    const std::int64_t batch = projection.size(0);
    const std::int64_t seq_len = projection.size(1);
    const std::int64_t channels = projection.size(2) / 3;
    check_companion(grad_output, projection, "grad_output");
    TORCH_CHECK(
        grad_output.size(0) == batch && grad_output.size(1) == seq_len && grad_output.size(2) == channels,
        "grad_output must be [batch, tokens, channels]");

    at::Tensor grad_projection = at::empty(projection.sizes(), projection.options());
    const at::TensorOptions partial_options = weight.options();
    if (batch == 0 || seq_len == 0) {
        grad_projection.zero_();
        return {
            grad_projection,
            at::zeros({channels, parameters.taps}, partial_options),
            at::zeros({bias.has_value() ? channels : 0}, partial_options)};
    }
    const LaunchContext context = launch_context();
    const std::int64_t chunk = shortconv_chunk_tokens(batch, seq_len, channels, context.multiprocessors);
    const std::int64_t chunks = (seq_len + chunk - 1) / chunk;
    at::Tensor grad_weight_partial = at::empty({batch * chunks, channels, parameters.taps}, partial_options);
    at::Tensor grad_bias_partial =
        at::empty({bias.has_value() ? batch * chunks : 0, bias.has_value() ? channels : 0}, partial_options);
    const ShortConvBackward request{
        storage_of(projection),
        const_view(projection, "projection"),
        parameters,
        const_view(grad_output, "grad_output"),
        mutable_view(grad_projection, "grad_projection"),
        grad_weight_partial.mutable_data_ptr<float>(),
        bias.has_value() ? grad_bias_partial.mutable_data_ptr<float>() : nullptr,
        chunk,
        chunks};
    require_launched(launch_shortconv_backward(request, context), "shortconv_backward");
    at::Tensor grad_weight = grad_weight_partial.sum(0);
    at::Tensor grad_bias = bias.has_value() ? grad_bias_partial.sum(0) : grad_bias_partial.reshape({0});
    return {grad_projection, grad_weight, grad_bias};
}

void shortconv_state_update(const at::Tensor& projection, at::Tensor& state, bool has_history)
{
    check_projection(projection);
    check_companion(state, projection, "state");
    const c10::cuda::CUDAGuard guard(projection.device());
    TORCH_CHECK(
        state.size(0) == projection.size(0) && state.size(1) == projection.size(2) / 3,
        "state must be [batch, channels, taps]");
    TORCH_CHECK(shortconv_supports_taps(state.size(2)), "state window must hold 2 to 4 taps");
    if (projection.size(0) == 0) {
        return;
    }
    const ShortConvStateUpdate request{
        storage_of(projection), const_view(projection, "projection"), mutable_view(state, "state"), has_history};
    require_launched(launch_shortconv_state_update(request, launch_context()), "shortconv_state_update");
}

std::tuple<at::Tensor, at::Tensor> rms_norm_forward(
    const at::Tensor& input, const at::Tensor& weight, double epsilon)
{
    const std::int64_t columns = check_rms_norm(input, weight);
    const c10::cuda::CUDAGuard guard(input.device());
    const std::int64_t rows = input.numel() / columns;
    at::Tensor output = at::empty(input.sizes(), input.options());
    at::Tensor inverse_rms = at::empty({rows}, input.options().dtype(at::kFloat));
    if (rows == 0) {
        return {output, inverse_rms};
    }
    const RmsNormForward request{
        storage_of(input),
        input.const_data_ptr(),
        weight.const_data_ptr(),
        output.mutable_data_ptr(),
        inverse_rms.mutable_data_ptr<float>(),
        rows,
        columns,
        static_cast<float>(epsilon)};
    require_launched(launch_rms_norm_forward(request, launch_context()), "rms_norm_forward");
    return {output, inverse_rms};
}

std::tuple<at::Tensor, at::Tensor> rms_norm_backward(
    const at::Tensor& grad_output,
    const at::Tensor& input,
    const at::Tensor& weight,
    const at::Tensor& inverse_rms)
{
    const std::int64_t columns = check_rms_norm(input, weight);
    TORCH_CHECK(grad_output.sizes() == input.sizes(), "RMSNorm grad_output must match the input shape");
    TORCH_CHECK(grad_output.scalar_type() == input.scalar_type(), "RMSNorm grad_output dtype must match input");
    TORCH_CHECK(grad_output.is_contiguous(), "RMSNorm grad_output must be contiguous");
    const std::int64_t rows = input.numel() / columns;
    TORCH_CHECK(
        inverse_rms.scalar_type() == at::kFloat && inverse_rms.numel() == rows && inverse_rms.is_contiguous(),
        "RMSNorm inverse_rms must hold one float per row");
    const c10::cuda::CUDAGuard guard(input.device());
    at::Tensor grad_input = at::empty(input.sizes(), input.options());
    if (rows == 0) {
        return {grad_input, at::zeros({columns}, weight.options())};
    }
    const LaunchContext context = launch_context();
    const std::int64_t partial_rows = rms_norm_backward_partial_rows(context.multiprocessors);
    at::Tensor grad_weight_partial = at::empty({partial_rows, columns}, input.options().dtype(at::kFloat));
    const RmsNormBackward request{
        storage_of(input),
        grad_output.const_data_ptr(),
        input.const_data_ptr(),
        weight.const_data_ptr(),
        inverse_rms.const_data_ptr<float>(),
        grad_input.mutable_data_ptr(),
        grad_weight_partial.mutable_data_ptr<float>(),
        rows,
        columns,
        partial_rows};
    require_launched(launch_rms_norm_backward(request, context), "rms_norm_backward");
    return {grad_input, grad_weight_partial.sum(0).to(weight.scalar_type())};
}

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
    std::int64_t step)
{
    check_flat(param, param, "param");
    check_flat(grad, param, "grad");
    check_flat(exp_avg, param, "exp_avg");
    check_flat(exp_avg_sq, param, "exp_avg_sq");
    TORCH_CHECK(clip_state.device() == param.device(), "clip_state must share the parameter device");
    TORCH_CHECK(clip_state.scalar_type() == at::kFloat, "clip_state must be float32");
    TORCH_CHECK(clip_state.numel() == 3 && clip_state.is_contiguous(), "clip_state must hold three floats");
    const c10::cuda::CUDAGuard guard(param.device());
    const LaunchContext context = launch_context();
    const std::int64_t partial_count = adamw_reduce_blocks(context.multiprocessors);
    at::Tensor partial = at::empty({partial_count}, clip_state.options());
    const AdamWBuffers buffers{
        param.mutable_data_ptr(),
        grad.const_data_ptr(),
        exp_avg.mutable_data_ptr(),
        exp_avg_sq.mutable_data_ptr(),
        param.numel(),
        clip_state.mutable_data_ptr<float>(),
        partial.mutable_data_ptr<float>(),
        partial_count};
    const AdamWSettings settings{
        static_cast<float>(max_norm),
        static_cast<float>(lr),
        static_cast<float>(beta1),
        static_cast<float>(beta2),
        static_cast<float>(1.0 - beta1),
        static_cast<float>(1.0 - beta2),
        static_cast<float>(eps),
        static_cast<float>(weight_decay),
        static_cast<float>(bias_correction1),
        static_cast<float>(bias_correction2_sqrt),
        static_cast<std::uint64_t>(seed),
        static_cast<std::uint64_t>(step)};
    require_launched(launch_sr_adamw(buffers, settings, context), "sr_adamw_step");
}

}

TORCH_LIBRARY(lfm2_kernels, library)
{
    library.def("shortconv_forward(Tensor projection, Tensor weight, Tensor? bias, Tensor? state) -> Tensor");
    library.def(
        "shortconv_backward(Tensor projection, Tensor weight, Tensor? bias, Tensor grad_output)"
        " -> (Tensor, Tensor, Tensor)");
    library.def("shortconv_state_update(Tensor projection, Tensor(a!) state, bool has_history) -> ()");
    library.def("rms_norm_forward(Tensor input, Tensor weight, float epsilon) -> (Tensor, Tensor)");
    library.def(
        "rms_norm_backward(Tensor grad_output, Tensor input, Tensor weight, Tensor inverse_rms)"
        " -> (Tensor, Tensor)");
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
    library.impl("rms_norm_forward", &lfm2_kernels::rms_norm_forward);
    library.impl("rms_norm_backward", &lfm2_kernels::rms_norm_backward);
    library.impl("sr_adamw_step", &lfm2_kernels::sr_adamw_step);
}
