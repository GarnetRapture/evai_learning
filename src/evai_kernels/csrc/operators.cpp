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

namespace evai_kernels {
namespace {

StorageType storage_of(const at::Tensor& tensor)
{
    if (tensor.scalar_type() == at::kBFloat16) {
        return StorageType::bfloat16;
    }
    TORCH_CHECK(
        tensor.scalar_type() == at::kFloat,
        "evai_kernels operators support bfloat16 and float32 activations, got ",
        tensor.scalar_type());
    return StorageType::float32;
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
    TORCH_CHECK(
        reinterpret_cast<std::uintptr_t>(tensor.const_data_ptr()) % (adamw_vector_width * sizeof(std::uint16_t)) == 0,
        name,
        " must start on a 16 byte boundary for vectorised access");
}

std::int64_t check_rope(const at::Tensor& input, const at::Tensor& cos, const at::Tensor& sin)
{
    TORCH_CHECK(input.is_cuda(), "RoPE input must be a CUDA tensor");
    TORCH_CHECK(input.dim() == 4, "RoPE input must be a [batch, heads, seq, head_dim] tensor");
    TORCH_CHECK(input.stride(3) == 1, "RoPE input head_dim axis must be contiguous");
    const std::int64_t head_dim = input.size(3);
    TORCH_CHECK(rope_supports_head_dim(head_dim), "RoPE supports head_dim 64, 128, or 256");
    const std::int64_t batch = input.size(0);
    const std::int64_t heads = input.size(1);
    const std::int64_t seq_len = input.size(2);
    TORCH_CHECK(heads > 0 && seq_len > 0, "RoPE heads and seq_len must be positive");
    TORCH_CHECK(cos.device() == input.device() && sin.device() == input.device(), "RoPE cos/sin must share the input device");
    TORCH_CHECK(cos.scalar_type() == input.scalar_type() && sin.scalar_type() == input.scalar_type(), "RoPE cos/sin dtype must match input");
    TORCH_CHECK(cos.is_contiguous() && sin.is_contiguous(), "RoPE cos/sin must be contiguous");
    TORCH_CHECK(
        cos.dim() == 2 && cos.size(0) == batch * seq_len && cos.size(1) == head_dim && cos.sizes() == sin.sizes(),
        "RoPE cos/sin must be [batch * seq_len, head_dim]");
    storage_of(input);
    return head_dim;
}

void check_swiglu_pair(const at::Tensor& gate, const at::Tensor& up)
{
    TORCH_CHECK(gate.is_cuda(), "SwiGLU gate must be a CUDA tensor");
    TORCH_CHECK(gate.is_contiguous() && up.is_contiguous(), "SwiGLU operands must be contiguous");
    TORCH_CHECK(up.device() == gate.device(), "SwiGLU up must share the gate device");
    TORCH_CHECK(up.scalar_type() == gate.scalar_type(), "SwiGLU up dtype must match gate");
    TORCH_CHECK(up.sizes() == gate.sizes(), "SwiGLU up shape must match gate");
    storage_of(gate);
}

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
    at::Tensor grad_weight = at::empty({columns}, weight.options());
    const RmsNormGradWeightReduce reduce_request{
        storage_of(weight),
        grad_weight_partial.const_data_ptr<float>(),
        grad_weight.mutable_data_ptr(),
        partial_rows,
        columns};
    require_launched(
        launch_rms_norm_grad_weight_reduce(reduce_request, context), "rms_norm_grad_weight_reduce");
    return {grad_input, grad_weight};
}

at::Tensor rope_apply(
    const at::Tensor& input, const at::Tensor& cos, const at::Tensor& sin, bool negate_sin)
{
    check_rope(input, cos, sin);
    const c10::cuda::CUDAGuard guard(input.device());
    at::Tensor output = at::empty(input.sizes(), input.options());
    if (input.numel() == 0) {
        return output;
    }
    const RopeApply request{
        storage_of(input),
        input.const_data_ptr(),
        Shape3{input.size(0), input.size(1), input.size(2)},
        Shape3{input.stride(0), input.stride(1), input.stride(2)},
        cos.const_data_ptr(),
        sin.const_data_ptr(),
        output.mutable_data_ptr(),
        input.size(3),
        negate_sin};
    require_launched(launch_rope_apply(request, launch_context()), "rope_apply");
    return output;
}

at::Tensor swiglu_forward(const at::Tensor& gate, const at::Tensor& up)
{
    check_swiglu_pair(gate, up);
    const c10::cuda::CUDAGuard guard(gate.device());
    at::Tensor output = at::empty(gate.sizes(), gate.options());
    const std::int64_t numel = gate.numel();
    if (numel == 0) {
        return output;
    }
    const SwiGluForward request{
        storage_of(gate), gate.const_data_ptr(), up.const_data_ptr(), output.mutable_data_ptr(), numel};
    require_launched(launch_swiglu_forward(request, launch_context()), "swiglu_forward");
    return output;
}

std::tuple<at::Tensor, at::Tensor> swiglu_backward(
    const at::Tensor& grad_output, const at::Tensor& gate, const at::Tensor& up)
{
    check_swiglu_pair(gate, up);
    TORCH_CHECK(grad_output.sizes() == gate.sizes(), "SwiGLU grad_output shape must match gate");
    TORCH_CHECK(grad_output.scalar_type() == gate.scalar_type(), "SwiGLU grad_output dtype must match gate");
    TORCH_CHECK(grad_output.is_contiguous(), "SwiGLU grad_output must be contiguous");
    const c10::cuda::CUDAGuard guard(gate.device());
    at::Tensor grad_gate = at::empty(gate.sizes(), gate.options());
    at::Tensor grad_up = at::empty(gate.sizes(), gate.options());
    const std::int64_t numel = gate.numel();
    if (numel == 0) {
        return {grad_gate, grad_up};
    }
    const SwiGluBackward request{
        storage_of(gate),
        gate.const_data_ptr(),
        up.const_data_ptr(),
        grad_output.const_data_ptr(),
        grad_gate.mutable_data_ptr(),
        grad_up.mutable_data_ptr(),
        numel};
    require_launched(launch_swiglu_backward(request, launch_context()), "swiglu_backward");
    return {grad_gate, grad_up};
}

void sr_adamw_step(
    at::Tensor& param,
    at::Tensor& grad,
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
        grad.mutable_data_ptr(),
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

TORCH_LIBRARY(evai_kernels, library)
{
    library.def("rms_norm_forward(Tensor input, Tensor weight, float epsilon) -> (Tensor, Tensor)");
    library.def(
        "rms_norm_backward(Tensor grad_output, Tensor input, Tensor weight, Tensor inverse_rms)"
        " -> (Tensor, Tensor)");
    library.def("swiglu_forward(Tensor gate, Tensor up) -> Tensor");
    library.def("swiglu_backward(Tensor grad_output, Tensor gate, Tensor up) -> (Tensor, Tensor)");
    library.def("rope_apply(Tensor input, Tensor cos, Tensor sin, bool negate_sin) -> Tensor");
    library.def(
        "sr_adamw_step(Tensor(a!) param, Tensor(e!) grad, Tensor(b!) exp_avg, Tensor(c!) exp_avg_sq,"
        " Tensor(d!) clip_state, float max_norm, float lr, float beta1, float beta2, float eps,"
        " float weight_decay, float bias_correction1, float bias_correction2_sqrt, int seed, int step) -> ()");
}

TORCH_LIBRARY_IMPL(evai_kernels, CUDA, library)
{
    library.impl("rms_norm_forward", &evai_kernels::rms_norm_forward);
    library.impl("rms_norm_backward", &evai_kernels::rms_norm_backward);
    library.impl("swiglu_forward", &evai_kernels::swiglu_forward);
    library.impl("swiglu_backward", &evai_kernels::swiglu_backward);
    library.impl("rope_apply", &evai_kernels::rope_apply);
    library.impl("sr_adamw_step", &evai_kernels::sr_adamw_step);
}
