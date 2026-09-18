#include "ops.cuh"
#include "storage.cuh"

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/util/Exception.h>

#include <cstdint>

namespace lfm2_kernels {
namespace {

constexpr int reduce_threads = 512;
constexpr int update_threads = 256;
constexpr std::uint64_t moment_stream = 0x5851f42d4c957f2dULL;
constexpr std::uint64_t variance_stream = 0x14057b7ef767814fULL;

struct AdamWHyperparameters {
    float max_norm;
    float lr;
    float beta1;
    float beta2;
    float eps;
    float weight_decay;
    float bias_correction1;
    float bias_correction2_sqrt;
    std::uint64_t seed;
    std::uint64_t step;
};

template <int Threads>
__device__ float block_sum(float value, float* shared)
{
    shared[threadIdx.x] = value;
    __syncthreads();
#pragma unroll
    for (int stride = Threads / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            shared[threadIdx.x] += shared[threadIdx.x + stride];
        }
        __syncthreads();
    }
    return shared[0];
}

__global__ void __launch_bounds__(reduce_threads) gradient_square_kernel(
    const std::uint16_t* __restrict__ grad, std::int64_t numel, float* __restrict__ partial)
{
    __shared__ float shared[reduce_threads];
    float local = 0.0f;
    const std::int64_t stride = static_cast<std::int64_t>(gridDim.x) * reduce_threads;
    for (std::int64_t index = static_cast<std::int64_t>(blockIdx.x) * reduce_threads + threadIdx.x; index < numel;
         index += stride) {
        const float value = bf16_bits_to_float(grad[index]);
        local += value * value;
    }
    const float total = block_sum<reduce_threads>(local, shared);
    if (threadIdx.x == 0) {
        partial[blockIdx.x] = total;
    }
}

__global__ void __launch_bounds__(reduce_threads) gradient_norm_kernel(
    const float* __restrict__ partial, int count, float max_norm, float* __restrict__ clip_state)
{
    __shared__ float shared[reduce_threads];
    float local = 0.0f;
    for (int index = threadIdx.x; index < count; index += reduce_threads) {
        local += partial[index];
    }
    const float total = block_sum<reduce_threads>(local, shared);
    if (threadIdx.x == 0) {
        const float norm = sqrtf(total);
        const bool finite = isfinite(norm);
        clip_state[0] = finite ? fminf(1.0f, max_norm / (norm + 1.0e-6f)) : 0.0f;
        clip_state[1] = norm;
        if (!finite) {
            clip_state[2] = 1.0f;
        }
    }
}

__global__ void __launch_bounds__(update_threads) sr_adamw_kernel(
    std::uint16_t* __restrict__ param,
    const std::uint16_t* __restrict__ grad,
    std::uint16_t* __restrict__ exp_avg,
    std::uint16_t* __restrict__ exp_avg_sq,
    std::int64_t numel,
    const float* __restrict__ clip_state,
    AdamWHyperparameters settings)
{
    if (clip_state[2] != 0.0f) {
        return;
    }
    const float coefficient = clip_state[0];
    const float step_size = settings.lr / settings.bias_correction1;
    const float decay = 1.0f - settings.lr * settings.weight_decay;
    const float keep_first = settings.beta1;
    const float take_first = 1.0f - settings.beta1;
    const float keep_second = settings.beta2;
    const float take_second = 1.0f - settings.beta2;
    const std::uint64_t base = settings.seed * 0x9e3779b97f4a7c15ULL ^ (settings.step << 40);
    const std::int64_t stride = static_cast<std::int64_t>(gridDim.x) * update_threads;
    for (std::int64_t index = static_cast<std::int64_t>(blockIdx.x) * update_threads + threadIdx.x; index < numel;
         index += stride) {
        const float gradient = bf16_bits_to_float(grad[index]) * coefficient;
        const float first = keep_first * bf16_bits_to_float(exp_avg[index]) + take_first * gradient;
        const float second = keep_second * bf16_bits_to_float(exp_avg_sq[index]) + take_second * gradient * gradient;
        const float denominator = sqrtf(second) / settings.bias_correction2_sqrt + settings.eps;
        const float updated = bf16_bits_to_float(param[index]) * decay - step_size * first / denominator;
        const std::uint64_t key = base ^ static_cast<std::uint64_t>(index);
        param[index] = float_to_bf16_stochastic(updated, mix_bits(key));
        exp_avg[index] = float_to_bf16_stochastic(first, mix_bits(key ^ moment_stream));
        exp_avg_sq[index] = float_to_bf16_stochastic(second, mix_bits(key ^ variance_stream));
    }
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
    const std::int64_t numel = param.numel();
    const int multiprocessors = at::cuda::getCurrentDeviceProperties()->multiProcessorCount;
    const int reduce_blocks = multiprocessors * 4;
    const int update_blocks = multiprocessors * 8;
    at::Tensor partial = at::empty({reduce_blocks}, clip_state.options());
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    const auto* grad_bits = reinterpret_cast<const std::uint16_t*>(grad.const_data_ptr());

    gradient_square_kernel<<<reduce_blocks, reduce_threads, 0, stream>>>(
        grad_bits, numel, partial.mutable_data_ptr<float>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    gradient_norm_kernel<<<1, reduce_threads, 0, stream>>>(
        partial.const_data_ptr<float>(), reduce_blocks, static_cast<float>(max_norm),
        clip_state.mutable_data_ptr<float>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    const AdamWHyperparameters settings{
        static_cast<float>(max_norm),
        static_cast<float>(lr),
        static_cast<float>(beta1),
        static_cast<float>(beta2),
        static_cast<float>(eps),
        static_cast<float>(weight_decay),
        static_cast<float>(bias_correction1),
        static_cast<float>(bias_correction2_sqrt),
        static_cast<std::uint64_t>(seed),
        static_cast<std::uint64_t>(step)};
    sr_adamw_kernel<<<update_blocks, update_threads, 0, stream>>>(
        reinterpret_cast<std::uint16_t*>(param.mutable_data_ptr()),
        grad_bits,
        reinterpret_cast<std::uint16_t*>(exp_avg.mutable_data_ptr()),
        reinterpret_cast<std::uint16_t*>(exp_avg_sq.mutable_data_ptr()),
        numel,
        clip_state.const_data_ptr<float>(),
        settings);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}
