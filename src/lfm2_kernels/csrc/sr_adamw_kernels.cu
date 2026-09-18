#include "device_storage.cuh"
#include "kernels.cuh"

#include <cuda/std/array>
#include <cuda/std/cstdint>
#include <cuda/std/span>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <driver_types.h>

#include <cstddef>
#include <cstdint>

namespace lfm2_kernels {
namespace {

constexpr int reduce_threads = 512;
constexpr int update_threads = 256;
constexpr cuda::std::uint64_t moment_stream = 0x5851f42d4c957f2dULL;
constexpr cuda::std::uint64_t variance_stream = 0x14057b7ef767814fULL;
constexpr cuda::std::uint64_t seed_multiplier = 0x9e3779b97f4a7c15ULL;

struct AdamWViews {
    cuda::std::span<__nv_bfloat16> weights;
    cuda::std::span<const __nv_bfloat16> grads;
    cuda::std::span<__nv_bfloat16> exp_avg;
    cuda::std::span<__nv_bfloat16> exp_avg_sq;
    cuda::std::span<float> clip_state;
    cuda::std::span<float> partial;
};

using ReduceScratch = cuda::std::array<float, reduce_threads>;

__device__ float block_sum(float value, ReduceScratch& scratch)
{
    scratch[threadIdx.x] = value;
    __syncthreads();
#pragma unroll
    for (unsigned int stride = reduce_threads / 2; stride > 0; stride >>= 1U) {
        if (threadIdx.x < stride) {
            scratch[threadIdx.x] += scratch[threadIdx.x + stride];
        }
        __syncthreads();
    }
    return scratch[0];
}

__device__ std::int64_t first_index(int threads)
{
    return (static_cast<std::int64_t>(blockIdx.x) * threads) + static_cast<std::int64_t>(threadIdx.x);
}

__device__ std::int64_t grid_stride(int threads)
{
    return static_cast<std::int64_t>(gridDim.x) * threads;
}

__global__ void __launch_bounds__(reduce_threads) gradient_square_kernel(AdamWViews views)
{
    __shared__ ReduceScratch scratch;
    float local = 0.0F;
    const auto count = static_cast<std::int64_t>(views.grads.size());
    for (std::int64_t index = first_index(reduce_threads); index < count; index += grid_stride(reduce_threads)) {
        const float value = __bfloat162float(views.grads[static_cast<std::size_t>(index)]);
        local += value * value;
    }
    const float total = block_sum(local, scratch);
    if (threadIdx.x == 0) {
        views.partial[blockIdx.x] = total;
    }
}

__global__ void __launch_bounds__(reduce_threads) gradient_norm_kernel(AdamWViews views, float max_norm)
{
    __shared__ ReduceScratch scratch;
    float local = 0.0F;
    for (std::size_t index = threadIdx.x; index < views.partial.size(); index += reduce_threads) {
        local += views.partial[index];
    }
    const float total = block_sum(local, scratch);
    if (threadIdx.x == 0) {
        const float norm = sqrtf(total);
        const bool finite = isfinite(norm);
        views.clip_state[0] = finite ? fminf(1.0F, max_norm / (norm + 1.0e-6F)) : 0.0F;
        views.clip_state[1] = norm;
        if (!finite) {
            views.clip_state[2] = 1.0F;
        }
    }
}

__global__ void __launch_bounds__(update_threads) sr_adamw_kernel(AdamWViews views, AdamWSettings settings)
{
    if (views.clip_state[2] != 0.0F) {
        return;
    }
    const float coefficient = views.clip_state[0];
    const float step_size = settings.lr / settings.bias_correction1;
    const float decay = 1.0F - (settings.lr * settings.weight_decay);
    const cuda::std::uint64_t base = (settings.seed * seed_multiplier) ^ (settings.step << 40U);
    const auto count = static_cast<std::int64_t>(views.weights.size());
    for (std::int64_t index = first_index(update_threads); index < count; index += grid_stride(update_threads)) {
        const auto slot = static_cast<std::size_t>(index);
        const float gradient = __bfloat162float(views.grads[slot]) * coefficient;
        const float first =
            (settings.beta1 * __bfloat162float(views.exp_avg[slot])) + (settings.one_minus_beta1 * gradient);
        const float second = (settings.beta2 * __bfloat162float(views.exp_avg_sq[slot]))
            + (settings.one_minus_beta2 * gradient * gradient);
        const float denominator = (sqrtf(second) / settings.bias_correction2_sqrt) + settings.eps;
        const float updated = (__bfloat162float(views.weights[slot]) * decay) - (step_size * first / denominator);
        const cuda::std::uint64_t key = base ^ static_cast<cuda::std::uint64_t>(index);
        views.weights[slot] = round_stochastic(updated, mix_bits(key));
        views.exp_avg[slot] = round_stochastic(first, mix_bits(key ^ moment_stream));
        views.exp_avg_sq[slot] = round_stochastic(second, mix_bits(key ^ variance_stream));
    }
}

}

std::int64_t adamw_reduce_blocks(int multiprocessors)
{
    return static_cast<std::int64_t>(multiprocessors) * 4;
}

cudaError_t launch_sr_adamw(const AdamWBuffers& buffers, const AdamWSettings& settings, const LaunchContext& context)
{
    const auto count = static_cast<std::size_t>(buffers.numel);
    const AdamWViews views{
        cuda::std::span<__nv_bfloat16>(static_cast<__nv_bfloat16*>(buffers.weights), count),
        cuda::std::span<const __nv_bfloat16>(static_cast<const __nv_bfloat16*>(buffers.grads), count),
        cuda::std::span<__nv_bfloat16>(static_cast<__nv_bfloat16*>(buffers.exp_avg), count),
        cuda::std::span<__nv_bfloat16>(static_cast<__nv_bfloat16*>(buffers.exp_avg_sq), count),
        cuda::std::span<float>(buffers.clip_state, 3),
        cuda::std::span<float>(buffers.partial, static_cast<std::size_t>(buffers.partial_count))};
    const auto reduce_blocks = static_cast<unsigned int>(buffers.partial_count);
    const auto update_blocks = static_cast<unsigned int>(context.multiprocessors) * 8U;
    gradient_square_kernel<<<reduce_blocks, reduce_threads, 0, context.stream>>>(views);
    cudaError_t status = cudaGetLastError();
    if (status != cudaSuccess) {
        return status;
    }
    gradient_norm_kernel<<<1, reduce_threads, 0, context.stream>>>(views, settings.max_norm);
    status = cudaGetLastError();
    if (status != cudaSuccess) {
        return status;
    }
    sr_adamw_kernel<<<update_blocks, update_threads, 0, context.stream>>>(views, settings);
    return cudaGetLastError();
}

}
