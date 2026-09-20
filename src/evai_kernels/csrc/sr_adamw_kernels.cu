#include "device_storage.cuh"
#include "kernels.cuh"

#include <cuda/std/algorithm>
#include <cuda/std/array>
#include <cuda/std/cstdint>
#include <cuda/std/span>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <driver_types.h>

#include <cstddef>
#include <cstdint>

namespace evai_kernels {
namespace {

constexpr int reduce_threads = 512;
constexpr int update_threads = 256;
constexpr cuda::std::uint64_t moment_stream = 0x5851f42d4c957f2dULL;
constexpr cuda::std::uint64_t variance_stream = 0x14057b7ef767814fULL;
constexpr cuda::std::uint64_t seed_multiplier = 0x9e3779b97f4a7c15ULL;

struct GradNormViews {
    cuda::std::span<BFloat16Vector> grads;
    cuda::std::span<float> clip_state;
    cuda::std::span<float> partial;
};

struct AdamWViews {
    cuda::std::span<BFloat16Vector> weights;
    cuda::std::span<BFloat16Vector> grads;
    cuda::std::span<BFloat16Vector> exp_avg;
    cuda::std::span<BFloat16Vector> exp_avg_sq;
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

template <int Threads>
__device__ float block_max(float value, cuda::std::array<float, Threads>& scratch)
{
    scratch[threadIdx.x] = value;
    __syncthreads();
#pragma unroll
    for (unsigned int stride = Threads / 2; stride > 0; stride >>= 1U) {
        if (threadIdx.x < stride) {
            scratch[threadIdx.x] = fmaxf(scratch[threadIdx.x], scratch[threadIdx.x + stride]);
        }
        __syncthreads();
    }
    return scratch[0];
}

__device__ unsigned char quantize_nearest(const float* sorted_code, float value)
{
    int pivot = 127;
    int lower_pivot = 0;
    int upper_pivot = 255;
    float lower = sorted_code[0];
    float upper = sorted_code[255];
    float at_pivot = sorted_code[pivot];
#pragma unroll
    for (int step = 64; step > 0; step >>= 1) {
        if (value > at_pivot) {
            lower_pivot = pivot;
            lower = at_pivot;
            pivot += step;
        } else {
            upper_pivot = pivot;
            upper = at_pivot;
            pivot -= step;
        }
        at_pivot = sorted_code[pivot];
    }
    if (value > at_pivot) {
        const float midpoint = (upper + at_pivot) * 0.5F;
        return static_cast<unsigned char>(value > midpoint ? upper_pivot : pivot);
    }
    const float midpoint = (lower + at_pivot) * 0.5F;
    return static_cast<unsigned char>(value < midpoint ? lower_pivot : pivot);
}

__device__ std::int64_t first_index(int threads)
{
    return (static_cast<std::int64_t>(blockIdx.x) * threads) + static_cast<std::int64_t>(threadIdx.x);
}

__device__ std::int64_t grid_stride(int threads)
{
    return static_cast<std::int64_t>(gridDim.x) * threads;
}

__global__ void __launch_bounds__(reduce_threads) gradient_square_kernel(GradNormViews views)
{
    __shared__ ReduceScratch scratch;
    float local = 0.0F;
    const auto count = static_cast<std::int64_t>(views.grads.size());
    for (std::int64_t index = first_index(reduce_threads); index < count; index += grid_stride(reduce_threads)) {
        const BFloat16Vector chunk = views.grads[static_cast<std::size_t>(index)];
#pragma unroll
        for (int slot = 0; slot < adamw_vector_width; ++slot) {
            const float value = __bfloat162float(chunk.lane[slot]);
            local += value * value;
        }
    }
    const float total = block_sum(local, scratch);
    if (threadIdx.x == 0) {
        views.partial[blockIdx.x] = total;
    }
}

__global__ void __launch_bounds__(reduce_threads) gradient_norm_kernel(GradNormViews views, float max_norm)
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
    const bool withheld = views.clip_state[2] != 0.0F;
    const float coefficient = views.clip_state[0];
    const float step_size = settings.lr / settings.bias_correction1;
    const float decay = 1.0F - (settings.lr * settings.weight_decay);
    const cuda::std::uint64_t base = (settings.seed * seed_multiplier) ^ (settings.step << 40U);
    const auto count = static_cast<std::int64_t>(views.weights.size());
    for (std::int64_t index = first_index(update_threads); index < count; index += grid_stride(update_threads)) {
        const auto position = static_cast<std::size_t>(index);
        const BFloat16Vector gradients = views.grads[position];
        views.grads[position] = zero_vector();
        if (withheld) {
            continue;
        }
        BFloat16Vector weights = views.weights[position];
        BFloat16Vector moments = views.exp_avg[position];
        BFloat16Vector variances = views.exp_avg_sq[position];
        const std::int64_t origin = index * adamw_vector_width;
#pragma unroll
        for (int slot = 0; slot < adamw_vector_width; ++slot) {
            const float gradient = __bfloat162float(gradients.lane[slot]) * coefficient;
            const float first =
                (settings.beta1 * __bfloat162float(moments.lane[slot])) + (settings.one_minus_beta1 * gradient);
            const float second = (settings.beta2 * __bfloat162float(variances.lane[slot]))
                + (settings.one_minus_beta2 * gradient * gradient);
            const float denominator = (sqrtf(second) / settings.bias_correction2_sqrt) + settings.eps;
            const float updated =
                (__bfloat162float(weights.lane[slot]) * decay) - (step_size * first / denominator);
            const cuda::std::uint64_t key = base ^ static_cast<cuda::std::uint64_t>(origin + slot);
            weights.lane[slot] = round_stochastic(updated, mix_bits(key));
            moments.lane[slot] = round_stochastic(first, mix_bits(key ^ moment_stream));
            variances.lane[slot] = round_stochastic(second, mix_bits(key ^ variance_stream));
        }
        views.weights[position] = weights;
        views.exp_avg[position] = moments;
        views.exp_avg_sq[position] = variances;
    }
}

constexpr int quantized_block_threads = static_cast<int>(quantized_moment_block_size);
using QuantizedScratch = cuda::std::array<float, quantized_block_threads>;

struct QuantizedAdamWViews {
    cuda::std::span<__nv_bfloat16> weights;
    cuda::std::span<__nv_bfloat16> grads;
    cuda::std::span<unsigned char> state1;
    cuda::std::span<unsigned char> state2;
    cuda::std::span<float> absmax1;
    cuda::std::span<float> absmax2;
    const float* quantiles1;
    const float* quantiles2;
    cuda::std::span<float> clip_state;
    std::int64_t numel;
};

__global__ void __launch_bounds__(quantized_block_threads, 3)
    sr_adamw_8bit_kernel(QuantizedAdamWViews views, AdamWSettings settings)
{
    __shared__ float smem_quantiles1[quantized_moment_codebook_size];
    __shared__ float smem_quantiles2[quantized_moment_codebook_size];
    smem_quantiles1[threadIdx.x] = views.quantiles1[threadIdx.x];
    smem_quantiles2[threadIdx.x] = views.quantiles2[threadIdx.x];
    __shared__ QuantizedScratch scratch1;
    __shared__ QuantizedScratch scratch2;
    __syncthreads();

    const bool withheld = views.clip_state[2] != 0.0F;
    const float coefficient = views.clip_state[0];
    const float step_size = settings.lr / settings.bias_correction1;
    const float decay = 1.0F - (settings.lr * settings.weight_decay);
    const cuda::std::uint64_t base = (settings.seed * seed_multiplier) ^ (settings.step << 40U);
    const std::int64_t numel = views.numel;
    const std::int64_t num_blocks =
        (numel + quantized_moment_block_size - 1) / quantized_moment_block_size;

    for (std::int64_t block = blockIdx.x; block < num_blocks; block += gridDim.x) {
        const std::int64_t index = (block * quantized_moment_block_size) + threadIdx.x;
        const bool valid = index < numel;
        const auto position = static_cast<std::size_t>(index);

        float gradient = 0.0F;
        float weight = 0.0F;
        unsigned char code1 = 128;
        unsigned char code2 = 0;
        if (valid) {
            gradient = __bfloat162float(views.grads[position]) * coefficient;
            weight = __bfloat162float(views.weights[position]);
            code1 = views.state1[position];
            code2 = views.state2[position];
        }

        const float old_absmax1 = views.absmax1[static_cast<std::size_t>(block)];
        const float old_absmax2 = views.absmax2[static_cast<std::size_t>(block)];
        float first = smem_quantiles1[code1] * old_absmax1;
        float second = smem_quantiles2[code2] * old_absmax2;

        if (!valid) {
            first = 0.0F;
            second = 0.0F;
        } else if (!withheld) {
            first = (settings.beta1 * first) + (settings.one_minus_beta1 * gradient);
            second = (settings.beta2 * second) + (settings.one_minus_beta2 * gradient * gradient);
        }

        const float local_abs1 = fabsf(first);
        const float local_abs2 = fabsf(second);
        const float new_absmax1 = fmaxf(block_max<quantized_block_threads>(local_abs1, scratch1), 1.0e-12F);
        __syncthreads();
        const float new_absmax2 = fmaxf(block_max<quantized_block_threads>(local_abs2, scratch2), 1.0e-12F);
        __syncthreads();

        if (valid) {
            if (!withheld) {
                const float denominator = (sqrtf(second) / settings.bias_correction2_sqrt) + settings.eps;
                weight = (weight * decay) - (step_size * first / denominator);
                const cuda::std::uint64_t key = base ^ static_cast<cuda::std::uint64_t>(index);
                views.weights[position] = round_stochastic(weight, mix_bits(key));

                unsigned char new_code1 = quantize_nearest(smem_quantiles1, first / new_absmax1);
                if (signbit(smem_quantiles1[new_code1]) != signbit(first)) {
                    new_code1 = static_cast<unsigned char>(first > 0.0F ? new_code1 + 1 : new_code1 - 1);
                }
                views.state1[position] = new_code1;
                views.state2[position] = quantize_nearest(smem_quantiles2, second / new_absmax2);
            }
            views.grads[position] = __float2bfloat16(0.0F);
        }
        if (threadIdx.x == 0) {
            views.absmax1[static_cast<std::size_t>(block)] = new_absmax1;
            views.absmax2[static_cast<std::size_t>(block)] = new_absmax2;
        }
        __syncthreads();
    }
}

}

std::int64_t adamw_reduce_blocks(int multiprocessors)
{
    return static_cast<std::int64_t>(multiprocessors) * 4;
}

cudaError_t launch_sr_adamw(const AdamWBuffers& buffers, const AdamWSettings& settings, const LaunchContext& context)
{
    const auto count = static_cast<std::size_t>(buffers.numel / adamw_vector_width);
    const AdamWViews views{
        cuda::std::span<BFloat16Vector>(static_cast<BFloat16Vector*>(buffers.weights), count),
        cuda::std::span<BFloat16Vector>(static_cast<BFloat16Vector*>(buffers.grads), count),
        cuda::std::span<BFloat16Vector>(static_cast<BFloat16Vector*>(buffers.exp_avg), count),
        cuda::std::span<BFloat16Vector>(static_cast<BFloat16Vector*>(buffers.exp_avg_sq), count),
        cuda::std::span<float>(buffers.clip_state, 3),
        cuda::std::span<float>(buffers.partial, static_cast<std::size_t>(buffers.partial_count))};
    const GradNormViews grad_views{views.grads, views.clip_state, views.partial};
    const auto reduce_blocks = static_cast<unsigned int>(buffers.partial_count);
    const auto update_blocks = static_cast<unsigned int>(context.multiprocessors) * 8U;
    gradient_square_kernel<<<reduce_blocks, reduce_threads, 0, context.stream>>>(grad_views);
    cudaError_t status = cudaGetLastError();
    if (status != cudaSuccess) {
        return status;
    }
    gradient_norm_kernel<<<1, reduce_threads, 0, context.stream>>>(grad_views, settings.max_norm);
    status = cudaGetLastError();
    if (status != cudaSuccess) {
        return status;
    }
    sr_adamw_kernel<<<update_blocks, update_threads, 0, context.stream>>>(views, settings);
    return cudaGetLastError();
}

std::int64_t quantized_adamw_reduce_blocks(int multiprocessors)
{
    return adamw_reduce_blocks(multiprocessors);
}

cudaError_t launch_sr_adamw_8bit(
    const QuantizedAdamWBuffers& buffers, const AdamWSettings& settings, const LaunchContext& context)
{
    const auto vector_count = static_cast<std::size_t>(buffers.numel / adamw_vector_width);
    const GradNormViews grad_views{
        cuda::std::span<BFloat16Vector>(static_cast<BFloat16Vector*>(buffers.grads), vector_count),
        cuda::std::span<float>(buffers.clip_state, 3),
        cuda::std::span<float>(buffers.partial, static_cast<std::size_t>(buffers.partial_count))};
    const auto reduce_blocks = static_cast<unsigned int>(buffers.partial_count);
    gradient_square_kernel<<<reduce_blocks, reduce_threads, 0, context.stream>>>(grad_views);
    cudaError_t status = cudaGetLastError();
    if (status != cudaSuccess) {
        return status;
    }
    gradient_norm_kernel<<<1, reduce_threads, 0, context.stream>>>(grad_views, settings.max_norm);
    status = cudaGetLastError();
    if (status != cudaSuccess) {
        return status;
    }

    const auto numel = static_cast<std::size_t>(buffers.numel);
    const std::int64_t total_blocks =
        (buffers.numel + quantized_moment_block_size - 1) / quantized_moment_block_size;
    const QuantizedAdamWViews views{
        cuda::std::span<__nv_bfloat16>(static_cast<__nv_bfloat16*>(buffers.weights), numel),
        cuda::std::span<__nv_bfloat16>(static_cast<__nv_bfloat16*>(buffers.grads), numel),
        cuda::std::span<unsigned char>(buffers.state1, numel),
        cuda::std::span<unsigned char>(buffers.state2, numel),
        cuda::std::span<float>(buffers.absmax1, static_cast<std::size_t>(total_blocks)),
        cuda::std::span<float>(buffers.absmax2, static_cast<std::size_t>(total_blocks)),
        buffers.quantiles1,
        buffers.quantiles2,
        cuda::std::span<float>(buffers.clip_state, 3),
        buffers.numel};
    const auto update_blocks = static_cast<unsigned int>(cuda::std::min<std::int64_t>(
        total_blocks, static_cast<std::int64_t>(context.multiprocessors) * 3));
    sr_adamw_8bit_kernel<<<update_blocks, quantized_block_threads, 0, context.stream>>>(views, settings);
    return cudaGetLastError();
}

}
