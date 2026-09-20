#include "device_storage.cuh"
#include "kernels.cuh"

#include <cuda/std/algorithm>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <driver_types.h>

#include <cstddef>
#include <cstdint>

namespace lfm2_kernels {
namespace {

constexpr int block_threads = 256;
constexpr int swiglu_vector_width = 8;

using BFloat16Vec8 = BFloat16Vector;

template <typename Storage>
struct SwiGluForwardViews {
    const Storage* gate;
    const Storage* up;
    Storage* output;
    std::int64_t numel;
};

template <typename Storage>
struct SwiGluBackwardViews {
    const Storage* gate;
    const Storage* up;
    const Storage* grad_output;
    Storage* grad_gate;
    Storage* grad_up;
    std::int64_t numel;
};

__device__ __forceinline__ float sigmoid(float value)
{
    return 1.0F / (1.0F + expf(-value));
}

__device__ __forceinline__ float silu_of(float gate)
{
    return gate * sigmoid(gate);
}

__device__ __forceinline__ float dsilu_of(float gate, float sig)
{
    return sig * (1.0F + (gate * (1.0F - sig)));
}

template <typename Storage>
__global__ void __launch_bounds__(block_threads) swiglu_forward_kernel(SwiGluForwardViews<Storage> views)
{
    const auto stride = static_cast<std::int64_t>(gridDim.x) * block_threads;
    for (std::int64_t index = (static_cast<std::int64_t>(blockIdx.x) * block_threads) + threadIdx.x;
         index < views.numel; index += stride) {
        const auto position = static_cast<std::size_t>(index);
        const float gate = to_float(views.gate[position]);
        const float up = to_float(views.up[position]);
        views.output[position] = from_float<Storage>(silu_of(gate) * up);
    }
}

template <typename Storage>
__global__ void __launch_bounds__(block_threads) swiglu_backward_kernel(SwiGluBackwardViews<Storage> views)
{
    const auto stride = static_cast<std::int64_t>(gridDim.x) * block_threads;
    for (std::int64_t index = (static_cast<std::int64_t>(blockIdx.x) * block_threads) + threadIdx.x;
         index < views.numel; index += stride) {
        const auto position = static_cast<std::size_t>(index);
        const float gate = to_float(views.gate[position]);
        const float up = to_float(views.up[position]);
        const float upstream = to_float(views.grad_output[position]);
        const float sig = sigmoid(gate);
        views.grad_up[position] = from_float<Storage>(upstream * silu_of(gate));
        views.grad_gate[position] = from_float<Storage>(upstream * up * dsilu_of(gate, sig));
    }
}

__global__ void __launch_bounds__(block_threads) swiglu_forward_bf16x8_kernel(
    const BFloat16Vec8* gate, const BFloat16Vec8* up, BFloat16Vec8* output, std::int64_t vectors)
{
    const auto stride = static_cast<std::int64_t>(gridDim.x) * block_threads;
    for (std::int64_t index = (static_cast<std::int64_t>(blockIdx.x) * block_threads) + threadIdx.x;
         index < vectors; index += stride) {
        const auto position = static_cast<std::size_t>(index);
        const BFloat16Vec8 gate_chunk = gate[position];
        const BFloat16Vec8 up_chunk = up[position];
        BFloat16Vec8 result{};
#pragma unroll
        for (int slot = 0; slot < swiglu_vector_width; ++slot) {
            const float gate_value = __bfloat162float(gate_chunk.lane[slot]);
            const float up_value = __bfloat162float(up_chunk.lane[slot]);
            result.lane[slot] = __float2bfloat16_rn(silu_of(gate_value) * up_value);
        }
        output[position] = result;
    }
}

__global__ void __launch_bounds__(block_threads) swiglu_backward_bf16x8_kernel(
    const BFloat16Vec8* gate, const BFloat16Vec8* up, const BFloat16Vec8* grad_output,
    BFloat16Vec8* grad_gate, BFloat16Vec8* grad_up, std::int64_t vectors)
{
    const auto stride = static_cast<std::int64_t>(gridDim.x) * block_threads;
    for (std::int64_t index = (static_cast<std::int64_t>(blockIdx.x) * block_threads) + threadIdx.x;
         index < vectors; index += stride) {
        const auto position = static_cast<std::size_t>(index);
        const BFloat16Vec8 gate_chunk = gate[position];
        const BFloat16Vec8 up_chunk = up[position];
        const BFloat16Vec8 upstream_chunk = grad_output[position];
        BFloat16Vec8 grad_gate_chunk{};
        BFloat16Vec8 grad_up_chunk{};
#pragma unroll
        for (int slot = 0; slot < swiglu_vector_width; ++slot) {
            const float gate_value = __bfloat162float(gate_chunk.lane[slot]);
            const float up_value = __bfloat162float(up_chunk.lane[slot]);
            const float upstream = __bfloat162float(upstream_chunk.lane[slot]);
            const float sig = sigmoid(gate_value);
            grad_up_chunk.lane[slot] = __float2bfloat16_rn(upstream * silu_of(gate_value));
            grad_gate_chunk.lane[slot] = __float2bfloat16_rn(upstream * up_value * dsilu_of(gate_value, sig));
        }
        grad_gate[position] = grad_gate_chunk;
        grad_up[position] = grad_up_chunk;
    }
}

bool vectorizable_bf16(const void* gate, const void* up, const void* third, std::int64_t numel)
{
    constexpr std::uintptr_t alignment = swiglu_vector_width * sizeof(std::uint16_t);
    return (numel % swiglu_vector_width) == 0
        && (reinterpret_cast<std::uintptr_t>(gate) % alignment) == 0
        && (reinterpret_cast<std::uintptr_t>(up) % alignment) == 0
        && (third == nullptr || (reinterpret_cast<std::uintptr_t>(third) % alignment) == 0);
}

template <typename Body>
cudaError_t dispatch_storage(StorageType storage, Body&& body)
{
    switch (storage) {
    case StorageType::bfloat16:
        return body(__nv_bfloat16{});
    case StorageType::float32:
        return body(float{});
    }
    return cudaErrorInvalidValue;
}

unsigned int elementwise_blocks(std::int64_t work_items, int multiprocessors)
{
    const std::int64_t wanted = (work_items + block_threads - 1) / block_threads;
    return static_cast<unsigned int>(cuda::std::max<std::int64_t>(
        1, cuda::std::min<std::int64_t>(wanted, static_cast<std::int64_t>(multiprocessors) * 32)));
}

}

cudaError_t launch_swiglu_forward(const SwiGluForward& request, const LaunchContext& context)
{
    if (request.storage == StorageType::bfloat16
        && vectorizable_bf16(request.gate, request.up, request.output, request.numel)) {
        const auto vectors = request.numel / swiglu_vector_width;
        const auto blocks = elementwise_blocks(vectors, context.multiprocessors);
        swiglu_forward_bf16x8_kernel<<<blocks, block_threads, 0, context.stream>>>(
            static_cast<const BFloat16Vec8*>(request.gate), static_cast<const BFloat16Vec8*>(request.up),
            static_cast<BFloat16Vec8*>(request.output), vectors);
        return cudaGetLastError();
    }
    const auto blocks = elementwise_blocks(request.numel, context.multiprocessors);
    return dispatch_storage(request.storage, [&](auto storage_tag) {
        using Storage = decltype(storage_tag);
        const SwiGluForwardViews<Storage> views{
            static_cast<const Storage*>(request.gate),
            static_cast<const Storage*>(request.up),
            static_cast<Storage*>(request.output),
            request.numel};
        swiglu_forward_kernel<Storage><<<blocks, block_threads, 0, context.stream>>>(views);
        return cudaGetLastError();
    });
}

cudaError_t launch_swiglu_backward(const SwiGluBackward& request, const LaunchContext& context)
{
    if (request.storage == StorageType::bfloat16
        && vectorizable_bf16(request.gate, request.up, request.grad_output, request.numel)
        && vectorizable_bf16(request.grad_gate, request.grad_up, nullptr, request.numel)) {
        const auto vectors = request.numel / swiglu_vector_width;
        const auto blocks = elementwise_blocks(vectors, context.multiprocessors);
        swiglu_backward_bf16x8_kernel<<<blocks, block_threads, 0, context.stream>>>(
            static_cast<const BFloat16Vec8*>(request.gate), static_cast<const BFloat16Vec8*>(request.up),
            static_cast<const BFloat16Vec8*>(request.grad_output), static_cast<BFloat16Vec8*>(request.grad_gate),
            static_cast<BFloat16Vec8*>(request.grad_up), vectors);
        return cudaGetLastError();
    }
    const auto blocks = elementwise_blocks(request.numel, context.multiprocessors);
    return dispatch_storage(request.storage, [&](auto storage_tag) {
        using Storage = decltype(storage_tag);
        const SwiGluBackwardViews<Storage> views{
            static_cast<const Storage*>(request.gate),
            static_cast<const Storage*>(request.up),
            static_cast<const Storage*>(request.grad_output),
            static_cast<Storage*>(request.grad_gate),
            static_cast<Storage*>(request.grad_up),
            request.numel};
        swiglu_backward_kernel<Storage><<<blocks, block_threads, 0, context.stream>>>(views);
        return cudaGetLastError();
    });
}

}
