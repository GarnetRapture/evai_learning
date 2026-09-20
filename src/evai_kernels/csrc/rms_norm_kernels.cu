#include "device_storage.cuh"
#include "kernels.cuh"

#include <cuda/std/algorithm>
#include <cuda/std/array>
#include <cuda/std/mdspan>
#include <cuda/std/span>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <driver_types.h>

#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace lfm2_kernels {
namespace {

constexpr int warp_lanes = 32;
constexpr int warps_per_block = 8;
constexpr int block_threads = warps_per_block * warp_lanes;
constexpr unsigned int full_warp = 0xffffffffU;

template <typename Element, int PerLane>
using Rows = cuda::std::mdspan<Element, cuda::std::extents<std::int64_t, cuda::std::dynamic_extent, PerLane * warp_lanes>>;

template <typename Storage, int PerLane>
struct RmsForwardViews {
    Rows<const Storage, PerLane> input;
    cuda::std::span<const Storage, PerLane * warp_lanes> weight;
    Rows<Storage, PerLane> output;
    cuda::std::span<float> inverse_rms;
    float epsilon;
};

template <typename Storage, int PerLane>
struct RmsBackwardViews {
    Rows<const Storage, PerLane> grad_output;
    Rows<const Storage, PerLane> input;
    cuda::std::span<const Storage, PerLane * warp_lanes> weight;
    cuda::std::span<const float> inverse_rms;
    Rows<Storage, PerLane> grad_input;
    Rows<float, PerLane> grad_weight_partial;
};

struct WarpLane {
    std::int64_t lane;
    std::int64_t warp;
    std::int64_t warps;
};

__device__ WarpLane current_warp_lane()
{
    const auto thread = static_cast<std::int64_t>(threadIdx.x);
    return WarpLane{
        thread % warp_lanes,
        (static_cast<std::int64_t>(blockIdx.x) * warps_per_block) + (thread / warp_lanes),
        static_cast<std::int64_t>(gridDim.x) * warps_per_block};
}

__device__ float warp_sum(float value)
{
#pragma unroll
    for (int offset = warp_lanes / 2; offset > 0; offset /= 2) {
        value += __shfl_xor_sync(full_warp, value, offset);
    }
    return value;
}

template <typename Storage>
__device__ float storage_rounded(float value)
{
    return to_float(from_float<Storage>(value));
}

template <typename Storage, int PerLane>
__global__ void __launch_bounds__(block_threads) rms_norm_forward_kernel(RmsForwardViews<Storage, PerLane> views)
{
    constexpr std::int64_t columns = PerLane * warp_lanes;
    const WarpLane place = current_warp_lane();
    cuda::std::array<float, PerLane> weight{};
#pragma unroll
    for (int slot = 0; slot < PerLane; ++slot) {
        weight[slot] = to_float(views.weight[static_cast<std::size_t>(place.lane + (slot * warp_lanes))]);
    }
    const std::int64_t rows = views.input.extent(0);
    for (std::int64_t row = place.warp; row < rows; row += place.warps) {
        cuda::std::array<float, PerLane> values{};
        float squares = 0.0F;
#pragma unroll
        for (int slot = 0; slot < PerLane; ++slot) {
            values[slot] = to_float(views.input[Index2{row, place.lane + (slot * warp_lanes)}]);
            squares += values[slot] * values[slot];
        }
        const float inverse = rsqrtf((warp_sum(squares) / static_cast<float>(columns)) + views.epsilon);
#pragma unroll
        for (int slot = 0; slot < PerLane; ++slot) {
            const float normalized = storage_rounded<Storage>(values[slot] * inverse);
            views.output[Index2{row, place.lane + (slot * warp_lanes)}] = from_float<Storage>(weight[slot] * normalized);
        }
        if (place.lane == 0) {
            views.inverse_rms[static_cast<std::size_t>(row)] = inverse;
        }
    }
}

template <typename Storage, int PerLane>
__global__ void __launch_bounds__(block_threads) rms_norm_backward_kernel(RmsBackwardViews<Storage, PerLane> views)
{
    constexpr std::int64_t columns = PerLane * warp_lanes;
    const WarpLane place = current_warp_lane();
    cuda::std::array<float, PerLane> weight{};
#pragma unroll
    for (int slot = 0; slot < PerLane; ++slot) {
        weight[slot] = to_float(views.weight[static_cast<std::size_t>(place.lane + (slot * warp_lanes))]);
    }
    cuda::std::array<float, PerLane> weight_sums{};
    const std::int64_t rows = views.input.extent(0);
    for (std::int64_t row = place.warp; row < rows; row += place.warps) {
        const float inverse = views.inverse_rms[static_cast<std::size_t>(row)];
        cuda::std::array<float, PerLane> values{};
        cuda::std::array<float, PerLane> upstream{};
        float dot = 0.0F;
#pragma unroll
        for (int slot = 0; slot < PerLane; ++slot) {
            const Index2 at{row, place.lane + (slot * warp_lanes)};
            values[slot] = to_float(views.input[at]);
            upstream[slot] = to_float(views.grad_output[at]);
            dot += upstream[slot] * weight[slot] * values[slot];
        }
        const float coefficient = inverse * inverse * inverse * warp_sum(dot) / static_cast<float>(columns);
#pragma unroll
        for (int slot = 0; slot < PerLane; ++slot) {
            const Index2 at{row, place.lane + (slot * warp_lanes)};
            const float gradient = (weight[slot] * upstream[slot] * inverse) - (values[slot] * coefficient);
            views.grad_input[at] = from_float<Storage>(gradient);
            weight_sums[slot] += upstream[slot] * storage_rounded<Storage>(values[slot] * inverse);
        }
    }
    if (place.warp < views.grad_weight_partial.extent(0)) {
#pragma unroll
        for (int slot = 0; slot < PerLane; ++slot) {
            views.grad_weight_partial[Index2{place.warp, place.lane + (slot * warp_lanes)}] = weight_sums[slot];
        }
    }
}

template <typename Body>
cudaError_t dispatch_rows(StorageType storage, std::int64_t columns, Body&& body)
{
    const auto with_width = [&](auto storage_tag) {
        switch (columns) {
        case 32:
            return body(storage_tag, std::integral_constant<int, 1>{});
        case 64:
            return body(storage_tag, std::integral_constant<int, 2>{});
        case 128:
            return body(storage_tag, std::integral_constant<int, 4>{});
        case 256:
            return body(storage_tag, std::integral_constant<int, 8>{});
        case 512:
            return body(storage_tag, std::integral_constant<int, 16>{});
        case 1024:
            return body(storage_tag, std::integral_constant<int, 32>{});
        default:
            return cudaErrorInvalidValue;
        }
    };
    switch (storage) {
    case StorageType::bfloat16:
        return with_width(__nv_bfloat16{});
    case StorageType::float32:
        return with_width(float{});
    }
    return cudaErrorInvalidValue;
}

}

std::int64_t rms_norm_backward_partial_rows(int multiprocessors)
{
    return static_cast<std::int64_t>(multiprocessors) * 4 * warps_per_block;
}

cudaError_t launch_rms_norm_forward(const RmsNormForward& request, const LaunchContext& context)
{
    const std::int64_t wanted = (request.rows + warps_per_block - 1) / warps_per_block;
    const auto blocks = static_cast<unsigned int>(cuda::std::max<std::int64_t>(
        1, cuda::std::min<std::int64_t>(wanted, static_cast<std::int64_t>(context.multiprocessors) * 16)));
    return dispatch_rows(request.storage, request.columns, [&](auto storage_tag, auto width_tag) {
        using Storage = decltype(storage_tag);
        constexpr int PerLane = decltype(width_tag)::value;
        constexpr std::size_t width = PerLane * warp_lanes;
        const RmsForwardViews<Storage, PerLane> views{
            Rows<const Storage, PerLane>(static_cast<const Storage*>(request.input), request.rows),
            cuda::std::span<const Storage, width>(static_cast<const Storage*>(request.weight), width),
            Rows<Storage, PerLane>(static_cast<Storage*>(request.output), request.rows),
            cuda::std::span<float>(request.inverse_rms, static_cast<std::size_t>(request.rows)),
            request.epsilon};
        rms_norm_forward_kernel<Storage, PerLane><<<blocks, block_threads, 0, context.stream>>>(views);
        return cudaGetLastError();
    });
}

cudaError_t launch_rms_norm_backward(const RmsNormBackward& request, const LaunchContext& context)
{
    const auto blocks = static_cast<unsigned int>(request.partial_rows / warps_per_block);
    return dispatch_rows(request.storage, request.columns, [&](auto storage_tag, auto width_tag) {
        using Storage = decltype(storage_tag);
        constexpr int PerLane = decltype(width_tag)::value;
        constexpr std::size_t width = PerLane * warp_lanes;
        const RmsBackwardViews<Storage, PerLane> views{
            Rows<const Storage, PerLane>(static_cast<const Storage*>(request.grad_output), request.rows),
            Rows<const Storage, PerLane>(static_cast<const Storage*>(request.input), request.rows),
            cuda::std::span<const Storage, width>(static_cast<const Storage*>(request.weight), width),
            cuda::std::span<const float>(request.inverse_rms, static_cast<std::size_t>(request.rows)),
            Rows<Storage, PerLane>(static_cast<Storage*>(request.grad_input), request.rows),
            Rows<float, PerLane>(request.grad_weight_partial, request.partial_rows)};
        rms_norm_backward_kernel<Storage, PerLane><<<blocks, block_threads, 0, context.stream>>>(views);
        return cudaGetLastError();
    });
}

const char* describe_cuda_error(cudaError_t status)
{
    return cudaGetErrorString(status);
}

}
