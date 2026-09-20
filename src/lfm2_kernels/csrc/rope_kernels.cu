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

template <typename Storage>
struct RopeViews {
    const Storage* input;
    const Storage* cos;
    const Storage* sin;
    Storage* output;
    std::int64_t rows;
    std::int64_t heads;
    std::int64_t seq_len;
    float sin_sign;
};

template <typename Storage, int HeadDim>
__global__ void __launch_bounds__(HeadDim) rope_apply_kernel(RopeViews<Storage> views)
{
    __shared__ float row_values[HeadDim];
    constexpr int half = HeadDim / 2;
    const int lane = static_cast<int>(threadIdx.x);
    for (std::int64_t row = blockIdx.x; row < views.rows; row += gridDim.x) {
        const std::int64_t base = row * HeadDim;
        row_values[lane] = to_float(views.input[static_cast<std::size_t>(base + lane)]);
        __syncthreads();

        const std::int64_t batch = row / (views.heads * views.seq_len);
        const std::int64_t position = row % views.seq_len;
        const std::int64_t angle_base = ((batch * views.seq_len) + position) * HeadDim;
        const float cos_value = to_float(views.cos[static_cast<std::size_t>(angle_base + lane)]);
        const float sin_value =
            to_float(views.sin[static_cast<std::size_t>(angle_base + lane)]) * views.sin_sign;
        const float rotated = (lane < half) ? -row_values[lane + half] : row_values[lane - half];
        views.output[static_cast<std::size_t>(base + lane)] =
            from_float<Storage>((row_values[lane] * cos_value) + (rotated * sin_value));
        __syncthreads();
    }
}

template <typename Body>
cudaError_t dispatch_head_dim(StorageType storage, std::int64_t head_dim, Body&& body)
{
    const auto with_dim = [&](auto storage_tag) {
        switch (head_dim) {
        case 64:
            return body(storage_tag, std::integral_constant<int, 64>{});
        case 128:
            return body(storage_tag, std::integral_constant<int, 128>{});
        case 256:
            return body(storage_tag, std::integral_constant<int, 256>{});
        default:
            return cudaErrorInvalidValue;
        }
    };
    switch (storage) {
    case StorageType::bfloat16:
        return with_dim(__nv_bfloat16{});
    case StorageType::float32:
        return with_dim(float{});
    }
    return cudaErrorInvalidValue;
}

}

cudaError_t launch_rope_apply(const RopeApply& request, const LaunchContext& context)
{
    const auto blocks = static_cast<unsigned int>(
        cuda::std::max<std::int64_t>(1, cuda::std::min<std::int64_t>(request.rows, static_cast<std::int64_t>(context.multiprocessors) * 32)));
    return dispatch_head_dim(request.storage, request.head_dim, [&](auto storage_tag, auto dim_tag) {
        using Storage = decltype(storage_tag);
        constexpr int HeadDim = decltype(dim_tag)::value;
        const RopeViews<Storage> views{
            static_cast<const Storage*>(request.input),
            static_cast<const Storage*>(request.cos),
            static_cast<const Storage*>(request.sin),
            static_cast<Storage*>(request.output),
            request.rows,
            request.heads,
            request.seq_len,
            request.negate_sin ? -1.0F : 1.0F};
        rope_apply_kernel<Storage, HeadDim><<<blocks, HeadDim, 0, context.stream>>>(views);
        return cudaGetLastError();
    });
}

}
