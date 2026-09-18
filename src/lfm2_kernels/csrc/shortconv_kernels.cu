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
#include <vector_types.h>

#include <algorithm>
#include <cstdint>
#include <type_traits>

namespace lfm2_kernels {
namespace {

constexpr std::int64_t channel_threads = 256;

template <int Taps>
using TapWeights = cuda::std::mdspan<const float, cuda::std::extents<std::int64_t, cuda::std::dynamic_extent, Taps>>;

template <int Taps>
using WeightPartial = cuda::std::mdspan<
    float,
    cuda::std::extents<std::int64_t, cuda::std::dynamic_extent, cuda::std::dynamic_extent, Taps>>;

using BiasPartial = cuda::std::mdspan<float, cuda::std::dextents<std::int64_t, 2>>;

template <typename Storage>
struct ProjectionRow {
    Strided3<const Storage> projection;
    std::int64_t batch;
    std::int64_t channel;
    std::int64_t channels;

    __device__ float gate(std::int64_t token) const
    {
        return to_float(projection[Index3{batch, token, channel}]);
    }

    __device__ float carrier(std::int64_t token) const
    {
        return to_float(projection[Index3{batch, token, channels + channel}]);
    }

    __device__ float value(std::int64_t token) const
    {
        return to_float(projection[Index3{batch, token, (2 * channels) + channel}]);
    }

    __device__ float gated(std::int64_t token) const
    {
        return gate(token) * value(token);
    }
};

template <typename Storage, int Taps>
struct ForwardViews {
    Strided3<const Storage> projection;
    TapWeights<Taps> weight;
    cuda::std::span<const float> bias;
    Strided3<const Storage> state;
    bool has_state;
    Strided3<Storage> output;
    std::int64_t chunk;
};

template <typename Storage, int Taps>
struct BackwardViews {
    Strided3<const Storage> projection;
    TapWeights<Taps> weight;
    cuda::std::span<const float> bias;
    Strided3<const Storage> grad_output;
    Strided3<Storage> grad_projection;
    WeightPartial<Taps> grad_weight_partial;
    BiasPartial grad_bias_partial;
    bool has_bias;
    std::int64_t chunk;
    std::int64_t chunks;
};

template <typename Storage>
struct StateViews {
    Strided3<const Storage> projection;
    Strided3<Storage> state;
    bool has_history;
};

struct Lane {
    std::int64_t channel;
    std::int64_t batch;
    std::int64_t chunk_index;
};

__device__ Lane current_lane()
{
    return Lane{
        (static_cast<std::int64_t>(blockIdx.x) * channel_threads) + static_cast<std::int64_t>(threadIdx.x),
        static_cast<std::int64_t>(blockIdx.z),
        static_cast<std::int64_t>(blockIdx.y)};
}

template <int Taps>
__device__ cuda::std::array<float, Taps> load_taps(TapWeights<Taps> weight, std::int64_t channel)
{
    cuda::std::array<float, Taps> taps{};
#pragma unroll
    for (int tap = 0; tap < Taps; ++tap) {
        taps[tap] = weight[Index2{channel, tap}];
    }
    return taps;
}

template <typename Storage, int Taps>
__global__ void __launch_bounds__(channel_threads) shortconv_forward_kernel(ForwardViews<Storage, Taps> views)
{
    const Lane lane = current_lane();
    const std::int64_t channels = views.output.extent(2);
    const std::int64_t seq_len = views.output.extent(1);
    const std::int64_t begin = lane.chunk_index * views.chunk;
    const std::int64_t end = cuda::std::min(begin + views.chunk, seq_len);
    if (lane.channel >= channels || begin >= end) {
        return;
    }
    const ProjectionRow<Storage> row{views.projection, lane.batch, lane.channel, channels};
    const cuda::std::array<float, Taps> taps = load_taps<Taps>(views.weight, lane.channel);
    const float shift = views.bias.empty() ? 0.0F : views.bias[lane.channel];
    const std::int64_t history = views.state.extent(2);

    cuda::std::array<float, Taps> window{};
#pragma unroll
    for (int tap = 0; tap < Taps - 1; ++tap) {
        const std::int64_t source = begin - (Taps - 1) + tap;
        if (source >= 0) {
            window[tap] = row.gated(source);
        } else if (views.has_state) {
            window[tap] = to_float(views.state[Index3{lane.batch, lane.channel, history + source}]);
        }
    }

    for (std::int64_t token = begin; token < end; ++token) {
        window[Taps - 1] = row.gated(token);
        float convolved = shift;
#pragma unroll
        for (int tap = 0; tap < Taps; ++tap) {
            convolved += taps[tap] * window[tap];
        }
        views.output[Index3{lane.batch, token, lane.channel}] = from_float<Storage>(row.carrier(token) * convolved);
#pragma unroll
        for (int tap = 0; tap < Taps - 1; ++tap) {
            window[tap] = window[tap + 1];
        }
    }
}

template <typename Storage, int Taps>
__global__ void __launch_bounds__(channel_threads) shortconv_backward_kernel(BackwardViews<Storage, Taps> views)
{
    const Lane lane = current_lane();
    const std::int64_t channels = views.grad_projection.extent(2) / 3;
    const std::int64_t seq_len = views.grad_projection.extent(1);
    const std::int64_t begin = lane.chunk_index * views.chunk;
    const std::int64_t end = cuda::std::min(begin + views.chunk, seq_len);
    if (lane.channel >= channels) {
        return;
    }
    const std::int64_t partial_row = (lane.batch * views.chunks) + lane.chunk_index;

    cuda::std::array<float, Taps> weight_sums{};
    float bias_sum = 0.0F;

    if (begin < end) {
        const ProjectionRow<Storage> row{views.projection, lane.batch, lane.channel, channels};
        const cuda::std::array<float, Taps> taps = load_taps<Taps>(views.weight, lane.channel);
        const float shift = views.bias.empty() ? 0.0F : views.bias[lane.channel];
        const std::int64_t last = end - 1;

        const auto upstream = [&](std::int64_t token) {
            return to_float(views.grad_output[Index3{lane.batch, token, lane.channel}]);
        };

        cuda::std::array<float, Taps> future{};
#pragma unroll
        for (int ahead = 1; ahead < Taps; ++ahead) {
            const std::int64_t token = last + ahead;
            future[ahead] = token < seq_len ? upstream(token) * row.carrier(token) : 0.0F;
        }
        cuda::std::array<float, Taps> past{};
#pragma unroll
        for (int tap = 0; tap < Taps; ++tap) {
            const std::int64_t source = last - (Taps - 1) + tap;
            past[tap] = source >= 0 ? row.gated(source) : 0.0F;
        }

        for (std::int64_t token = last; token >= begin; --token) {
            const float incoming = upstream(token);
            float convolved = shift;
#pragma unroll
            for (int tap = 0; tap < Taps; ++tap) {
                convolved += taps[tap] * past[tap];
            }
            future[0] = incoming * row.carrier(token);
            float grad_gated = 0.0F;
#pragma unroll
            for (int tap = 0; tap < Taps; ++tap) {
                grad_gated += taps[tap] * future[Taps - 1 - tap];
            }
            views.grad_projection[Index3{lane.batch, token, lane.channel}] =
                from_float<Storage>(grad_gated * row.value(token));
            views.grad_projection[Index3{lane.batch, token, channels + lane.channel}] =
                from_float<Storage>(incoming * convolved);
            views.grad_projection[Index3{lane.batch, token, (2 * channels) + lane.channel}] =
                from_float<Storage>(grad_gated * row.gate(token));
#pragma unroll
            for (int tap = 0; tap < Taps; ++tap) {
                weight_sums[tap] += future[0] * past[tap];
            }
            bias_sum += future[0];
#pragma unroll
            for (int ahead = Taps - 1; ahead > 0; --ahead) {
                future[ahead] = future[ahead - 1];
            }
#pragma unroll
            for (int tap = Taps - 1; tap > 0; --tap) {
                past[tap] = past[tap - 1];
            }
            const std::int64_t incoming_source = token - Taps;
            past[0] = incoming_source >= 0 ? row.gated(incoming_source) : 0.0F;
        }
    }

#pragma unroll
    for (int tap = 0; tap < Taps; ++tap) {
        views.grad_weight_partial[Index3{partial_row, lane.channel, tap}] = weight_sums[tap];
    }
    if (views.has_bias) {
        views.grad_bias_partial[Index2{partial_row, lane.channel}] = bias_sum;
    }
}

template <typename Storage, int Taps>
__global__ void __launch_bounds__(channel_threads) shortconv_state_kernel(StateViews<Storage> views)
{
    const std::int64_t channel =
        (static_cast<std::int64_t>(blockIdx.x) * channel_threads) + static_cast<std::int64_t>(threadIdx.x);
    const std::int64_t batch = static_cast<std::int64_t>(blockIdx.y);
    const std::int64_t channels = views.state.extent(1);
    if (channel >= channels) {
        return;
    }
    const std::int64_t seq_len = views.projection.extent(1);
    const ProjectionRow<Storage> row{views.projection, batch, channel, channels};
#pragma unroll
    for (int slot = 0; slot < Taps; ++slot) {
        const std::int64_t position = seq_len + slot - Taps;
        float value = 0.0F;
        if (position >= 0) {
            value = row.gated(position);
        } else if (views.has_history) {
            value = to_float(views.state[Index3{batch, channel, Taps + position}]);
        }
        views.state[Index3{batch, channel, slot}] = from_float<Storage>(value);
    }
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

template <typename Body>
cudaError_t dispatch_taps(std::int64_t taps, Body&& body)
{
    switch (taps) {
    case 2:
        return body(std::integral_constant<int, 2>{});
    case 3:
        return body(std::integral_constant<int, 3>{});
    case 4:
        return body(std::integral_constant<int, 4>{});
    default:
        return cudaErrorInvalidValue;
    }
}

std::int64_t channel_blocks(std::int64_t channels)
{
    return (channels + channel_threads - 1) / channel_threads;
}

template <typename Element>
Strided3<Element> view_of(const ConstTensor3& tensor)
{
    return strided_view<Element>(static_cast<Element*>(tensor.data), tensor.sizes, tensor.strides);
}

template <typename Element>
Strided3<Element> view_of(const MutableTensor3& tensor)
{
    return strided_view<Element>(static_cast<Element*>(tensor.data), tensor.sizes, tensor.strides);
}

cuda::std::span<const float> bias_span(const ShortConvParameters& parameters, std::int64_t channels)
{
    return parameters.bias == nullptr
        ? cuda::std::span<const float>{}
        : cuda::std::span<const float>(parameters.bias, static_cast<std::size_t>(channels));
}

}

std::int64_t shortconv_chunk_tokens(
    std::int64_t batch, std::int64_t seq_len, std::int64_t channels, int multiprocessors)
{
    const std::int64_t lanes = std::max<std::int64_t>(1, batch * channel_blocks(channels));
    const std::int64_t wanted_chunks = std::max<std::int64_t>(1, ((multiprocessors * 4LL) + lanes - 1) / lanes);
    return std::max<std::int64_t>(8, (seq_len + wanted_chunks - 1) / wanted_chunks);
}

cudaError_t launch_shortconv_forward(const ShortConvForward& request, const LaunchContext& context)
{
    const std::int64_t batch = request.output.sizes[0];
    const std::int64_t seq_len = request.output.sizes[1];
    const std::int64_t channels = request.output.sizes[2];
    const dim3 grid(
        static_cast<unsigned int>(channel_blocks(channels)),
        static_cast<unsigned int>((seq_len + request.chunk - 1) / request.chunk),
        static_cast<unsigned int>(batch));
    return dispatch_storage(request.storage, [&](auto storage_tag) {
        using Storage = decltype(storage_tag);
        return dispatch_taps(request.parameters.taps, [&](auto taps_tag) {
            constexpr int Taps = decltype(taps_tag)::value;
            const ForwardViews<Storage, Taps> views{
                view_of<const Storage>(request.projection),
                TapWeights<Taps>(request.parameters.weight, channels),
                bias_span(request.parameters, channels),
                view_of<const Storage>(request.state),
                request.state.data != nullptr,
                view_of<Storage>(request.output),
                request.chunk};
            shortconv_forward_kernel<Storage, Taps><<<grid, channel_threads, 0, context.stream>>>(views);
            return cudaGetLastError();
        });
    });
}

cudaError_t launch_shortconv_backward(const ShortConvBackward& request, const LaunchContext& context)
{
    const std::int64_t batch = request.grad_projection.sizes[0];
    const std::int64_t channels = request.grad_projection.sizes[2] / 3;
    const std::int64_t rows = batch * request.chunks;
    const dim3 grid(
        static_cast<unsigned int>(channel_blocks(channels)),
        static_cast<unsigned int>(request.chunks),
        static_cast<unsigned int>(batch));
    return dispatch_storage(request.storage, [&](auto storage_tag) {
        using Storage = decltype(storage_tag);
        return dispatch_taps(request.parameters.taps, [&](auto taps_tag) {
            constexpr int Taps = decltype(taps_tag)::value;
            const bool has_bias = request.parameters.bias != nullptr;
            const BackwardViews<Storage, Taps> views{
                view_of<const Storage>(request.projection),
                TapWeights<Taps>(request.parameters.weight, channels),
                bias_span(request.parameters, channels),
                view_of<const Storage>(request.grad_output),
                view_of<Storage>(request.grad_projection),
                WeightPartial<Taps>(request.grad_weight_partial, rows, channels),
                has_bias ? BiasPartial(request.grad_bias_partial, rows, channels) : BiasPartial{},
                has_bias,
                request.chunk,
                request.chunks};
            shortconv_backward_kernel<Storage, Taps><<<grid, channel_threads, 0, context.stream>>>(views);
            return cudaGetLastError();
        });
    });
}

cudaError_t launch_shortconv_state_update(const ShortConvStateUpdate& request, const LaunchContext& context)
{
    const std::int64_t batch = request.state.sizes[0];
    const std::int64_t channels = request.state.sizes[1];
    const dim3 grid(static_cast<unsigned int>(channel_blocks(channels)), static_cast<unsigned int>(batch));
    return dispatch_storage(request.storage, [&](auto storage_tag) {
        using Storage = decltype(storage_tag);
        return dispatch_taps(request.state.sizes[2], [&](auto taps_tag) {
            constexpr int Taps = decltype(taps_tag)::value;
            const StateViews<Storage> views{
                view_of<const Storage>(request.projection),
                view_of<Storage>(request.state),
                request.has_history};
            shortconv_state_kernel<Storage, Taps><<<grid, channel_threads, 0, context.stream>>>(views);
            return cudaGetLastError();
        });
    });
}

const char* describe_cuda_error(cudaError_t status)
{
    return cudaGetErrorString(status);
}

}
