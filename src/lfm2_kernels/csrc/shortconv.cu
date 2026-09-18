#include "ops.cuh"
#include "storage.cuh"

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/util/BFloat16.h>
#include <c10/util/Exception.h>

#include <algorithm>
#include <cstdint>
#include <type_traits>

namespace lfm2_kernels {
namespace {

constexpr int channel_threads = 256;

struct ForwardGeometry {
    int seq_len;
    int channels;
    int chunk;
    std::int64_t projection_batch_stride;
    std::int64_t projection_token_stride;
    std::int64_t state_batch_stride;
    std::int64_t state_channel_stride;
    std::int64_t state_token_stride;
    int state_len;
    std::int64_t output_batch_stride;
    std::int64_t output_token_stride;
};

struct BackwardGeometry {
    int seq_len;
    int channels;
    int chunk;
    int chunks;
    std::int64_t projection_batch_stride;
    std::int64_t projection_token_stride;
    std::int64_t grad_output_batch_stride;
    std::int64_t grad_output_token_stride;
    std::int64_t grad_projection_batch_stride;
    std::int64_t grad_projection_token_stride;
};

struct StateGeometry {
    int seq_len;
    int channels;
    std::int64_t projection_batch_stride;
    std::int64_t projection_token_stride;
    std::int64_t state_batch_stride;
    std::int64_t state_channel_stride;
    std::int64_t state_token_stride;
};

template <typename Storage>
struct ProjectionRow {
    const Storage* base;
    std::int64_t token_stride;
    int channels;

    __device__ float gate(int token) const
    {
        return to_float(base[token * token_stride]);
    }

    __device__ float carrier(int token) const
    {
        return to_float(base[token * token_stride + channels]);
    }

    __device__ float value(int token) const
    {
        return to_float(base[token * token_stride + 2 * channels]);
    }

    __device__ float gated(int token) const
    {
        return gate(token) * value(token);
    }
};

template <typename Storage, int Taps>
__global__ void __launch_bounds__(channel_threads) shortconv_forward_kernel(
    const Storage* __restrict__ projection,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    const Storage* __restrict__ state,
    Storage* __restrict__ output,
    ForwardGeometry geometry)
{
    const int channel = blockIdx.x * channel_threads + threadIdx.x;
    const int batch = blockIdx.z;
    const int begin = blockIdx.y * geometry.chunk;
    const int end = min(begin + geometry.chunk, geometry.seq_len);
    if (channel >= geometry.channels || begin >= end) {
        return;
    }
    const ProjectionRow<Storage> row{
        projection + batch * geometry.projection_batch_stride + channel,
        geometry.projection_token_stride,
        geometry.channels};
    const Storage* history = state == nullptr
        ? nullptr
        : state + batch * geometry.state_batch_stride + channel * geometry.state_channel_stride;

    float taps[Taps];
#pragma unroll
    for (int tap = 0; tap < Taps; ++tap) {
        taps[tap] = weight[channel * Taps + tap];
    }
    const float shift = bias == nullptr ? 0.0f : bias[channel];

    float window[Taps];
#pragma unroll
    for (int tap = 0; tap < Taps - 1; ++tap) {
        const int source = begin - (Taps - 1) + tap;
        if (source >= 0) {
            window[tap] = row.gated(source);
        } else if (history != nullptr) {
            window[tap] = to_float(history[(geometry.state_len + source) * geometry.state_token_stride]);
        } else {
            window[tap] = 0.0f;
        }
    }

    Storage* out = output + batch * geometry.output_batch_stride + channel;
    for (int token = begin; token < end; ++token) {
        window[Taps - 1] = row.gated(token);
        float convolved = shift;
#pragma unroll
        for (int tap = 0; tap < Taps; ++tap) {
            convolved += taps[tap] * window[tap];
        }
        out[token * geometry.output_token_stride] = from_float<Storage>(row.carrier(token) * convolved);
#pragma unroll
        for (int tap = 0; tap < Taps - 1; ++tap) {
            window[tap] = window[tap + 1];
        }
    }
}

template <typename Storage, int Taps>
__global__ void __launch_bounds__(channel_threads) shortconv_backward_kernel(
    const Storage* __restrict__ projection,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    const Storage* __restrict__ grad_output,
    Storage* __restrict__ grad_projection,
    float* __restrict__ grad_weight_partial,
    float* __restrict__ grad_bias_partial,
    BackwardGeometry geometry)
{
    const int channel = blockIdx.x * channel_threads + threadIdx.x;
    const int batch = blockIdx.z;
    const int begin = blockIdx.y * geometry.chunk;
    const int end = min(begin + geometry.chunk, geometry.seq_len);
    if (channel >= geometry.channels) {
        return;
    }
    const std::int64_t partial_row = static_cast<std::int64_t>(batch) * geometry.chunks + blockIdx.y;

    float weight_sums[Taps];
#pragma unroll
    for (int tap = 0; tap < Taps; ++tap) {
        weight_sums[tap] = 0.0f;
    }
    float bias_sum = 0.0f;

    if (begin < end) {
        const ProjectionRow<Storage> row{
            projection + batch * geometry.projection_batch_stride + channel,
            geometry.projection_token_stride,
            geometry.channels};
        const Storage* upstream = grad_output + batch * geometry.grad_output_batch_stride + channel;
        Storage* downstream = grad_projection + batch * geometry.grad_projection_batch_stride + channel;

        float taps[Taps];
#pragma unroll
        for (int tap = 0; tap < Taps; ++tap) {
            taps[tap] = weight[channel * Taps + tap];
        }
        const float shift = bias == nullptr ? 0.0f : bias[channel];

        const int last = end - 1;
        float future[Taps];
        future[0] = 0.0f;
#pragma unroll
        for (int ahead = 1; ahead < Taps; ++ahead) {
            const int token = last + ahead;
            future[ahead] = token < geometry.seq_len
                ? to_float(upstream[token * geometry.grad_output_token_stride]) * row.carrier(token)
                : 0.0f;
        }
        float past[Taps];
#pragma unroll
        for (int tap = 0; tap < Taps; ++tap) {
            const int source = last - (Taps - 1) + tap;
            past[tap] = source >= 0 ? row.gated(source) : 0.0f;
        }

        for (int token = last; token >= begin; --token) {
            const float incoming = to_float(upstream[token * geometry.grad_output_token_stride]);
            float convolved = shift;
#pragma unroll
            for (int tap = 0; tap < Taps; ++tap) {
                convolved += taps[tap] * past[tap];
            }
            future[0] = incoming * row.carrier(token);
            float grad_gated = 0.0f;
#pragma unroll
            for (int tap = 0; tap < Taps; ++tap) {
                grad_gated += taps[tap] * future[Taps - 1 - tap];
            }
            Storage* slot = downstream + token * geometry.grad_projection_token_stride;
            slot[0] = from_float<Storage>(grad_gated * row.value(token));
            slot[geometry.channels] = from_float<Storage>(incoming * convolved);
            slot[2 * geometry.channels] = from_float<Storage>(grad_gated * row.gate(token));
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
            const int incoming_source = token - Taps;
            past[0] = incoming_source >= 0 ? row.gated(incoming_source) : 0.0f;
        }
    }

    float* weight_row = grad_weight_partial + (partial_row * geometry.channels + channel) * Taps;
#pragma unroll
    for (int tap = 0; tap < Taps; ++tap) {
        weight_row[tap] = weight_sums[tap];
    }
    if (grad_bias_partial != nullptr) {
        grad_bias_partial[partial_row * geometry.channels + channel] = bias_sum;
    }
}

template <typename Storage, int Taps>
__global__ void __launch_bounds__(channel_threads) shortconv_state_kernel(
    const Storage* __restrict__ projection,
    Storage* state,
    StateGeometry geometry,
    bool has_history)
{
    const int channel = blockIdx.x * channel_threads + threadIdx.x;
    const int batch = blockIdx.y;
    if (channel >= geometry.channels) {
        return;
    }
    const ProjectionRow<Storage> row{
        projection + batch * geometry.projection_batch_stride + channel,
        geometry.projection_token_stride,
        geometry.channels};
    Storage* entry = state + batch * geometry.state_batch_stride + channel * geometry.state_channel_stride;
#pragma unroll
    for (int slot = 0; slot < Taps; ++slot) {
        const int position = geometry.seq_len + slot - Taps;
        float value = 0.0f;
        if (position >= 0) {
            value = row.gated(position);
        } else if (has_history) {
            value = to_float(entry[(Taps + position) * geometry.state_token_stride]);
        }
        entry[slot * geometry.state_token_stride] = from_float<Storage>(value);
    }
}

template <typename Body>
void dispatch_storage(at::ScalarType type, Body&& body)
{
    if (type == at::kBFloat16) {
        body(c10::BFloat16{});
    } else if (type == at::kFloat) {
        body(float{});
    } else {
        TORCH_CHECK(false, "lfm2_kernels short-convolution supports bfloat16 and float32 activations, got ", type);
    }
}

template <typename Body>
void dispatch_taps(std::int64_t taps, Body&& body)
{
    switch (taps) {
    case 2:
        body(std::integral_constant<int, 2>{});
        break;
    case 3:
        body(std::integral_constant<int, 3>{});
        break;
    case 4:
        body(std::integral_constant<int, 4>{});
        break;
    default:
        TORCH_CHECK(false, "lfm2_kernels short-convolution supports 2 to 4 taps, got ", taps);
    }
}

int channel_blocks(int channels)
{
    return (channels + channel_threads - 1) / channel_threads;
}

int chunk_tokens(int batch, int seq_len, int channels)
{
    const int multiprocessors = at::cuda::getCurrentDeviceProperties()->multiProcessorCount;
    const int lanes = std::max(1, batch * channel_blocks(channels));
    const int wanted_chunks = std::max(1, (multiprocessors * 4 + lanes - 1) / lanes);
    return std::max(8, (seq_len + wanted_chunks - 1) / wanted_chunks);
}

void check_projection(const at::Tensor& projection)
{
    TORCH_CHECK(projection.is_cuda(), "projection must be a CUDA tensor");
    TORCH_CHECK(projection.dim() == 3, "projection must be [batch, tokens, 3 * channels]");
    TORCH_CHECK(projection.size(2) % 3 == 0, "projection channel axis must split into gate|carrier|value");
    TORCH_CHECK(projection.stride(2) == 1, "projection must be contiguous along its channel axis");
}

void check_parameters(
    const at::Tensor& projection, const at::Tensor& weight, const std::optional<at::Tensor>& bias)
{
    const std::int64_t channels = projection.size(2) / 3;
    TORCH_CHECK(weight.device() == projection.device(), "weight must share the projection device");
    TORCH_CHECK(weight.scalar_type() == at::kFloat, "weight must be float32");
    TORCH_CHECK(weight.dim() == 2 && weight.size(0) == channels, "weight must be [channels, taps]");
    TORCH_CHECK(weight.is_contiguous(), "weight must be contiguous");
    if (bias.has_value()) {
        TORCH_CHECK(bias->device() == projection.device(), "bias must share the projection device");
        TORCH_CHECK(bias->scalar_type() == at::kFloat, "bias must be float32");
        TORCH_CHECK(bias->dim() == 1 && bias->size(0) == channels, "bias must be [channels]");
        TORCH_CHECK(bias->is_contiguous(), "bias must be contiguous");
    }
}

}

at::Tensor shortconv_forward(
    const at::Tensor& projection,
    const at::Tensor& weight,
    const std::optional<at::Tensor>& bias,
    const std::optional<at::Tensor>& state)
{
    check_projection(projection);
    check_parameters(projection, weight, bias);
    const c10::cuda::CUDAGuard guard(projection.device());
    const int batch = static_cast<int>(projection.size(0));
    const int seq_len = static_cast<int>(projection.size(1));
    const int channels = static_cast<int>(projection.size(2) / 3);
    const std::int64_t taps = weight.size(1);
    if (state.has_value()) {
        TORCH_CHECK(state->device() == projection.device(), "state must share the projection device");
        TORCH_CHECK(state->scalar_type() == projection.scalar_type(), "state dtype must match projection");
        TORCH_CHECK(
            state->dim() == 3 && state->size(0) == batch && state->size(1) == channels,
            "state must be [batch, channels, history]");
        TORCH_CHECK(state->size(2) >= taps - 1, "state must hold at least taps - 1 history entries");
    }
    at::Tensor output = at::empty({batch, seq_len, channels}, projection.options());
    if (seq_len == 0 || batch == 0) {
        return output;
    }
    const int chunk = chunk_tokens(batch, seq_len, channels);
    const ForwardGeometry geometry{
        seq_len,
        channels,
        chunk,
        projection.stride(0),
        projection.stride(1),
        state.has_value() ? state->stride(0) : 0,
        state.has_value() ? state->stride(1) : 0,
        state.has_value() ? state->stride(2) : 0,
        state.has_value() ? static_cast<int>(state->size(2)) : 0,
        output.stride(0),
        output.stride(1)};
    const dim3 grid(channel_blocks(channels), (seq_len + chunk - 1) / chunk, batch);
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    dispatch_storage(projection.scalar_type(), [&](auto storage_tag) {
        using Storage = decltype(storage_tag);
        dispatch_taps(taps, [&](auto taps_tag) {
            constexpr int Taps = decltype(taps_tag)::value;
            shortconv_forward_kernel<Storage, Taps><<<grid, channel_threads, 0, stream>>>(
                projection.const_data_ptr<Storage>(),
                weight.const_data_ptr<float>(),
                bias.has_value() ? bias->const_data_ptr<float>() : nullptr,
                state.has_value() ? state->const_data_ptr<Storage>() : nullptr,
                output.mutable_data_ptr<Storage>(),
                geometry);
            C10_CUDA_KERNEL_LAUNCH_CHECK();
        });
    });
    return output;
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> shortconv_backward(
    const at::Tensor& projection,
    const at::Tensor& weight,
    const std::optional<at::Tensor>& bias,
    const at::Tensor& grad_output)
{
    check_projection(projection);
    check_parameters(projection, weight, bias);
    const c10::cuda::CUDAGuard guard(projection.device());
    const int batch = static_cast<int>(projection.size(0));
    const int seq_len = static_cast<int>(projection.size(1));
    const int channels = static_cast<int>(projection.size(2) / 3);
    const std::int64_t taps = weight.size(1);
    TORCH_CHECK(grad_output.device() == projection.device(), "grad_output must share the projection device");
    TORCH_CHECK(grad_output.scalar_type() == projection.scalar_type(), "grad_output dtype must match projection");
    TORCH_CHECK(
        grad_output.dim() == 3 && grad_output.size(0) == batch && grad_output.size(1) == seq_len
            && grad_output.size(2) == channels,
        "grad_output must be [batch, tokens, channels]");
    TORCH_CHECK(grad_output.stride(2) == 1, "grad_output must be contiguous along its channel axis");

    at::Tensor grad_projection = at::empty(projection.sizes(), projection.options());
    const int chunk = chunk_tokens(batch, std::max(seq_len, 1), channels);
    const int chunks = std::max(1, (seq_len + chunk - 1) / chunk);
    const at::TensorOptions partial_options = weight.options();
    at::Tensor grad_weight_partial = at::empty({static_cast<std::int64_t>(batch) * chunks, channels, taps}, partial_options);
    at::Tensor grad_bias_partial = bias.has_value()
        ? at::empty({static_cast<std::int64_t>(batch) * chunks, channels}, partial_options)
        : at::empty({0}, partial_options);
    if (seq_len == 0 || batch == 0) {
        grad_weight_partial.zero_();
        grad_bias_partial.zero_();
    } else {
        const BackwardGeometry geometry{
            seq_len,
            channels,
            chunk,
            chunks,
            projection.stride(0),
            projection.stride(1),
            grad_output.stride(0),
            grad_output.stride(1),
            grad_projection.stride(0),
            grad_projection.stride(1)};
        const dim3 grid(channel_blocks(channels), chunks, batch);
        const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
        dispatch_storage(projection.scalar_type(), [&](auto storage_tag) {
            using Storage = decltype(storage_tag);
            dispatch_taps(taps, [&](auto taps_tag) {
                constexpr int Taps = decltype(taps_tag)::value;
                shortconv_backward_kernel<Storage, Taps><<<grid, channel_threads, 0, stream>>>(
                    projection.const_data_ptr<Storage>(),
                    weight.const_data_ptr<float>(),
                    bias.has_value() ? bias->const_data_ptr<float>() : nullptr,
                    grad_output.const_data_ptr<Storage>(),
                    grad_projection.mutable_data_ptr<Storage>(),
                    grad_weight_partial.mutable_data_ptr<float>(),
                    bias.has_value() ? grad_bias_partial.mutable_data_ptr<float>() : nullptr,
                    geometry);
                C10_CUDA_KERNEL_LAUNCH_CHECK();
            });
        });
    }
    at::Tensor grad_weight = grad_weight_partial.sum(0);
    at::Tensor grad_bias = bias.has_value() ? grad_bias_partial.sum(0) : grad_bias_partial;
    return {grad_projection, grad_weight, grad_bias};
}

void shortconv_state_update(const at::Tensor& projection, at::Tensor& state, bool has_history)
{
    check_projection(projection);
    const c10::cuda::CUDAGuard guard(projection.device());
    const int batch = static_cast<int>(projection.size(0));
    const int seq_len = static_cast<int>(projection.size(1));
    const int channels = static_cast<int>(projection.size(2) / 3);
    TORCH_CHECK(state.device() == projection.device(), "state must share the projection device");
    TORCH_CHECK(state.scalar_type() == projection.scalar_type(), "state dtype must match projection");
    TORCH_CHECK(
        state.dim() == 3 && state.size(0) == batch && state.size(1) == channels,
        "state must be [batch, channels, taps]");
    if (batch == 0) {
        return;
    }
    const StateGeometry geometry{
        seq_len,
        channels,
        projection.stride(0),
        projection.stride(1),
        state.stride(0),
        state.stride(1),
        state.stride(2)};
    const dim3 grid(channel_blocks(channels), batch);
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    dispatch_storage(projection.scalar_type(), [&](auto storage_tag) {
        using Storage = decltype(storage_tag);
        dispatch_taps(state.size(2), [&](auto taps_tag) {
            constexpr int Taps = decltype(taps_tag)::value;
            shortconv_state_kernel<Storage, Taps><<<grid, channel_threads, 0, stream>>>(
                projection.const_data_ptr<Storage>(), state.mutable_data_ptr<Storage>(), geometry, has_history);
            C10_CUDA_KERNEL_LAUNCH_CHECK();
        });
    });
}

}
