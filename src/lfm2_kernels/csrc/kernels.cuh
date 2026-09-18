#pragma once

#include <cuda_runtime_api.h>
#include <driver_types.h>

#include <array>
#include <cstdint>

namespace lfm2_kernels {

enum class StorageType : std::uint8_t { float32, bfloat16 };

using Shape3 = std::array<std::int64_t, 3>;

struct ConstTensor3 {
    const void* data;
    Shape3 sizes;
    Shape3 strides;
};

struct MutableTensor3 {
    void* data;
    Shape3 sizes;
    Shape3 strides;
};

struct ShortConvParameters {
    const float* weight;
    const float* bias;
    std::int64_t taps;
};

struct ShortConvForward {
    StorageType storage;
    ConstTensor3 projection;
    ShortConvParameters parameters;
    ConstTensor3 state;
    MutableTensor3 output;
    std::int64_t chunk;
};

struct ShortConvBackward {
    StorageType storage;
    ConstTensor3 projection;
    ShortConvParameters parameters;
    ConstTensor3 grad_output;
    MutableTensor3 grad_projection;
    float* grad_weight_partial;
    float* grad_bias_partial;
    std::int64_t chunk;
    std::int64_t chunks;
};

struct ShortConvStateUpdate {
    StorageType storage;
    ConstTensor3 projection;
    MutableTensor3 state;
    bool has_history;
};

struct LaunchContext {
    cudaStream_t stream;
    int multiprocessors;
};

struct AdamWBuffers {
    void* weights;
    const void* grads;
    void* exp_avg;
    void* exp_avg_sq;
    std::int64_t numel;
    float* clip_state;
    float* partial;
    std::int64_t partial_count;
};

struct AdamWSettings {
    float max_norm;
    float lr;
    float beta1;
    float beta2;
    float one_minus_beta1;
    float one_minus_beta2;
    float eps;
    float weight_decay;
    float bias_correction1;
    float bias_correction2_sqrt;
    std::uint64_t seed;
    std::uint64_t step;
};

struct RmsNormForward {
    StorageType storage;
    const void* input;
    const void* weight;
    void* output;
    float* inverse_rms;
    std::int64_t rows;
    std::int64_t columns;
    float epsilon;
};

struct RmsNormBackward {
    StorageType storage;
    const void* grad_output;
    const void* input;
    const void* weight;
    const float* inverse_rms;
    void* grad_input;
    float* grad_weight_partial;
    std::int64_t rows;
    std::int64_t columns;
    std::int64_t partial_rows;
};

constexpr bool shortconv_supports_taps(std::int64_t taps)
{
    return taps >= 2 && taps <= 4;
}

constexpr bool rms_norm_supports_columns(std::int64_t columns)
{
    return columns == 32 || columns == 64 || columns == 128 || columns == 256 || columns == 512
        || columns == 1024;
}

std::int64_t rms_norm_backward_partial_rows(int multiprocessors);

cudaError_t launch_rms_norm_forward(const RmsNormForward& request, const LaunchContext& context);

cudaError_t launch_rms_norm_backward(const RmsNormBackward& request, const LaunchContext& context);

std::int64_t shortconv_chunk_tokens(
    std::int64_t batch, std::int64_t seq_len, std::int64_t channels, int multiprocessors);

std::int64_t adamw_reduce_blocks(int multiprocessors);

cudaError_t launch_shortconv_forward(const ShortConvForward& request, const LaunchContext& context);

cudaError_t launch_shortconv_backward(const ShortConvBackward& request, const LaunchContext& context);

cudaError_t launch_shortconv_state_update(const ShortConvStateUpdate& request, const LaunchContext& context);

cudaError_t launch_sr_adamw(
    const AdamWBuffers& buffers, const AdamWSettings& settings, const LaunchContext& context);

const char* describe_cuda_error(cudaError_t status);

}
