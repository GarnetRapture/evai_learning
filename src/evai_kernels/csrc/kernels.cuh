#pragma once

#include <cuda_runtime_api.h>
#include <driver_types.h>

#include <array>
#include <cstdint>

namespace evai_kernels {

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

struct LaunchContext {
    cudaStream_t stream;
    int multiprocessors;
};

struct AdamWBuffers {
    void* weights;
    void* grads;
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

struct RmsNormGradWeightReduce {
    StorageType storage;
    const float* partial;
    void* grad_weight;
    std::int64_t partial_rows;
    std::int64_t columns;
};

cudaError_t launch_rms_norm_grad_weight_reduce(
    const RmsNormGradWeightReduce& request, const LaunchContext& context);

struct RopeApply {
    StorageType storage;
    const void* input;
    Shape3 outer_sizes;
    Shape3 outer_strides;
    const void* cos;
    const void* sin;
    void* output;
    std::int64_t head_dim;
    bool negate_sin;
};

constexpr bool rope_supports_head_dim(std::int64_t head_dim)
{
    return head_dim == 64 || head_dim == 128 || head_dim == 256;
}

cudaError_t launch_rope_apply(const RopeApply& request, const LaunchContext& context);

struct SwiGluForward {
    StorageType storage;
    const void* gate;
    const void* up;
    void* output;
    std::int64_t numel;
};

struct SwiGluBackward {
    StorageType storage;
    const void* gate;
    const void* up;
    const void* grad_output;
    void* grad_gate;
    void* grad_up;
    std::int64_t numel;
};

constexpr bool rms_norm_supports_columns(std::int64_t columns)
{
    return columns == 32 || columns == 64 || columns == 128 || columns == 256 || columns == 512
        || columns == 1024;
}

std::int64_t rms_norm_backward_partial_rows(int multiprocessors);

cudaError_t launch_rms_norm_forward(const RmsNormForward& request, const LaunchContext& context);

cudaError_t launch_rms_norm_backward(const RmsNormBackward& request, const LaunchContext& context);

cudaError_t launch_swiglu_forward(const SwiGluForward& request, const LaunchContext& context);

cudaError_t launch_swiglu_backward(const SwiGluBackward& request, const LaunchContext& context);

inline constexpr std::int64_t adamw_vector_width = 8;

std::int64_t adamw_reduce_blocks(int multiprocessors);

cudaError_t launch_sr_adamw(
    const AdamWBuffers& buffers, const AdamWSettings& settings, const LaunchContext& context);

const char* describe_cuda_error(cudaError_t status);

}
