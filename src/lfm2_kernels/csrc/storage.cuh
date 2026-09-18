#pragma once

#include <c10/util/BFloat16.h>

#include <cstdint>

namespace lfm2_kernels {

template <typename Storage>
__device__ __forceinline__ float to_float(Storage value);

template <>
__device__ __forceinline__ float to_float<float>(float value)
{
    return value;
}

template <>
__device__ __forceinline__ float to_float<c10::BFloat16>(c10::BFloat16 value)
{
    return static_cast<float>(value);
}

template <typename Storage>
__device__ __forceinline__ Storage from_float(float value);

template <>
__device__ __forceinline__ float from_float<float>(float value)
{
    return value;
}

template <>
__device__ __forceinline__ c10::BFloat16 from_float<c10::BFloat16>(float value)
{
    return c10::BFloat16(value);
}

__device__ __forceinline__ float bf16_bits_to_float(std::uint16_t bits)
{
    return __uint_as_float(static_cast<std::uint32_t>(bits) << 16);
}

__device__ __forceinline__ std::uint16_t float_to_bf16_stochastic(float value, std::uint32_t noise)
{
    std::uint32_t bits = __float_as_uint(value);
    if ((bits & 0x7f800000u) != 0x7f800000u) {
        bits += noise & 0xffffu;
    }
    return static_cast<std::uint16_t>(bits >> 16);
}

__device__ __forceinline__ std::uint32_t mix_bits(std::uint64_t key)
{
    key ^= key >> 33;
    key *= 0xff51afd7ed558ccdULL;
    key ^= key >> 33;
    key *= 0xc4ceb9fe1a85ec53ULL;
    key ^= key >> 33;
    return static_cast<std::uint32_t>(key);
}

}
