#pragma once

#include <cuda/std/array>
#include <cuda/std/cstdint>
#include <cuda/std/mdspan>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include "kernels.cuh"

namespace lfm2_kernels {

using Extents3 = cuda::std::dextents<std::int64_t, 3>;
using Index2 = cuda::std::array<std::int64_t, 2>;
using Index3 = cuda::std::array<std::int64_t, 3>;

template <typename Element>
using Strided3 = cuda::std::mdspan<Element, Extents3, cuda::std::layout_stride>;

template <typename Element>
Strided3<Element> strided_view(Element* data, const Shape3& sizes, const Shape3& strides)
{
    const Extents3 extents(sizes[0], sizes[1], sizes[2]);
    const Index3 steps{strides[0], strides[1], strides[2]};
    return Strided3<Element>(data, cuda::std::layout_stride::mapping<Extents3>(extents, steps));
}

__device__ __forceinline__ float to_float(float value)
{
    return value;
}

__device__ __forceinline__ float to_float(__nv_bfloat16 value)
{
    return __bfloat162float(value);
}

template <typename Storage>
__device__ __forceinline__ Storage from_float(float value);

template <>
__device__ __forceinline__ float from_float<float>(float value)
{
    return value;
}

template <>
__device__ __forceinline__ __nv_bfloat16 from_float<__nv_bfloat16>(float value)
{
    return __float2bfloat16_rn(value);
}

__device__ __forceinline__ __nv_bfloat16 round_stochastic(float value, cuda::std::uint32_t noise)
{
    cuda::std::uint32_t bits = __float_as_uint(value);
    if ((bits & 0x7f800000U) != 0x7f800000U) {
        bits += noise & 0xffffU;
    }
    return __ushort_as_bfloat16(static_cast<unsigned short>(bits >> 16U));
}

__device__ __forceinline__ cuda::std::uint32_t mix_bits(cuda::std::uint64_t key)
{
    key ^= key >> 33U;
    key *= 0xff51afd7ed558ccdULL;
    key ^= key >> 33U;
    key *= 0xc4ceb9fe1a85ec53ULL;
    key ^= key >> 33U;
    return static_cast<cuda::std::uint32_t>(key);
}

}
