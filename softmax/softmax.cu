#include <algorithm>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <float.h>
#include <stdio.h>
#include <stdlib.h>
#include <cstdint>

#define WARP_SIZE 32
#define FLOAT4(value) (reinterpret_cast<float4 *>(&(value))[0])
#define INT4(value) (reinterpret_cast<int4 *>(&(value))[0])
#define HALF2(value) (reinterpret_cast<half2 *>(&(value))[0])
#define BFLOAT2(value) (reinterpret_cast<__nv_bfloat162 *>(&(value))[0])
#define LOAD128BIT(value) (reinterpret_cast<float4 *>(&(value))[0])

struct __align__(8) MD{
    float m;
    float d;
};

template<typename T>
__inline__ __device__ T warp_reduce_max(T val) {
    int mask = 0xFFFFFFFF;
    
    for (int stride = (WARP_SIZE >> 1); stride >= 1; stride >>= 1) {
        val = fmaxf(val,__shfl_xor_sync(mask, val, stride));
    }
    return val;
}

template <const int NUM_THREAD = 256, typename T>
__device__ T block_reduce_max(T val) {
    int idx = threadIdx.x;
    constexpr int warpnum = (NUM_THREAD + 32 - 1) / 32;
    int laneid = idx % WARP_SIZE;
    int warpid = idx / WARP_SIZE;
    int mask = 0xFFFFFFFF;

    __shared__ T warp_result[warpnum]; 

    T value = warp_reduce_max(val);

    if (laneid == 0) {
        warp_result[warpid] = value;
    }
    __syncthreads();

    // 每个warp都计算到了block_max
    value = (laneid < warpnum) ? warp_result[laneid] : -FLT_MAX;
    value = warp_reduce_max(value);
    value = __shfl_sync(mask, value, 0);
    return value;
}

template<typename T = MD>
__inline__ __device__ MD warp_reduce_max(MD val) {
    int mask = 0xFFFFFFFF;
    MD other;
    float larger_m;

    #pragma unroll
    for (int stride = (WARP_SIZE >> 1); stride >= 1; stride >>= 1) {
        other.m = __shfl_xor_sync(mask, val.m, stride);
        other.d = __shfl_xor_sync(mask, val.d, stride);
        larger_m = fmaxf(val.m, other.m);
        val.d = val.d * __expf(val.m - larger_m) + other.d * __expf(other.m - larger_m);
        val.m = larger_m;
    }
    return val;
}

template <const int NUM_THREAD = 256, typename T = MD>
__device__ MD block_reduce_max(MD val) {
    int idx = threadIdx.x;
    constexpr int warpnum = (NUM_THREAD + 32 - 1) / 32;
    int laneid = idx & (WARP_SIZE - 1);
    int warpid = idx / WARP_SIZE;
    int mask = 0xFFFFFFFF;

    __shared__ MD warp_result[warpnum]; 

    MD value = warp_reduce_max(val);

    if (laneid == 0) {
        warp_result[warpid] = value;
    }
    __syncthreads();

    // 每个warp都计算到了block_max
    value = (laneid < warpnum) ? warp_result[laneid] : MD{-FLT_MAX, 0.0f};
    value = warp_reduce_max(value);
    value.m = __shfl_sync(mask, value.m, 0);
    value.d = __shfl_sync(mask, value.d, 0);
    return value;
}

// softmax op for fp32
template <const int NUM_THREAD = 256>
__global__ void online_safe_softmax_fp32(float * a, float * b, uint32_t N) {
    int laneid = threadIdx.x % WARP_SIZE;
    int warpid = threadIdx.x / WARP_SIZE;
    int tid = threadIdx.x;
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    constexpr int warpnum = (NUM_THREAD + 32 - 1) / 32;
    
    MD val;
    val.m = idx < N ? a[idx]: -FLT_MAX;
    val.d = idx < N ? 1.0: 0.0;

    val = block_reduce_max<1024, MD>(val);

    // __shared__ MD warp_result[warpnum];
    // val = warp_reduce_max(val);
    // if (laneid == 0) {
    //     warp_result[warpid] = val;
    // }
    // __syncthreads();

    // if (tid < WARP_SIZE) {
    //     val.m = laneid < warpnum ? warp_result[laneid].m: -FLT_MAX;
    //     val.d = laneid < warpnum ? warp_result[laneid].d: 0.0f;
    //     val = warp_reduce_max(val);
    //     if (laneid == 0) {
    //         warp_result[0] = val;
    //     }
    // }
    // __syncthreads();
    // val = warp_result[0];

    if (idx < N) {
        b[idx] = __expf(a[idx] - val.m) / val.d;
    }
    return;
}



