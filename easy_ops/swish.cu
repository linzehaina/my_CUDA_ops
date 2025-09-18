#include <algorithm>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <float.h>
#include <stdio.h>
#include <stdlib.h>
#include <torch/extension.h>
#include <torch/types.h>
#include <vector>

#define WARP_SIZE 32
#define INT4(value) (reinterpret_cast<int4 *>(&(value))[0])
#define FLOAT4(value) (reinterpret_cast<float4 *>(&(value))[0])
#define HALF2(value) (reinterpret_cast<half2 *>(&(value))[0])
#define BFLOAT2(value) (reinterpret_cast<__nv_bfloat162 *>(&(value))[0])
#define LDST128BITS(value) (reinterpret_cast<float4 *>(&(value))[0])

// FP32
// Swish x: N, y: N y=x*sigmoid(x)
__device__ __forceinline__ float swish_fp32(float x) {
    return x / (1.0f + expf(-x));
}
  
__global__ void swish_fp32(float *x, float *y, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N)
        y[idx] = swish_fp32(x[idx]);
}

__global__ void swish_fp32x4(float *x, float *y, int N) {
    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
    if (idx < N) {
        float4 reg_x = FLOAT4(x[idx]);
        float4 reg_y;
        reg_y.x = swish_fp32(reg_x.x);
        reg_y.y = swish_fp32(reg_x.y);
        reg_y.z = swish_fp32(reg_x.z);
        reg_y.w = swish_fp32(reg_x.w);
        FLOAT4(y[idx]) = reg_y;
    }
}

//  FP16
__device__ __forceinline__ half swish_half(half x) {
    return __hmul(x, __hdiv(__float2half(1.0f), 
                __hadd(__float2half(1.0f), hexp(__hneg(x)))));
}

__global__ void swish_f16_kernel(half *x, half *y, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N)
        y[idx] = swish_half(x[idx]);
}

__global__ void swish_f16x2_kernel(half *x, half *y, int N) {
    int idx = 2 * (blockIdx.x * blockDim.x + threadIdx.x);
    if (idx < N) {
        half2 reg_x = HALF2(x[idx]);
        half2 reg_y;
        reg_y.x = swish_half(reg_x.x);
        reg_y.y = swish_half(reg_x.y);
        HALF2(y[idx]) = reg_y;
    }
}