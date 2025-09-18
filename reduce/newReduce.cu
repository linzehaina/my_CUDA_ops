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

__inline__ __device__ float warp_reduce_max(float val) {
    int mask = 0xFFFFFFFF;
    
    for (int stride = (WARP_SIZE >> 1); stride >= 1; stride >>= 1) {
        val = fmaxf(val,__shfl_xor_sync(mask, val, stride));
    }
    return val;
}

template <const int NUM_THREAD = 256>
__device__ float block_reduce_max(float val) {
    int idx = threadIdx.x;
    constexpr int warpnum = (NUM_THREAD + 32 - 1) / 32;
    int laneid = idx % WARP_SIZE;
    int warpid = idx / WARP_SIZE;
    int mask = 0xFFFFFFFF;

    __shared__ float warp_result[warpnum]; 

    float value = warp_reduce_max(val);

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

__device__ float atomicMax_fp32(float * a, float value) {
    int old_value = * reinterpret_cast<int *>(a), assumed_value;
    do {
        assumed_value = old_value;
        old_value = atomicCAS(reinterpret_cast<int*>(a), assumed_value, \
            __float_as_int(fmaxf(value, __int_as_float(assumed_value))));
        // old_value = * (reinterpret_cast<float *>(& atomicCAS(reinterpret_cast<int *>(a), \
        //     __float_as_int(assumed_value), __float_as_int(fmaxf(value, assumed_value)))));
    } while (old_value != assumed_value);
    return old_value;
}

// reduce all numbers in a to a single number
// blockDim must be 1024
__global__ void reduce_one_max_fp32(float * a, float * b, uint32_t N) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    // if (idx >= N) return;
    float value = idx < N ? a[idx]: -FLT_MAX;

    for(int i = idx + gridDim.x * blockDim.x; i < N; i += gridDim.x * blockDim.x) {
        value = fmaxf(value, a[i]);
    }

    value = block_reduce_max<1024>(value);
    if (threadIdx.x == 0) {
        atomicMax_fp32(b, value);
    }
}

// 测试代码
#define CUDA_CHECK(cmd) \
    do { \
        cudaError_t e = (cmd); \
        if (e != cudaSuccess) { \
            printf("CUDA error %s at %s:%d\n", cudaGetErrorString(e), __FILE__, __LINE__); \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

int main() {
    printf("开始测试 reduce_max_fp32 函数...\n");
    
    // 测试参数
    const uint32_t N = 1024079;
    const int blockSize = 1024;
    const int gridSize = (N + blockSize - 1) / blockSize;
    
    // 分配主机内存
    float *h_input = (float*)malloc(N * sizeof(float));
    float *h_output = (float*)malloc(sizeof(float));
    
    // 生成测试数据
    srand(42);
    float expected_max = -FLT_MAX;
    for (uint32_t i = 0; i < N; i++) {
        h_input[i] = (float)(rand() % 10000) / 100.0f - 50.0f; // -50.0 到 50.0 的随机数
        expected_max = fmaxf(expected_max, h_input[i]);
    }
    
    printf("数组大小: %u\n", N);
    printf("预期最大值: %.6f\n", expected_max);
    
    // 分配设备内存
    float *d_input, *d_output;
    CUDA_CHECK(cudaMalloc(&d_input, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_output, sizeof(float)));
    
    // 复制数据到设备
    CUDA_CHECK(cudaMemcpy(d_input, h_input, N * sizeof(float), cudaMemcpyHostToDevice));
    
    // 初始化输出为负无穷
    float init_val = -FLT_MAX;
    CUDA_CHECK(cudaMemcpy(d_output, &init_val, sizeof(float), cudaMemcpyHostToDevice));
    
    // 启动内核
    printf("启动内核: gridSize=%d, blockSize=%d\n", gridSize, blockSize);
    reduce_one_max_fp32<<<gridSize, blockSize>>>(d_input, d_output, N);
    
    // 检查内核启动错误
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // 复制结果回主机
    CUDA_CHECK(cudaMemcpy(h_output, d_output, sizeof(float), cudaMemcpyDeviceToHost));
    
    // 验证结果
    float actual_max = h_output[0];
    float error = fabsf(actual_max - expected_max);
    bool test_passed = error < 1e-6f;
    
    printf("实际最大值: %.6f\n", actual_max);
    printf("误差: %.8f\n", error);
    printf("测试结果: %s\n", test_passed ? "通过" : "失败");
    
    // 清理内存
    free(h_input);
    free(h_output);
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));
    
    return test_passed ? 0 : 1;
}
