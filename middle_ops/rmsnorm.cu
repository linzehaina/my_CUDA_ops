#include <algorithm>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <float.h>
#include <stdio.h>
#include <stdlib.h>
#include <cstdint>
#include <math.h>
#include <random>
#include <chrono>
#include <vector>

#define WARP_SIZE 32
#define FLOAT4(value) (reinterpret_cast<float4 *>(&(value))[0])
#define INT4(value) (reinterpret_cast<int4 *>(&(value))[0])
#define HALF2(value) (reinterpret_cast<half2 *>(&(value))[0])
#define BFLOAT2(value) (reinterpret_cast<__nv_bfloat162 *>(&(value))[0])
#define LOAD128BIT(value) (reinterpret_cast<float4 *>(&(value))[0])

template<typename T>
 __device__ T warp_reduce_max(T val) {
    int mask = 0xFFFFFFFF;
    
    for (int stride = (WARP_SIZE >> 1); stride >= 1; stride >>= 1) {
        val = fmaxf(val,__shfl_xor_sync(mask, val, stride));
    }
    return val;
}

template<typename T>
 __device__ T warp_reduce_add(T val) {
    int mask = 0xFFFFFFFF;
    
    for (int stride = (WARP_SIZE >> 1); stride >= 1; stride >>= 1) {
        val = val + __shfl_xor_sync(mask, val, stride);
    }
    return val;
}

template <const int NUM_THREAD = 256, typename T>
__device__ T block_reduce_add(T val) {
    int idx = threadIdx.x;
    constexpr int warpnum = (NUM_THREAD + 32 - 1) / 32;
    int laneid = idx % WARP_SIZE;
    int warpid = idx / WARP_SIZE;

    __shared__ T warp_result[warpnum]; 

    T value = warp_reduce_add(val);
    if (laneid == 0)
        warp_result[warpid] = value;
    __syncthreads();
    value = (laneid < warpnum) ? warp_result[laneid] : 0.0f;
    value = warp_reduce_add(value);
    return value;
}

// blockDim = 1024
template<const int NUM_THREAD = 1024>
__global__ void rmsnorm_fp16x2(half * x, half * y, half * alpha, int N) {
    int global_tid = 2 * (threadIdx.x + blockIdx.x * blockDim.x);
    int local_tid = 2 * threadIdx.x;

    if (global_tid < N) {
        half2 reg = HALF2(x[global_tid]);
        float var = __half2float(reg.x) * __half2float(reg.x) + __half2float(reg.y) * __half2float(reg.y);
        var = block_reduce_add<NUM_THREAD>(var) / (NUM_THREAD * 2);

        reg.x = __float2half(__half2float(reg.x) * rsqrtf(var) * __half2float(alpha[local_tid]));
        reg.y = __float2half(__half2float(reg.y) * rsqrtf(var) * __half2float(alpha[local_tid + 1]));
        HALF2(y[global_tid]) = reg;
    }
}

// CPU参考实现
void cpu_rmsnorm_fp16_single(half* input, half* output, half* alpha, int N) {
    // 计算平方和
    float sum_squares = 0.0f;
    for (int i = 0; i < N; i++) {
        float val = __half2float(input[i]);
        sum_squares += val * val;
    }
    
    // 计算RMS
    float rms = sqrtf(sum_squares / N);
    
    // 应用RMSNorm
    for (int i = 0; i < N; i++) {
        float val = __half2float(input[i]);
        float alpha_val = __half2float(alpha[i]);
        output[i] = __float2half(val / rms * alpha_val);
    }
}

template<const int hidden_dim = 2048>
void cpu_rmsnorm_fp16(half* input, half* output, half* alpha, int N) {
    for (int i = 0; i < N; i+= 2048) {
        cpu_rmsnorm_fp16_single(input + i, output + i, alpha, hidden_dim);
    }
}

// 检查结果是否正确
bool check_result(half* gpu_result, half* cpu_result, int N, float tolerance = 1e-2) {
    for (int i = 0; i < N; i++) {
        float gpu_val = __half2float(gpu_result[i]);
        float cpu_val = __half2float(cpu_result[i]);
        if (fabsf(gpu_val - cpu_val) > tolerance) {
            printf("错误: 位置 %d, GPU: %f, CPU: %f, 差异: %f\n", 
                   i, gpu_val, cpu_val, fabsf(gpu_val - cpu_val));
            return false;
        }
    }
    return true;
}

// 打印数组
void print_array(half* arr, int N, const char* name) {
    printf("%s: [", name);
    for (int i = 0; i < std::min(N, 10); i++) {
        printf("%.6f", __half2float(arr[i]));
        if (i < std::min(N, 10) - 1) printf(", ");
    }
    if (N > 10) printf("...");
    printf("]\n");
}

int main() {
    printf("=== RMSNorm函数测试 ===\n");
    
    const int hidden_dim = 2048;
    // 测试参数
    const int N = 1024000;
    const int block_size = hidden_dim / 2;
    const int grid_size = (N + block_size * 2 - 1) / (block_size * 2);
    
    printf("测试大小: %d\n", N);
    printf("Block大小: %d\n", block_size);
    printf("Grid大小: %d\n", grid_size);
    
    // 分配内存
    half *h_input = (half*)malloc(N * sizeof(half));
    half *h_alpha = (half*)malloc(hidden_dim * sizeof(half));
    half *h_output_gpu = (half*)malloc(N * sizeof(half));
    half *h_output_cpu = (half*)malloc(N * sizeof(half));
    
    half *d_input, *d_alpha, *d_output;
    cudaMalloc(&d_input, N * sizeof(half));
    cudaMalloc(&d_alpha, hidden_dim * sizeof(half));
    cudaMalloc(&d_output, N * sizeof(half));
    
    // 生成测试数据
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(-5.0f, 5.0f);
    
    for (int i = 0; i < N; i++) {
        h_input[i] = __float2half(dis(gen));
        
    }

    for (int i = 0; i < hidden_dim; i++) {
        h_alpha[i] = __float2half(1.0f + dis(gen) * 0.1f); // alpha在0.9-1.1之间
    }
    
    printf("\n输入数据 (前10个): ");
    print_array(h_input, N, "");
    printf("Alpha数据 (前10个): ");
    print_array(h_alpha, hidden_dim, "");
    
    // CPU计算
    auto start_cpu = std::chrono::high_resolution_clock::now();
    cpu_rmsnorm_fp16<2048>(h_input, h_output_cpu, h_alpha, N);
    auto end_cpu = std::chrono::high_resolution_clock::now();
    auto cpu_time = std::chrono::duration_cast<std::chrono::microseconds>(end_cpu - start_cpu);
    
    printf("CPU结果 (前10个): ");
    print_array(h_output_cpu, N, "");
    
    // GPU计算
    cudaMemcpy(d_input, h_input, N * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_alpha, h_alpha, hidden_dim * sizeof(half), cudaMemcpyHostToDevice);
    
    auto start_gpu = std::chrono::high_resolution_clock::now();
    rmsnorm_fp16x2<block_size><<<grid_size, block_size>>>(d_input, d_output, d_alpha, N);
    cudaDeviceSynchronize();
    auto end_gpu = std::chrono::high_resolution_clock::now();
    auto gpu_time = std::chrono::duration_cast<std::chrono::microseconds>(end_gpu - start_gpu);
    
    cudaMemcpy(h_output_gpu, d_output, N * sizeof(half), cudaMemcpyDeviceToHost);
    
    printf("GPU结果 (前10个): ");
    print_array(h_output_gpu, N, "");
    
    // 检查结果
    bool is_correct = check_result(h_output_gpu, h_output_cpu, N);
    
    printf("\n=== 性能测试 ===\n");
    printf("CPU时间: %ld 微秒\n", cpu_time.count());
    printf("GPU时间: %ld 微秒\n", gpu_time.count());
    printf("加速比: %.2fx\n", (float)cpu_time.count() / gpu_time.count());
    
    printf("\n=== 正确性测试 ===\n");
    if (is_correct) {
        printf("✓ 测试通过! GPU和CPU结果一致\n");
    } else {
        printf("✗ 测试失败! GPU和CPU结果不一致\n");
    }
    
    // 测试边界情况
    printf("\n=== 边界情况测试 ===\n");
    
    // 测试1: 所有元素相同
    printf("测试1: 所有元素相同\n");
    for (int i = 0; i < N; i++) {
        h_input[i] = __float2half(3.0f);
    }
    
    cpu_rmsnorm_fp16(h_input, h_output_cpu, h_alpha, N);
    cudaMemcpy(d_input, h_input, N * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_alpha, h_alpha, hidden_dim * sizeof(half), cudaMemcpyHostToDevice);
    rmsnorm_fp16x2<block_size><<<grid_size, block_size>>>(d_input, d_output, d_alpha, N);
    cudaDeviceSynchronize();
    cudaMemcpy(h_output_gpu, d_output, N * sizeof(half), cudaMemcpyDeviceToHost);
    
    bool test1_pass = check_result(h_output_gpu, h_output_cpu, N);
    printf("测试1结果: %s\n", test1_pass ? "通过" : "失败");
    
    // 测试2: 包含零值
    printf("测试2: 包含零值\n");
    for (int i = 0; i < N; i++) {
        h_input[i] = __float2half(i % 2 == 0 ? 0.0f : 1.0f);
    }
    
    cpu_rmsnorm_fp16(h_input, h_output_cpu, h_alpha, N);
    cudaMemcpy(d_input, h_input, N * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_alpha, h_alpha, hidden_dim * sizeof(half), cudaMemcpyHostToDevice);
    rmsnorm_fp16x2<block_size><<<grid_size, block_size>>>(d_input, d_output, d_alpha, N);
    cudaDeviceSynchronize();
    cudaMemcpy(h_output_gpu, d_output, N * sizeof(half), cudaMemcpyDeviceToHost);
    
    bool test2_pass = check_result(h_output_gpu, h_output_cpu, N);
    printf("测试2结果: %s\n", test2_pass ? "通过" : "失败");
    
    // 清理内存
    free(h_input);
    free(h_alpha);
    free(h_output_gpu);
    free(h_output_cpu);
    cudaFree(d_input);
    cudaFree(d_alpha);
    cudaFree(d_output);
    
    printf("\n=== 测试完成 ===\n");
    return 0;
}