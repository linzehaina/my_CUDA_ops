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
#define PAD 1

__global__ void transpose_f32(float * x, float * y, int M, int N) {
    int col = threadIdx.x + blockIdx.x * blockDim.x;
    int row = threadIdx.y + blockIdx.y * blockDim.y;
    int local_col = threadIdx.x;
    int local_row = threadIdx.y;
    __shared__ float data[32][32 + PAD];

    if (row < M && col < N) {
        data[local_row][local_col] = x[row * N + col];
    }

    __syncthreads();

    int t_row = threadIdx.y + blockIdx.x * blockDim.x;
    int t_col = threadIdx.x + blockIdx.y * blockDim.y;
    if (t_row < N && t_col < M) {
        y[t_row * M + t_col] = data[local_col][local_row];
    }
}

__global__ void transpose_f32_swizzle(float * x, float * y, int M, int N) {
    int col = threadIdx.x + blockIdx.x * blockDim.x;
    int row = threadIdx.y + blockIdx.y * blockDim.y;
    int local_col = threadIdx.x;
    int local_row = threadIdx.y;
    __shared__ float data[32][32];

    if (row < M && col < N) {
        data[local_row][local_col ^ local_row] = x[row * N + col];
    }

    __syncthreads();

    int t_row = threadIdx.y + blockIdx.x * blockDim.x;
    int t_col = threadIdx.x + blockIdx.y * blockDim.y;
    if (t_row < N && t_col < M) {
        y[t_row * M + t_col] = data[local_col][local_row ^ local_col];
    }
}

// CPU参考实现
void transpose_cpu(float* input, float* output, int M, int N) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            output[j * M + i] = input[i * N + j];
        }
    }
}

// 验证结果正确性
bool verify_result(float* gpu_result, float* cpu_result, int size, float tolerance = 1e-5) {
    for (int i = 0; i < size; i++) {
        if (fabs(gpu_result[i] - cpu_result[i]) > tolerance) {
            printf("验证失败: 位置 %d, GPU: %f, CPU: %f\n", i, gpu_result[i], cpu_result[i]);
            return false;
        }
    }
    return true;
}

// 性能测试函数
void benchmark_transpose(float* d_input, float* d_output, int M, int N, 
                        void (*kernel)(float*, float*, int, int), 
                        const char* kernel_name, int iterations = 100) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    // 预热
    for (int i = 0; i < 10; i++) {
        dim3 blockSize(32, 32);
        dim3 gridSize((N + blockSize.x - 1) / blockSize.x, (M + blockSize.y - 1) / blockSize.y);
        kernel<<<gridSize, blockSize>>>(d_input, d_output, M, N);
    }
    cudaDeviceSynchronize();
    
    // 性能测试
    cudaEventRecord(start);
    for (int i = 0; i < iterations; i++) {
        dim3 blockSize(32, 32);
        dim3 gridSize((N + blockSize.x - 1) / blockSize.x, (M + blockSize.y - 1) / blockSize.y);
        kernel<<<gridSize, blockSize>>>(d_input, d_output, M, N);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    
    float avg_time = milliseconds / iterations;
    float bandwidth = (2.0f * M * N * sizeof(float)) / (avg_time * 1e-3) / 1e9; // GB/s
    
    printf("%s 平均时间: %.4f ms, 带宽: %.2f GB/s\n", kernel_name, avg_time, bandwidth);
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

int main() {
    // 测试参数
    const int M = 3000;
    const int N = 4000;
    const int size = M * N;
    const size_t bytes = size * sizeof(float);
    
    printf("测试矩阵转置算子\n");
    printf("矩阵大小: %d x %d\n", M, N);
    printf("数据大小: %.2f MB\n", bytes / (1024.0f * 1024.0f));
    printf("========================================\n");
    
    // 分配主机内存
    float* h_input = (float*)malloc(bytes);
    float* h_output_gpu1 = (float*)malloc(bytes);
    float* h_output_gpu2 = (float*)malloc(bytes);
    float* h_output_cpu = (float*)malloc(bytes);
    
    // 初始化输入数据
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(-1.0f, 1.0f);
    
    for (int i = 0; i < size; i++) {
        h_input[i] = dis(gen);
    }
    
    // 分配设备内存
    float* d_input;
    float* d_output;
    cudaMalloc(&d_input, bytes);
    cudaMalloc(&d_output, bytes);
    
    // 复制数据到设备
    cudaMemcpy(d_input, h_input, bytes, cudaMemcpyHostToDevice);
    
    // CPU参考实现
    printf("运行CPU参考实现...\n");
    auto start = std::chrono::high_resolution_clock::now();
    transpose_cpu(h_input, h_output_cpu, M, N);
    auto end = std::chrono::high_resolution_clock::now();
    auto cpu_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    printf("CPU时间: %.4f ms\n", cpu_time / 1000.0f);
    
    // 测试第一个GPU实现
    printf("\n测试 transpose_f32...\n");
    dim3 blockSize(32, 32);
    dim3 gridSize((N + blockSize.x - 1) / blockSize.x, (M + blockSize.y - 1) / blockSize.y);
    
    transpose_f32<<<gridSize, blockSize>>>(d_input, d_output, M, N);
    cudaDeviceSynchronize();
    
    // 检查CUDA错误
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        printf("CUDA错误: %s\n", cudaGetErrorString(error));
        return -1;
    }
    
    // 复制结果回主机
    cudaMemcpy(h_output_gpu1, d_output, bytes, cudaMemcpyDeviceToHost);
    
    // 验证正确性
    bool correct1 = verify_result(h_output_gpu1, h_output_cpu, size);
    printf("transpose_f32 正确性: %s\n", correct1 ? "通过" : "失败");
    
    // 性能测试
    benchmark_transpose(d_input, d_output, M, N, transpose_f32, "transpose_f32");
    
    // 测试第二个GPU实现
    printf("\n测试 transpose_f32_swizzle...\n");
    
    transpose_f32_swizzle<<<gridSize, blockSize>>>(d_input, d_output, M, N);
    cudaDeviceSynchronize();
    
    // 检查CUDA错误
    error = cudaGetLastError();
    if (error != cudaSuccess) {
        printf("CUDA错误: %s\n", cudaGetErrorString(error));
        return -1;
    }
    
    // 复制结果回主机
    cudaMemcpy(h_output_gpu2, d_output, bytes, cudaMemcpyDeviceToHost);
    
    // 验证正确性
    bool correct2 = verify_result(h_output_gpu2, h_output_cpu, size);
    printf("transpose_f32_swizzle 正确性: %s\n", correct2 ? "通过" : "失败");
    
    // 性能测试
    benchmark_transpose(d_input, d_output, M, N, transpose_f32_swizzle, "transpose_f32_swizzle");
    
    // 清理内存
    free(h_input);
    free(h_output_gpu1);
    free(h_output_gpu2);
    free(h_output_cpu);
    cudaFree(d_input);
    cudaFree(d_output);
    
    printf("\n========================================\n");
    printf("测试完成!\n");
    
    return 0;
}
