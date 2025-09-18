#include <algorithm>
#include <cuda_runtime.h>
#include <float.h>
#include <stdio.h>
#include <stdlib.h>
#include <tuple>
#include <vector>
#include <chrono>
#include <random>

#define WARP_SIZE 32
#define INT4(value) (reinterpret_cast<int4 *>(&(value))[0])
#define FLOAT4(value) (reinterpret_cast<float4 *>(&(value))[0])

__global__ void histogram_i32(int * x, int * y, int N) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if(tid < N) {
        atomicAdd(&(y[x[tid]]), 1);
    }
}

__global__ void histogram_i32x4(int * x, int * y, int N) {
    int tid = 4 * (threadIdx.x + blockIdx.x * blockDim.x);
    if(tid < N) {
        int4 reg = INT4(x[tid]);
        atomicAdd(&(y[reg.x]), 1);
        atomicAdd(&(y[reg.y]), 1);
        atomicAdd(&(y[reg.z]), 1);
        atomicAdd(&(y[reg.w]), 1);
    }
}

// CPU参考实现
void cpu_histogram(int* input, int* output, int N, int num_bins) {
    // 初始化输出数组
    for (int i = 0; i < num_bins; i++) {
        output[i] = 0;
    }
    
    // 计算直方图
    for (int i = 0; i < N; i++) {
        if (input[i] >= 0 && input[i] < num_bins) {
            output[input[i]]++;
        }
    }
}

// 检查结果是否正确
bool check_result(int* gpu_result, int* cpu_result, int num_bins) {
    for (int i = 0; i < num_bins; i++) {
        if (gpu_result[i] != cpu_result[i]) {
            printf("错误: 位置 %d, GPU: %d, CPU: %d\n", 
                   i, gpu_result[i], cpu_result[i]);
            return false;
        }
    }
    return true;
}

// 打印数组
void print_array(int* arr, int N, const char* name) {
    printf("%s: [", name);
    for (int i = 0; i < std::min(N, 20); i++) {
        printf("%d", arr[i]);
        if (i < std::min(N, 20) - 1) printf(", ");
    }
    if (N > 20) printf("...");
    printf("]\n");
}

int main() {
    printf("=== Histogram函数测试 ===\n");
    
    // 测试参数
    const int N = 102400000;
    const int num_bins = 256;
    const int block_size = 256;
    const int grid_size = (N + block_size - 1) / block_size;
    const int grid_size_x4 = (N + block_size * 4 - 1) / (block_size * 4);
    
    printf("测试大小: %d\n", N);
    printf("直方图bins: %d\n", num_bins);
    printf("Block大小: %d\n", block_size);
    printf("Grid大小: %d\n", grid_size);
    printf("Grid大小(x4): %d\n", grid_size_x4);
    
    // 分配内存
    int *h_input = (int*)malloc(N * sizeof(int));
    int *h_output_gpu = (int*)malloc(num_bins * sizeof(int));
    int *h_output_cpu = (int*)malloc(num_bins * sizeof(int));
    
    int *d_input, *d_output;
    cudaMalloc(&d_input, N * sizeof(int));
    cudaMalloc(&d_output, num_bins * sizeof(int));
    
    // 生成测试数据 (0到num_bins-1之间的随机整数)
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> dis(0, num_bins - 1);
    
    for (int i = 0; i < N; i++) {
        h_input[i] = dis(gen);
    }
    
    printf("\n输入数据 (前20个): ");
    print_array(h_input, N, "");
    
    // CPU计算
    auto start_cpu = std::chrono::high_resolution_clock::now();
    cpu_histogram(h_input, h_output_cpu, N, num_bins);
    auto end_cpu = std::chrono::high_resolution_clock::now();
    auto cpu_time = std::chrono::duration_cast<std::chrono::microseconds>(end_cpu - start_cpu);
    
    printf("CPU结果 (前20个): ");
    print_array(h_output_cpu, num_bins, "");
    
    // GPU计算 - 单元素版本
    cudaMemcpy(d_input, h_input, N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemset(d_output, 0, num_bins * sizeof(int));
    
    auto start_gpu = std::chrono::high_resolution_clock::now();
    histogram_i32<<<grid_size, block_size>>>(d_input, d_output, N);
    cudaDeviceSynchronize();
    auto end_gpu = std::chrono::high_resolution_clock::now();
    auto gpu_time = std::chrono::duration_cast<std::chrono::microseconds>(end_gpu - start_gpu);
    
    cudaMemcpy(h_output_gpu, d_output, num_bins * sizeof(int), cudaMemcpyDeviceToHost);
    
    printf("GPU结果 (前20个): ");
    print_array(h_output_gpu, num_bins, "");
    
    // 检查结果
    bool is_correct = check_result(h_output_gpu, h_output_cpu, num_bins);
    
    printf("\n=== 性能测试 (单元素版本) ===\n");
    printf("CPU时间: %ld 微秒\n", cpu_time.count());
    printf("GPU时间: %ld 微秒\n", gpu_time.count());
    printf("加速比: %.2fx\n", (float)cpu_time.count() / gpu_time.count());
    
    printf("\n=== 正确性测试 (单元素版本) ===\n");
    if (is_correct) {
        printf("✓ 单元素版本测试通过! GPU和CPU结果一致\n");
    } else {
        printf("✗ 单元素版本测试失败! GPU和CPU结果不一致\n");
    }
    
    // GPU计算 - 4元素版本
    cudaMemset(d_output, 0, num_bins * sizeof(int));
    
    auto start_gpu_x4 = std::chrono::high_resolution_clock::now();
    histogram_i32x4<<<grid_size_x4, block_size>>>(d_input, d_output, N);
    cudaDeviceSynchronize();
    auto end_gpu_x4 = std::chrono::high_resolution_clock::now();
    auto gpu_time_x4 = std::chrono::duration_cast<std::chrono::microseconds>(end_gpu_x4 - start_gpu_x4);
    
    cudaMemcpy(h_output_gpu, d_output, num_bins * sizeof(int), cudaMemcpyDeviceToHost);
    
    printf("\nGPU结果 (x4版本, 前20个): ");
    print_array(h_output_gpu, num_bins, "");
    
    // 检查x4版本结果
    bool is_correct_x4 = check_result(h_output_gpu, h_output_cpu, num_bins);
    
    printf("\n=== 性能测试 (4元素版本) ===\n");
    printf("GPU时间 (x4): %ld 微秒\n", gpu_time_x4.count());
    printf("相对于单元素版本加速比: %.2fx\n", (float)gpu_time.count() / gpu_time_x4.count());
    
    printf("\n=== 正确性测试 (4元素版本) ===\n");
    if (is_correct_x4) {
        printf("✓ 4元素版本测试通过! GPU和CPU结果一致\n");
    } else {
        printf("✗ 4元素版本测试失败! GPU和CPU结果不一致\n");
    }
    
    // 测试边界情况
    printf("\n=== 边界情况测试 ===\n");
    
    // 测试1: 所有元素相同
    printf("测试1: 所有元素相同\n");
    for (int i = 0; i < N; i++) {
        h_input[i] = 5;  // 所有元素都是5
    }
    
    cpu_histogram(h_input, h_output_cpu, N, num_bins);
    cudaMemcpy(d_input, h_input, N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemset(d_output, 0, num_bins * sizeof(int));
    histogram_i32<<<grid_size, block_size>>>(d_input, d_output, N);
    cudaDeviceSynchronize();
    cudaMemcpy(h_output_gpu, d_output, num_bins * sizeof(int), cudaMemcpyDeviceToHost);
    
    bool test1_pass = check_result(h_output_gpu, h_output_cpu, num_bins);
    printf("测试1结果: %s\n", test1_pass ? "通过" : "失败");
    
    // 测试2: 均匀分布
    printf("测试2: 均匀分布\n");
    for (int i = 0; i < N; i++) {
        h_input[i] = i % num_bins;  // 均匀分布
    }
    
    cpu_histogram(h_input, h_output_cpu, N, num_bins);
    cudaMemcpy(d_input, h_input, N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemset(d_output, 0, num_bins * sizeof(int));
    histogram_i32<<<grid_size, block_size>>>(d_input, d_output, N);
    cudaDeviceSynchronize();
    cudaMemcpy(h_output_gpu, d_output, num_bins * sizeof(int), cudaMemcpyDeviceToHost);
    
    bool test2_pass = check_result(h_output_gpu, h_output_cpu, num_bins);
    printf("测试2结果: %s\n", test2_pass ? "通过" : "失败");
    
    // 清理内存
    free(h_input);
    free(h_output_gpu);
    free(h_output_cpu);
    cudaFree(d_input);
    cudaFree(d_output);
    
    printf("\n=== 测试完成 ===\n");
    return 0;
}