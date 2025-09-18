#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <random>
#include <chrono>
#include <algorithm>

// 包含softmax函数
#include "softmax.cu"

// CPU参考实现
void cpu_single_token_softmax(float* input, float* output, int N) {
    // 找到最大值
    float max_val = input[0];
    for (int i = 1; i < N; i++) {
        max_val = fmaxf(max_val, input[i]);
    }
    
    // 计算exp和sum
    float sum = 0.0f;
    for (int i = 0; i < N; i++) {
        output[i] = expf(input[i] - max_val);
        sum += output[i];
    }
    
    // 归一化
    for (int i = 0; i < N; i++) {
        output[i] /= sum;
    }
}

template<const int blockDim = 1024>
void cpu_softmax(float* input, float* output, int N) {
    for (int i = 0; i < N; i+= blockDim) {
        cpu_single_token_softmax(input + i, output + i, blockDim);
    }
}

// 检查结果是否正确
bool check_result(float* gpu_result, float* cpu_result, int N, float tolerance = 1e-5) {
    for (int i = 0; i < N; i++) {
        if (fabsf(gpu_result[i] - cpu_result[i]) > tolerance) {
            printf("错误: 位置 %d, GPU: %f, CPU: %f, 差异: %f\n", 
                   i, gpu_result[i], cpu_result[i], fabsf(gpu_result[i] - cpu_result[i]));
            return false;
        }
    }
    return true;
}

// 打印数组
void print_array(float* arr, int N, const char* name) {
    printf("%s: [", name);
    for (int i = 0; i < std::min(N, 10); i++) {
        printf("%.6f", arr[i]);
        if (i < std::min(N, 10) - 1) printf(", ");
    }
    if (N > 10) printf("...");
    printf("]\n");
}

int main() {
    printf("=== Softmax函数测试 ===\n");
    
    // 测试参数
    const int N = 1024000;
    const int block_size = 1024;
    const int grid_size = (N + block_size - 1) / block_size;
    
    printf("测试大小: %d\n", N);
    printf("Block大小: %d\n", block_size);
    printf("Grid大小: %d\n", grid_size);
    
    // 分配内存
    float *h_input = (float*)malloc(N * sizeof(float));
    float *h_output_gpu = (float*)malloc(N * sizeof(float));
    float *h_output_cpu = (float*)malloc(N * sizeof(float));
    
    float *d_input, *d_output;
    cudaMalloc(&d_input, N * sizeof(float));
    cudaMalloc(&d_output, N * sizeof(float));
    
    // 生成测试数据
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(-10.0f, 10.0f);
    
    for (int i = 0; i < N; i++) {
        h_input[i] = dis(gen);
    }
    
    printf("\n输入数据 (前10个): ");
    print_array(h_input, N, "");
    
    // CPU计算
    auto start_cpu = std::chrono::high_resolution_clock::now();
    cpu_softmax<1024>(h_input, h_output_cpu, N);
    auto end_cpu = std::chrono::high_resolution_clock::now();
    auto cpu_time = std::chrono::duration_cast<std::chrono::microseconds>(end_cpu - start_cpu);
    
    printf("CPU结果 (前10个): ");
    print_array(h_output_cpu, N, "");
    
    // GPU计算
    cudaMemcpy(d_input, h_input, N * sizeof(float), cudaMemcpyHostToDevice);
    
    auto start_gpu = std::chrono::high_resolution_clock::now();
    online_safe_softmax_fp32<block_size><<<grid_size, block_size>>>(d_input, d_output, N);
    cudaDeviceSynchronize();
    auto end_gpu = std::chrono::high_resolution_clock::now();
    auto gpu_time = std::chrono::duration_cast<std::chrono::microseconds>(end_gpu - start_gpu);
    
    cudaMemcpy(h_output_gpu, d_output, N * sizeof(float), cudaMemcpyDeviceToHost);
    
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
        h_input[i] = 5.0f;
    }
    
    cudaMemcpy(d_input, h_input, N * sizeof(float), cudaMemcpyHostToDevice);
    online_safe_softmax_fp32<block_size><<<grid_size, block_size>>>(d_input, d_output, N);
    cudaDeviceSynchronize();
    cudaMemcpy(h_output_gpu, d_output, N * sizeof(float), cudaMemcpyDeviceToHost);
    
    cpu_softmax<1024>(h_input, h_output_cpu, N);
    bool test1_pass = check_result(h_output_gpu, h_output_cpu, N);
    printf("测试1结果: %s\n", test1_pass ? "通过" : "失败");
    
    // 测试2: 包含极大值
    printf("测试2: 包含极大值\n");
    for (int i = 0; i < N; i++) {
        h_input[i] = (i == N/2) ? 100.0f : 1.0f;
    }
    
    cudaMemcpy(d_input, h_input, N * sizeof(float), cudaMemcpyHostToDevice);
    online_safe_softmax_fp32<block_size><<<grid_size, block_size>>>(d_input, d_output, N);
    cudaDeviceSynchronize();
    cudaMemcpy(h_output_gpu, d_output, N * sizeof(float), cudaMemcpyDeviceToHost);
    
    cpu_softmax<1024>(h_input, h_output_cpu, N);
    bool test2_pass = check_result(h_output_gpu, h_output_cpu, N);
    printf("测试2结果: %s\n", test2_pass ? "通过" : "失败");
    
    // 测试3: 包含负值
    printf("测试3: 包含负值\n");
    for (int i = 0; i < N; i++) {
        h_input[i] = -10.0f + i * 0.1f;
    }
    
    cudaMemcpy(d_input, h_input, N * sizeof(float), cudaMemcpyHostToDevice);
    online_safe_softmax_fp32<block_size><<<grid_size, block_size>>>(d_input, d_output, N);
    cudaDeviceSynchronize();
    cudaMemcpy(h_output_gpu, d_output, N * sizeof(float), cudaMemcpyDeviceToHost);
    
    cpu_softmax<1024>(h_input, h_output_cpu, N);
    bool test3_pass = check_result(h_output_gpu, h_output_cpu, N);
    printf("测试3结果: %s\n", test3_pass ? "通过" : "失败");
    
    // 清理内存
    free(h_input);
    free(h_output_gpu);
    free(h_output_cpu);
    cudaFree(d_input);
    cudaFree(d_output);
    
    printf("\n=== 测试完成 ===\n");
    return 0;
}
