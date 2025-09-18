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
#define theta 10000

// gpu avg latency over 50 iters = 0.011 ms
// cpu avg latency over 50 iters = 7.472 ms

// add rope for a whole seq in prefill without prefix cache
__global__ void rope(half * x, half * y, int hidden_dim, int N) {
    int global_tid = threadIdx.x + blockIdx.x * blockDim.x;
    int local_tid = threadIdx.x;

    if (global_tid * 2 < N) {
        half2 reg = HALF2(x[global_tid * 2]);
        int token_idx = 2 * global_tid % hidden_dim;
        int token_pos = 2 * global_tid / hidden_dim;
        float angle = (token_pos * 1.0f) / powf(theta, token_idx * 1.0f / (hidden_dim * 1.0f));
        float cos_v = cosf(angle);
        float sin_v = sinf(angle);
        float y1 = __half2float(reg.x) * cos_v - __half2float(reg.y) * sin_v;
        float y2 = __half2float(reg.x) * sin_v + __half2float(reg.y) * cos_v;
        reg.x = __float2half(y1);
        reg.y = __float2half(y2);
        HALF2(y[global_tid * 2]) = reg;
    }
}

static void rope_cpu_ref(const half *x, float *y_ref, int hidden_dim, int N) {
    for (int i = 0; i * 2 < N; ++i) {
        int pair_base = i * 2;
        float x0 = __half2float(x[pair_base + 0]);
        float x1 = __half2float(x[pair_base + 1]);
        int token_idx = pair_base % hidden_dim;
        int token_pos = pair_base / hidden_dim;
        float angle = (token_pos * 1.0f) / powf(theta, token_idx * 1.0f / (hidden_dim * 1.0f));
        float cos_v = cosf(angle);
        float sin_v = sinf(angle);
        float y0 = x0 * cos_v - x1 * sin_v;
        float y1 = x0 * sin_v + x1 * cos_v;
        y_ref[pair_base + 0] = y0;
        y_ref[pair_base + 1] = y1;
    }
}

static bool check_close(const half *y, const float *y_ref, int N, float atol = 1e-2f, float rtol = 1e-2f) {
    for (int i = 0; i < N; ++i) {
        float gy = __half2float(y[i]);
        float diff = fabsf(gy - y_ref[i]);
        float tol = atol + rtol * fabsf(y_ref[i]);
        if (diff > tol) {
            // 可根据需要打印前几个不匹配项
            return false;
        }
    }
    return true;
}

int main() {
    // 配置
    int hidden_dim = 128;  // 必须为偶数
    int seq_len = 4096;
    int N = hidden_dim * seq_len; // 标量 half 元素个数

    // 设备选择
    cudaSetDevice(0);

    // 主机侧分配与初始化
    half *hx = (half*)malloc(sizeof(half) * N);
    half *hy = (half*)malloc(sizeof(half) * N);
    float *y_ref = (float*)malloc(sizeof(float) * N);

    std::mt19937 rng(123);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (int i = 0; i < N; ++i) {
        hx[i] = __float2half(dist(rng));
        hy[i] = __float2half(0.0f);
    }

    // 设备侧分配
    half *dx = nullptr;
    half *dy = nullptr;
    cudaMalloc(&dx, sizeof(half) * N);
    cudaMalloc(&dy, sizeof(half) * N);

    // 拷贝输入
    cudaMemcpy(dx, hx, sizeof(half) * N, cudaMemcpyHostToDevice);

    // 启动参数：每个线程处理 1 对 (2 个元素)
    int num_pairs = (N + 1) / 2;
    int blockSize = 256;
    int gridSize = (num_pairs + blockSize - 1) / blockSize;

    // 计时并运行
    float ms = 0.0f;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    rope<<<gridSize, blockSize>>>(dx, dy, hidden_dim, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    // 回传输出
    cudaMemcpy(hy, dy, sizeof(half) * N, cudaMemcpyDeviceToHost);

    // 计算 CPU 参考
    rope_cpu_ref(hx, y_ref, hidden_dim, N);

    // 校验
    bool ok = check_close(hy, y_ref, N);
    if (ok) {
        printf("rope: the ans is right\n");
    } else {
        printf("rope: the ans is wrong\n");
    }
    printf("rope latency (single run) = %f ms\n", ms);

    // GPU 多次计时（热身 + 循环），计算平均值
    int warmup = 5;
    int iters = 50;
    for (int i = 0; i < warmup; ++i) {
        rope<<<gridSize, blockSize>>>(dx, dy, hidden_dim, N);
    }
    cudaDeviceSynchronize();
    ms = 0.0f;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    for (int i = 0; i < iters; ++i) {
        rope<<<gridSize, blockSize>>>(dx, dy, hidden_dim, N);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    float gpu_avg_ms = ms / iters;

    // CPU 多次计时（热身 + 循环），计算平均值
    for (int i = 0; i < warmup; ++i) {
        rope_cpu_ref(hx, y_ref, hidden_dim, N);
    }
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iters; ++i) {
        rope_cpu_ref(hx, y_ref, hidden_dim, N);
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    double cpu_total_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    double cpu_avg_ms = cpu_total_ms / iters;

    printf("gpu avg latency over %d iters = %.3f ms\n", iters, gpu_avg_ms);
    printf("cpu avg latency over %d iters = %.3f ms\n", iters, cpu_avg_ms);

    // 释放
    cudaFree(dx);
    cudaFree(dy);
    free(hx);
    free(hy);
    free(y_ref);
    return 0;
}