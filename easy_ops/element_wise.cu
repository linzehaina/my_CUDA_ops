#include <algorithm>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <float.h>
#include <stdio.h>
#include <stdlib.h>
// #include <torch/extension.h>
// #include <torch/types.h>
#include <vector>

#define WARP_SIZE 32
#define FLOAT4(value) (reinterpret_cast<float4 *>(&(value))[0])
#define INT4(value) (reinterpret_cast<int4 *>(&(value))[0])
#define HALF2(value) (reinterpret_cast<half2 *>(&(value))[0])
#define BFLOAT2(value) (reinterpret_cast<__nv_bfloat162 *>(&(value))[0])
#define LOAD128BIT(value) (reinterpret_cast<float4 *>(&(value))[0])


__global__ void elementwise_add_fp32(float * a, float * b, float * c, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        c[idx] = a[idx] + b[idx];
    }
}

__global__ void elementwise_add_fp32x4(float * a, float * b, float * c, int N) {
    int idx = 4 * (blockIdx.x * blockDim.x + threadIdx.x);
    if (idx < N) {
        float4 reg_a = FLOAT4(a[idx]);
        float4 reg_b = FLOAT4(b[idx]);
        float4 reg_c;
        reg_c.x = reg_a.x + reg_b.x;
        reg_c.y = reg_a.y + reg_b.y;
        reg_c.z = reg_a.z + reg_b.z;
        reg_c.w = reg_a.w + reg_b.w;

        FLOAT4(c[idx]) = reg_c;
    }
}


__global__ void elementwise_add_fp16x2(half * a, half * b, half * c, int N) {
    int idx = 2 * (blockIdx.x * blockDim.x + threadIdx.x);
    if (idx < N) {
        half2 reg_a = HALF2(a[idx]);
        half2 reg_b = HALF2(b[idx]);
        half2 reg_c;
        reg_c = __hadd2(reg_a, reg_b);
        HALF2(c[idx]) = reg_c;
    }
}

__global__ void elementwise_add_fp16(half * a, half * b, half * c, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        c[idx] = __hadd(a[idx], b[idx]);
    }
}


int main() {
    // parameters
    const int N = 1 << 24; // 16,777,216 elements (even), ~32MB per half array
    const int warmup_iters = 10;
    const int iters = 100;

    // allocate host memory
    std::vector<half> h_a(N), h_b(N), h_c(N), h_ref(N);

    // init host data (simple ramp to avoid denormals)
    for (int i = 0; i < N; ++i) {
        float fa = static_cast<float>((i % 1024) - 512) * 0.001f;
        float fb = static_cast<float>((i % 2048) - 1024) * 0.0005f;
        h_a[i] = __float2half(fa);
        h_b[i] = __float2half(fb);
    }

    // device memory
    half *d_a = nullptr, *d_b = nullptr, *d_c = nullptr;
    cudaMalloc(&d_a, N * sizeof(half));
    cudaMalloc(&d_b, N * sizeof(half));
    cudaMalloc(&d_c, N * sizeof(half));
    cudaMemcpy(d_a, h_a.data(), N * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b.data(), N * sizeof(half), cudaMemcpyHostToDevice);

    // launch config
    const int threads = 256;
    const int blocks_fp16 = (N + threads - 1) / threads;
    const int n_pairs = (N + 1) / 2; // ceil(N/2)
    const int blocks_fp16x2 = (n_pairs + threads - 1) / threads;

    // timing helpers
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // warmup fp16x2
    for (int i = 0; i < warmup_iters; ++i) {
        elementwise_add_fp16x2<<<blocks_fp16x2, threads>>>(d_a, d_b, d_c, N);
    }
    cudaDeviceSynchronize();

    // measure fp16x2
    cudaEventRecord(start);
    for (int i = 0; i < iters; ++i) {
        elementwise_add_fp16x2<<<blocks_fp16x2, threads>>>(d_a, d_b, d_c, N);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms_fp16x2 = 0.0f;
    cudaEventElapsedTime(&ms_fp16x2, start, stop);
    ms_fp16x2 /= iters; // average per iteration

    // copy for correctness baseline
    cudaMemcpy(h_c.data(), d_c, N * sizeof(half), cudaMemcpyDeviceToHost);

    // warmup fp16
    for (int i = 0; i < warmup_iters; ++i) {
        elementwise_add_fp16<<<blocks_fp16, threads>>>(d_a, d_b, d_c, N);
    }
    cudaDeviceSynchronize();

    // measure fp16 scalar
    cudaEventRecord(start);
    for (int i = 0; i < iters; ++i) {
        elementwise_add_fp16<<<blocks_fp16, threads>>>(d_a, d_b, d_c, N);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms_fp16 = 0.0f;
    cudaEventElapsedTime(&ms_fp16, start, stop);
    ms_fp16 /= iters; // average per iteration

    // correctness check: compare fp16x2 result (h_c) vs scalar result (h_ref)
    cudaMemcpy(h_ref.data(), d_c, N * sizeof(half), cudaMemcpyDeviceToHost);
    int mismatches = 0;
    for (int i = 0; i < N; ++i) {
        if (__half2float(h_c[i]) != __half2float(h_ref[i])) {
            ++mismatches;
            if (mismatches <= 5) {
                printf("mismatch at %d: %f vs %f\n", i, __half2float(h_c[i]), __half2float(h_ref[i]));
            }
        }
    }

    // compute throughput (GB/s): read a + b and write c
    const double bytes = static_cast<double>(N) * sizeof(half) * 3.0;
    const double gb = bytes / (1024.0 * 1024.0 * 1024.0);
    const double gbps_fp16x2 = gb / (ms_fp16x2 / 1000.0);
    const double gbps_fp16 = gb / (ms_fp16 / 1000.0);

    printf("N = %d (half)\n", N);
    printf("fp16x2:  avg %.3f ms, %.2f GB/s\n", ms_fp16x2, gbps_fp16x2);
    printf("fp16  :  avg %.3f ms, %.2f GB/s\n", ms_fp16, gbps_fp16);
    if (mismatches == 0) {
        printf("Results match.\n");
    } else {
        printf("Results differ: %d mismatches.\n", mismatches);
    }

    // cleanup
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return 0;
}










