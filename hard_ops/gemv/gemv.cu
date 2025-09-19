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
__device__ T warp_reduce_sum(T val) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_xor_sync(0xffffffff, val, offset);
    }
    return val;
}

template<typename T = half>
__device__ half warp_reduce_sum(half val) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        val = __hadd(val, __shfl_xor_sync(0xffffffff, val, offset));
    }
    return val;
}

template<typename T>
__device__ T block_reduce_sum(T val) {
    int laneid = threadIdx.x & 31;
    int warpid = threadIdx.x >> 5;
    int warpnum = (blockDim.x + 31) >> 5;

    __shared__ T warpsum[32];
    val = warp_reduce_sum(val);
    if (laneid == 0) {
        warpsum[warpid] = val;
    }
    __syncthreads();

    if (threadIdx.x < 32) {
        T acc = (laneid < warpnum) ? warpsum[laneid] : (T)0;
        acc = warp_reduce_sum(acc);
        if (laneid == 0) warpsum[0] = acc;
    }
    __syncthreads();

    return warpsum[0];
}



// (M * K) x (K * 1)
// gridsize: M / 16    blocksize: 32 * 16
__global__ void sgemv(float *mat, float *vec, float * dst, int M, int K){
    // int global_tid = threadIdx.x + blockIdx.x * blockDim.x;
    int laneid = threadIdx.x & 31;
    int row = blockIdx.x * blockDim.y + threadIdx.y;

    if (row < M) {
        float sum = 0.0f;
        int group = (K + 32 - 1) / 32;
        for (int i = 0; i < group; i++) {
            int index = i * 32 + laneid;
            if (index < K) {
                sum += mat[row * K + index] * vec[index];
            }
        }
        sum = warp_reduce_sum(sum);
        if (laneid == 0) {
            dst[row] = sum;
        }
    }
}

// (M * K) x (K * 1)
// gridsize: M  blocksize: 1024
__global__ void sgemv_block(float *mat, float *vec, float * dst, int M, int K) {
    int tid = threadIdx.x;
    int row = blockIdx.x;
    float sum = 0.0f;

    for (int i = tid; i < K; i += blockDim.x) {
        sum += mat[row * K + i] * vec[i];
    }
    __syncthreads();
    sum = block_reduce_sum(sum);
    if (tid == 0) {
        dst[row] = sum;
    }
}


// (M * K) x (K * 1)
// gridsize: M / 16  blocksize: 32 * 16
__global__ void hgemv_f16x8(half * mat, half * vec, half * dst, int M, int K) {
    int row = blockIdx.x * blockDim.y + threadIdx.y;
    int laneid = threadIdx.x & 31;
    half sum = 0;

    if (row < M) {
        for (int idx = laneid; idx < K / 8; idx += 32) {
            float4 val_mat = LOAD128BIT(mat[row * K + idx * 8]);
            float4 val_vec = LOAD128BIT(vec[idx * 8]);

            half2 first = __hmul2(*(half2*)(&val_mat.x), *(half2*)(&val_vec.x));
            half2 second = __hmul2(*(half2*)(&val_mat.y), *(half2*)(&val_vec.y));
            half2 third = __hmul2(*(half2*)(&val_mat.z), *(half2*)(&val_vec.z));
            half2 forth = __hmul2(*(half2*)(&val_mat.w), *(half2*)(&val_vec.w));
            half2 tmp = __hadd2(__hadd2(first, second), __hadd2(third, forth));
            half pair_sum = __hadd(tmp.x, tmp.y);
            sum = __hadd(sum, pair_sum);
        }
        sum = warp_reduce_sum(sum);
        if (laneid == 0) dst[row] = sum;
    }
}

// CPU 参考实现（float）
static void sgemv_cpu_ref(const std::vector<float>& mat, const std::vector<float>& vec, std::vector<float>& out, int M, int K) {
    for (int r = 0; r < M; ++r) {
        float acc = 0.0;
        const float* rowp = &mat[r * K];
        for (int c = 0; c < K; ++c) acc += rowp[c] * vec[c];
        out[r] = acc;
    }
}

// CPU 参考实现（half）
static void hgemv_cpu_ref(const std::vector<half>& mat, const std::vector<half>& vec, std::vector<half>& out, int M, int K) {
    for (int r = 0; r < M; ++r) {
        half acc = 0.0;
        const half* rowp = &mat[r * K];
        // for (int c = 0; c < K; ++c) acc += __half2float(rowp[c]) * __half2float(vec[c]);
        // out[r] = __float2half(acc);
        for (int c = 0; c < K; ++c) acc += __hmul(rowp[c], vec[c]);
        out[r] = acc;
    }
}

static bool check_close_vec(const std::vector<float>& a, const std::vector<float>& b, float atol = 1e-4f, float rtol = 1e-4f) {
    if (a.size() != b.size()) return false;
    for (size_t i = 0; i < a.size(); ++i) {
        float diff = fabsf(a[i] - b[i]);
        float tol = atol + rtol * fabsf(b[i]);
        if (diff > tol) 
        {
            printf("diff is %f\n", diff);
            return false;
        }
    }
    return true;
}

static bool check_close_vec_half(const std::vector<half>& a, const std::vector<half>& b, float atol = 1.0f, float rtol = 1e-2f) {
    if (a.size() != b.size()) return false;
    for (size_t i = 0; i < a.size(); ++i) {
        float va = __half2float(a[i]);
        float vb = __half2float(b[i]);
        float diff = fabsf(va - vb);
        float tol = atol + rtol * fabsf(b[i]);
        if (diff > tol) 
        {
            printf("diff is %f\n", diff);
            return false;
        }
    }
    return true;
}

int main() {
    // 规模设置
    int M = 4096;
    int K = 4096;

    // 随机初始化
    std::mt19937 rng(123);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    std::vector<float> h_mat_f(M * K), h_vec_f(K), h_out_f(M), h_out_ref_f(M);
    for (int i = 0; i < M * K; ++i) h_mat_f[i] = dist(rng);
    for (int i = 0; i < K; ++i) h_vec_f[i] = dist(rng);

    // CPU 参考
    sgemv_cpu_ref(h_mat_f, h_vec_f, h_out_ref_f, M, K);

    // 设备内存（float）
    float *d_mat_f = nullptr, *d_vec_f = nullptr, *d_out_f = nullptr;
    cudaMalloc(&d_mat_f, sizeof(float) * M * K);
    cudaMalloc(&d_vec_f, sizeof(float) * K);
    cudaMalloc(&d_out_f, sizeof(float) * M);
    cudaMemcpy(d_mat_f, h_mat_f.data(), sizeof(float) * M * K, cudaMemcpyHostToDevice);
    cudaMemcpy(d_vec_f, h_vec_f.data(), sizeof(float) * K, cudaMemcpyHostToDevice);

    // 启动参数（float 版本）
    dim3 block1(32, 16);
    dim3 grid1((M + block1.y - 1) / block1.y);

    // 正确性 + 单次计时
    cudaEvent_t start, stop;
    float ms = 0.0f;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    sgemv<<<grid1, block1>>>(d_mat_f, d_vec_f, d_out_f, M, K);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaMemcpy(h_out_f.data(), d_out_f, sizeof(float) * M, cudaMemcpyDeviceToHost);
    bool ok_f = check_close_vec(h_out_f, h_out_ref_f);
    printf("sgemv correctness: %s\n", ok_f ? "ok" : "fail");
    printf("sgemv latency (single run) = %.3f ms\n", ms);

    // 多次计时（float）
    int warmup = 3, iters = 20;
    for (int i = 0; i < warmup; ++i) sgemv<<<grid1, block1>>>(d_mat_f, d_vec_f, d_out_f, M, K);
    cudaDeviceSynchronize();
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    for (int i = 0; i < iters; ++i) sgemv<<<grid1, block1>>>(d_mat_f, d_vec_f, d_out_f, M, K);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    printf("sgemv gpu avg latency over %d iters = %.3f ms\n", iters, ms / iters);

    // CPU 多次计时（float）
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iters; ++i) sgemv_cpu_ref(h_mat_f, h_vec_f, h_out_ref_f, M, K);
    auto t1 = std::chrono::high_resolution_clock::now();
    double cpu_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    printf("sgemv cpu avg latency over %d iters = %.3f ms\n", iters, cpu_ms / iters);

    // sgemv_block 正确性与性能测试
    // 选择 blockSize 不超过 1024，且对 K 做 stride 累加
    int blockSizeBlk = std::min(1024, K);
    dim3 blockBlk(blockSizeBlk, 1, 1);
    dim3 gridBlk(M, 1, 1);

    // 单次计时 + 校验
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    sgemv_block<<<gridBlk, blockBlk>>>(d_mat_f, d_vec_f, d_out_f, M, K);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaMemcpy(h_out_f.data(), d_out_f, sizeof(float) * M, cudaMemcpyDeviceToHost);
    bool ok_f_blk = check_close_vec(h_out_f, h_out_ref_f);
    printf("sgemv_block correctness: %s\n", ok_f_blk ? "ok" : "fail");
    printf("sgemv_block latency (single run) = %.3f ms\n", ms);

    // 多次计时
    for (int i = 0; i < warmup; ++i) sgemv_block<<<gridBlk, blockBlk>>>(d_mat_f, d_vec_f, d_out_f, M, K);
    cudaDeviceSynchronize();
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    for (int i = 0; i < iters; ++i) sgemv_block<<<gridBlk, blockBlk>>>(d_mat_f, d_vec_f, d_out_f, M, K);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    printf("sgemv_block gpu avg latency over %d iters = %.3f ms\n", iters, ms / iters);

    // half 数据准备
    std::vector<half> h_mat_h(M * K), h_vec_h(K), h_out_h(M), h_out_ref_h(M);
    for (int i = 0; i < M * K; ++i) h_mat_h[i] = __float2half(h_mat_f[i]);
    for (int i = 0; i < K; ++i) h_vec_h[i] = __float2half(h_vec_f[i]);
    hgemv_cpu_ref(h_mat_h, h_vec_h, h_out_ref_h, M, K);

    // 设备内存（half）
    half *d_mat_h = nullptr, *d_vec_h = nullptr, *d_out_h = nullptr;
    cudaMalloc(&d_mat_h, sizeof(half) * M * K);
    cudaMalloc(&d_vec_h, sizeof(half) * K);
    cudaMalloc(&d_out_h, sizeof(half) * M);
    cudaMemcpy(d_mat_h, h_mat_h.data(), sizeof(half) * M * K, cudaMemcpyHostToDevice);
    cudaMemcpy(d_vec_h, h_vec_h.data(), sizeof(half) * K, cudaMemcpyHostToDevice);

    // 启动参数（half 版本与 float 相同）
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    hgemv_f16x8<<<grid1, block1>>>(d_mat_h, d_vec_h, d_out_h, M, K);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaMemcpy(h_out_h.data(), d_out_h, sizeof(half) * M, cudaMemcpyDeviceToHost);
    bool ok_h = check_close_vec_half(h_out_h, h_out_ref_h);
    printf("hgemv correctness: %s\n", ok_h ? "ok" : "fail");
    printf("hgemv latency (single run) = %.3f ms\n", ms);

    for (int i = 0; i < warmup; ++i) hgemv_f16x8<<<grid1, block1>>>(d_mat_h, d_vec_h, d_out_h, M, K);
    cudaDeviceSynchronize();
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    for (int i = 0; i < iters; ++i) hgemv_f16x8<<<grid1, block1>>>(d_mat_h, d_vec_h, d_out_h, M, K);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    printf("hgemv gpu avg latency over %d iters = %.3f ms\n", iters, ms / iters);

    t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iters; ++i) hgemv_cpu_ref(h_mat_h, h_vec_h, h_out_ref_h, M, K);
    t1 = std::chrono::high_resolution_clock::now();
    cpu_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    printf("hgemv cpu avg latency over %d iters = %.3f ms\n", iters, cpu_ms / iters);

    // 释放
    cudaFree(d_mat_f); cudaFree(d_vec_f); cudaFree(d_out_f);
    cudaFree(d_mat_h); cudaFree(d_vec_h); cudaFree(d_out_h);
    return 0;
}
