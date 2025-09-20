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
#include "macro.h"

using namespace std;

#define WARP_SIZE 32
#define FLOAT4(value) (reinterpret_cast<float4 *>(&(value))[0])
#define INT4(value) (reinterpret_cast<int4 *>(&(value))[0])
#define HALF2(value) (reinterpret_cast<half2 *>(&(value))[0])
#define BFLOAT2(value) (reinterpret_cast<__nv_bfloat162 *>(&(value))[0])
#define LOAD128BIT(value) (reinterpret_cast<float4 *>(&(value))[0])
#define OFFSET(row, col, ld) ((row) * (ld) + (col))

float testError(
    void (*gpuSgemm)(float *, float *, float *, const int, const int, const int),
    dim3 gridDim, dim3 blockDim, const int M, const int N, const int K);
float testPerformance(
    void (*gpuSgemm)(float *, float *, float *, const int, const int, const int),
    dim3 gridDim, dim3 blockDim, const int M, const int N, const int K, const int repeat);

//K must be a multiple of 16 
//N must be a multiple of 4
// gridsize: (N + BN - 1) / BN * (M + BM - 1) / BM   
// blocksize: BN / TN * BM / TM
__global__ void sgemm(float * __restrict__ a, float * __restrict__ b, float * __restrict__ c, 
    const int M, const int N, const int K) {
    const int BM = 128;
    const int BN = 128;
    const int BK = 8;
    const int TM = 8;
    const int TN = 8;

    __shared__ float s_a[2][BK][BM];
    __shared__ float s_b[2][BK][BN];

    float load_a[4];
    float load_b[4];
    float comp_a[8];
    float comp_b[8];
    float reg_c[TM][TN] = {0.0};
    int tid = threadIdx.y * blockDim.x + threadIdx.x;

    int load_a_smem_m = tid >> 1;
    int load_a_smem_k = (tid & 1) << 2;
    int load_b_smem_n = (tid & 31) << 2;
    int load_b_smem_k = tid >> 5;
    
    int load_a_global_m = blockIdx.y * BM + load_a_smem_m;
    int load_a_global_k = load_a_smem_k;
    int load_b_global_n = blockIdx.x * BN + load_b_smem_n;
    int load_b_global_k = load_b_smem_k;
    
    {
        if (load_a_global_m < M)
            FLOAT4(load_a[0]) = FLOAT4(a[OFFSET(load_a_global_m, load_a_global_k, K)]);

        if (load_b_global_n < N)
            FLOAT4(load_b[0]) = FLOAT4(b[OFFSET(load_b_global_k, load_b_global_n, N)]);

        s_a[0][load_a_smem_k][load_a_smem_m] = load_a[0];
        s_a[0][load_a_smem_k + 1][load_a_smem_m] = load_a[1];
        s_a[0][load_a_smem_k + 2][load_a_smem_m] = load_a[2];
        s_a[0][load_a_smem_k + 3][load_a_smem_m] = load_a[3];
        FLOAT4(s_b[0][load_b_smem_k][load_b_smem_n]) = FLOAT4(load_b[0]);
    }

    __syncthreads();

    for (int i = 1; i < (K + BK - 1) / BK; i++) {
        int smem_sel = (i - 1) & 1;
        int smem_sel_next = i & 1;

        load_a_global_k += BK;
        load_b_global_k += BK;
        if (load_a_global_m < M)
            FLOAT4(load_a[0]) = FLOAT4(a[OFFSET(load_a_global_m, load_a_global_k, K)]);
        if (load_b_global_n < N)
            FLOAT4(load_b[0]) = FLOAT4(b[OFFSET(load_b_global_k, load_b_global_n, N)]);

        //calculate
        for(int j = 0; j < BK; j++) {
            FLOAT4(comp_a[0]) = FLOAT4(s_a[smem_sel][j][threadIdx.y * 4]);
            FLOAT4(comp_a[4]) = FLOAT4(s_a[smem_sel][j][threadIdx.y * 4 + BM / 2]);
            FLOAT4(comp_b[0]) = FLOAT4(s_b[smem_sel][j][threadIdx.x * 4]);
            FLOAT4(comp_b[4]) = FLOAT4(s_b[smem_sel][j][threadIdx.x * 4 + BN / 2]);

            for (int row = 0; row < TM; row++) {
                for (int col = 0; col < TN; col++) {
                    reg_c[row][col] += comp_a[row] * comp_b[col];
                }
            }
        }

        s_a[smem_sel_next][load_a_smem_k][load_a_smem_m] = load_a[0];
        s_a[smem_sel_next][load_a_smem_k + 1][load_a_smem_m] = load_a[1];
        s_a[smem_sel_next][load_a_smem_k + 2][load_a_smem_m] = load_a[2];
        s_a[smem_sel_next][load_a_smem_k + 3][load_a_smem_m] = load_a[3];
        FLOAT4(s_b[smem_sel_next][load_b_smem_k][load_b_smem_n]) = FLOAT4(load_b[0]);

        __syncthreads();
    }

    #pragma unroll
    for(int j = 0; j < BK; j++) {
        FLOAT4(comp_a[0]) = FLOAT4(s_a[1][j][threadIdx.y * 4]);
        FLOAT4(comp_a[4]) = FLOAT4(s_a[1][j][threadIdx.y * 4 + BM / 2]);
        FLOAT4(comp_b[0]) = FLOAT4(s_b[1][j][threadIdx.x * 4]);
        FLOAT4(comp_b[4]) = FLOAT4(s_b[1][j][threadIdx.x * 4 + BN / 2]);

        #pragma unroll
        for (int row = 0; row < TM; row++) {
            #pragma unroll
            for (int col = 0; col < TN; col++) {
                reg_c[row][col] += comp_a[row] * comp_b[col];
            }
        }
    }

    // store
    int store_c_smem_m = threadIdx.y * 4;
    int store_c_smem_n = threadIdx.x * 4;

    int store_c_global_m = blockIdx.y * BM + store_c_smem_m;
    int store_c_global_n = blockIdx.x * BN + store_c_smem_n;

    #pragma unroll
    for (int row = 0; row < TM / 2; row ++) {
        if (store_c_global_m + row < M & store_c_global_n < N)
            FLOAT4(c[OFFSET(store_c_global_m + row, store_c_global_n, N)]) = FLOAT4(reg_c[row][0]);
        
        if (store_c_global_m + row < M & store_c_global_n + BN / 2 < N)
            FLOAT4(c[OFFSET(store_c_global_m + row, store_c_global_n + BN / 2, N)]) = FLOAT4(reg_c[row][4]);
    }

    #pragma unroll
    for (int row = 4; row < TM; row ++) {
        if (store_c_global_m + row - 4 + BM / 2 < M & store_c_global_n < N)
            FLOAT4(c[OFFSET(store_c_global_m + row - 4 + BM / 2, store_c_global_n, N)]) = FLOAT4(reg_c[row][0]);

        if (store_c_global_m + row - 4 + BM / 2 < M & store_c_global_n + BN / 2 < N)
            FLOAT4(c[OFFSET(store_c_global_m + row - 4 + BM / 2, store_c_global_n + BN / 2, N)]) = FLOAT4(reg_c[row][4]);
    }
}

void cpuSgemm(
    float *a, float *b, float *c, const int M, const int N, const int K)
{

    for (int m = 0; m < M; m++)
    {
        for (int n = 0; n < N; n++)
        {
            float psum = 0.0;
            for (int k = 0; k < K; k++)
            {
                psum += a[OFFSET(m, k, K)] * b[OFFSET(k, n, N)];
            }
            c[OFFSET(m, n, N)] = psum;
        }
    }
}

int main(void)
{
    printf("\nKernal = sgemm\n");
    const int outer_repeat = 10, inner_repeat = 1;
    const int BM = 128, BN = 128, TM = 8, TN = 8;
    void (*gpuSgemm)(float *, float *, float *, const int, const int, const int) = sgemm;

    {
        const int M = 1131, N = 1132, K = 816;
        dim3 blockDim(BN / TN , BM / TM);
        dim3 gridDim((N + BN - 1) / BN, (M + BM - 1) / BM);
        float max_error = testError(gpuSgemm, gridDim, blockDim, M, N, K);
        printf("Max Error = %f\n", max_error);
    }

    // return 0;
    const int M_list[12] = { 128, 256, 512, 768, 1024, 1536, 2048, 3072, 4096, 6144, 8192};
    const int N_list[12] = { 128, 256, 512, 768, 1024, 1536, 2048, 3072, 4096, 6144, 8192};
    const int K_list[12] = { 128, 256, 512, 768, 1024, 1536, 2048, 3072, 4096, 6144, 8192};
    
    const int TESTNUM = 11;
    for (int i = 0; i < TESTNUM; i++)
    {
        const int M = M_list[i], N = N_list[i], K = K_list[i];

        dim3 blockDim(BN / TN , BM / TM);
        dim3 gridDim((N + BN - 1) / BN, (M + BM - 1) / BM);

        double max_sec = 0.0;
        double min_sec = DBL_MAX;
        double total_sec = 0.0;

        for (int j = 0; j < outer_repeat; j++)
        {
            double this_sec = testPerformance(gpuSgemm, gridDim, blockDim, M, N, K, inner_repeat);
            max_sec = max(max_sec, this_sec);
            min_sec = min(min_sec, this_sec);
            total_sec += this_sec;
        }

        double avg_sec = total_sec / outer_repeat;
        double avg_Gflops = ((double)M) * N * K * 2 / 1024 / 1024 / 1024 / avg_sec;

        printf("M N K = %6d %6d %6d, Time = %12.8lf %12.8lf %12.8lf s, AVG Performance = %10.4lf Gflops\n", M, N, K, min_sec, avg_sec, max_sec, avg_Gflops);
    }
    return 0;
}

float testError(
    void (*gpuSgemm)(float *, float *, float *, const int, const int, const int),
    dim3 gridDim, dim3 blockDim, const int M, const int N, const int K)
{

    size_t size_a = M * K * sizeof(float);
    size_t size_b = K * N * sizeof(float);
    size_t size_c = M * N * sizeof(float);

    float *h_a, *h_b, *h_c, *d_a, *d_b, *d_c, *h_d_c;
    h_a = (float *)malloc(size_a);
    h_b = (float *)malloc(size_b);
    h_c = (float *)malloc(size_c);
    cudaMalloc(&d_a, size_a);
    cudaMalloc(&d_b, size_b);
    cudaMalloc(&d_c, size_c);
    h_d_c = (float *)malloc(size_c);

    srand(time(0));
    for (int i = 0; i < M * K; i++)
        h_a[i] = 1 * rand() / float(RAND_MAX);
    for (int i = 0; i < K * N; i++)
        h_b[i] = 1 * rand() / float(RAND_MAX);
    cudaMemset(d_c, 15, size_c);

    cpuSgemm(h_a, h_b, h_c, M, N, K);

    cudaMemcpy(d_a, h_a, size_a, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size_b, cudaMemcpyHostToDevice);
    gpuSgemm<<<gridDim, blockDim>>>(d_a, d_b, d_c, M, N, K);
    CHECK(cudaMemcpy(h_d_c, d_c, size_c, cudaMemcpyDeviceToHost));

    float max_error = 0.0;
    float max_error_percentage = 0.0;
    float avg1 = 0.0;
    float avg2 = 0.0;
    for (int i = 0; i < M * N; i++)
    {
        float this_error = abs(h_d_c[i] - h_c[i]);
        float this_error_percentage = abs(h_d_c[i] - h_c[i]) / (h_c[i]);
        avg1 += h_d_c[i];
        avg2 += h_c[i];
        if (max_error != max_error || this_error != this_error) // nan
            max_error = -NAN;
        else
            max_error = max(max_error, this_error);

        if(max_error_percentage != max_error_percentage || this_error_percentage != this_error_percentage)
            max_error_percentage = NAN;
        else {
            // if (this_error_percentage > max_error_percentage) {
            //     cout << "position: " << i / M << " " << i % N <<  " " << this_error_percentage << endl;
            //     cout << "h_d_c: " << h_d_c[i] << ", h_c: " << h_c[i] << endl;
            // }
            max_error_percentage = max(max_error_percentage, this_error_percentage);
        }
    }

    cout << "Average of GPU result: " << avg1 / (M*N)<< endl;
    cout << "Average of CPU result: " << avg2 / (M*N)<< endl;
    cout << "Max error percentage: " << max_error_percentage * 100 << "%" << endl;
    cout << "Max error: " << max_error << endl;

    free(h_a);
    free(h_b);
    free(h_c);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    free(h_d_c);

    return max_error;
}

float testPerformance(
    void (*gpuSgemm)(float *, float *, float *, const int, const int, const int),
    dim3 gridDim, dim3 blockDim, const int M, const int N, const int K, const int repeat)
{

    size_t size_a = M * K * sizeof(float);
    size_t size_b = K * N * sizeof(float);
    size_t size_c = M * N * sizeof(float);

    float *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, size_a);
    cudaMalloc(&d_b, size_b);
    cudaMalloc(&d_c, size_c);

    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);
    CHECK(cudaEventRecord(start));
    for (int i = 0; i < repeat; i++)
        gpuSgemm<<<gridDim, blockDim>>> (d_a, d_b, d_c, M, N, K);
    CHECK(cudaEventRecord(end));
    CHECK(cudaEventSynchronize(end));

    float msec, sec;
    cudaEventElapsedTime(&msec, start, end);
    sec = msec / 1000.0 / repeat;

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return sec;
}




