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

// gridsize: (N + 127) / 128 * (M + 127) / 128   
// blocksize: 256
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

    int start_m = blockIdx.y * BM;
    int start_n = blockIdx.x * BN;
    int warpid = threadIdx.x >> 5;
    int laneid = threadIdx.x & 31;

    int load_a_smem_m = threadIdx.x >> 1;
    int load_a_smem_k = (threadIdx.x & 1) << 2;
    int load_b_smem_n = (threadIdx.x & 31) << 2;
    int load_b_smem_k = threadIdx.x >> 5;

    int load_a_global_m = start_m + load_a_smem_m;
    int load_a_global_k = load_a_smem_k;
    int load_b_global_n = start_n + load_b_smem_n;
    int load_b_global_k = load_b_smem_k;
        
    {
        FLOAT4(load_a[0]) = FLOAT4(a[OFFSET(load_a_global_m, load_a_global_k, K)]);
        s_a[0][load_a_smem_k][load_a_smem_m] = load_a[0];
        s_a[0][load_a_smem_k + 1][load_a_smem_m] = load_a[1];
        s_a[0][load_a_smem_k + 2][load_a_smem_m] = load_a[2];
        s_a[0][load_a_smem_k + 3][load_a_smem_m] = load_a[3];

        FLOAT4(load_b[0]) = FLOAT4(b[OFFSET(load_b_global_k, load_b_global_n, N)]);
        FLOAT4(s_b[0][load_b_smem_k][load_b_smem_n]) = FLOAT4(load_b[0]);
    }

    __syncthreads();


    int read_s_a_row = (warpid / 2) * 32 + (laneid & 3) * 8;
    int read_s_b_col = (warpid & 1) * 64 + (laneid >> 2) * 8;
    for(int i = 1; i < (K + BK - 1) / BK; i += 1) {
        int smem_sel = (i - 1) & 1;
        int smem_next = i & 1;

        load_a_global_k += BK;
        load_b_global_k += BK;
        FLOAT4(load_a[0]) = FLOAT4(a[OFFSET(load_a_global_m, load_a_global_k, K)]);
        FLOAT4(load_b[0]) = FLOAT4(b[OFFSET(load_b_global_k, load_b_global_n, N)]);
        
        #pragma unroll
        for (int j = 0; j < BK; j++) {
            
            FLOAT4(comp_a[0]) = FLOAT4(s_a[smem_sel][j][read_s_a_row]);
            FLOAT4(comp_a[4]) = FLOAT4(s_a[smem_sel][j][read_s_a_row + 4]);
            FLOAT4(comp_b[0]) = FLOAT4(s_b[smem_sel][j][read_s_b_col]);
            FLOAT4(comp_b[4]) = FLOAT4(s_b[smem_sel][j][read_s_b_col + 4]);

            #pragma unroll
            for (int row = 0; row < TM; row++) {
                #pragma unroll
                for (int col = 0; col < TN; col++) {
                    reg_c[row][col] += comp_a[row] * comp_b[col];
                }
            }
        }

        s_a[smem_next][load_a_smem_k][load_a_smem_m] = load_a[0];
        s_a[smem_next][load_a_smem_k + 1][load_a_smem_m] = load_a[1];
        s_a[smem_next][load_a_smem_k + 2][load_a_smem_m] = load_a[2];
        s_a[smem_next][load_a_smem_k + 3][load_a_smem_m] = load_a[3];
        FLOAT4(s_b[smem_next][load_b_smem_k][load_b_smem_n]) = FLOAT4(load_b[0]);

        __syncthreads();
    }


    #pragma unroll
    for (int j = 0; j < BK; j++) {
        FLOAT4(comp_a[0]) = FLOAT4(s_a[1][j][read_s_a_row]);
        FLOAT4(comp_a[4]) = FLOAT4(s_a[1][j][read_s_a_row + 4]);
        FLOAT4(comp_b[0]) = FLOAT4(s_b[1][j][read_s_b_col]);
        FLOAT4(comp_b[4]) = FLOAT4(s_b[1][j][read_s_b_col + 4]);

        #pragma unroll
        for (int row = 0; row < TM; row++) {
            #pragma unroll
            for (int col = 0; col < TN; col++) {
                reg_c[row][col] += comp_a[row] * comp_b[col];
            }
        }
    }

    int store_c_global_m = BM * blockIdx.y + read_s_a_row;
    int store_c_global_n = BN * blockIdx.x + read_s_b_col;
    #pragma unroll
    for(int row = 0; row < TM; row++) {
        FLOAT4(c[OFFSET(store_c_global_m + row, store_c_global_n, N)]) = FLOAT4(reg_c[row][0]);
        FLOAT4(c[OFFSET(store_c_global_m + row, store_c_global_n + 4, N)]) = FLOAT4(reg_c[row][4]);
    }
}



void cpuSgemm(
    float *a, float *b, float *c, const int M, const int N, const int K) {

    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {
            float psum = 0.0;
            for (int k = 0; k < K; k++) {
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
        const int M = 128, N = 128, K = 256;
        dim3 blockDim(BN / TN * BM / TM);
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

        dim3 blockDim(BN / TN * BM / TM);
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