#ifdef USE_DP
typedef double real;
#else
typedef float real;
#endif

#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>

//CUDA错误检测宏
#define CHECK(call)                                     \
    do                                                  \
    {                                                   \
        const cudaError_t error_code = call;            \
        if (error_code != cudaSuccess)                  \
        {                                               \
            printf("CUDA Error:\n");                    \
            printf("    File:       %s\n", __FILE__);   \
            printf("    Line:       %d\n", __LINE__);   \
            printf("    Error code: %d\n", error_code); \
            printf("    Error text: %s\n",              \
                   cudaGetErrorString(error_code));     \
            exit(1);                                    \
        }                                               \
    } while (0)

const int NUM_REPEATS = 10;
const int N = 100000000;
const int M = sizeof(real) * N;
const int block_size = 128;
const int grid_size = (N - 1) / block_size + 1;

// 封装 timing 函数, 通过参数 ratio 与 overlap,
// 调节 cpu 计算的复杂度与是否重叠主机与设备的计算
void timing(const real *h_x,
            const real *h_y,
            real *h_z,
            const real *d_x,
            const real *d_y,
            real *d_z,
            const int ratio,
            bool overlap);

int main()
{
    real *h_x = (real *)malloc(M);
    real *h_y = (real *)malloc(M);
    real *h_z = (real *)malloc(M);
    for (int n = 0; n < N; ++n)
    {
        h_x[n] = 1.23;
        h_y[n] = 2.34;
    }

    real *d_x, *d_y, *d_z;
    CHECK(cudaMalloc(&d_x, M));
    CHECK(cudaMalloc(&d_y, M));
    CHECK(cudaMalloc(&d_z, M));
    CHECK(cudaMemcpy(d_x, h_x, M, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_y, h_y, M, cudaMemcpyHostToDevice));

    printf("Without CPU-GPU overlap (ratio = 10)\n");
    timing(h_x, h_y, h_z, d_x, d_y, d_z, 10, false);
    printf("With CPU-GPU overlap (ratio = 10)\n");
    timing(h_x, h_y, h_z, d_x, d_y, d_z, 10, true);

    printf("Without CPU-GPU overlap (ratio = 1)\n");
    timing(h_x, h_y, h_z, d_x, d_y, d_z, 1, false);
    printf("With CPU-GPU overlap (ratio = 1)\n");
    timing(h_x, h_y, h_z, d_x, d_y, d_z, 1, true);

    printf("Without CPU-GPU overlap (ratio = 1000)\n");
    timing(h_x, h_y, h_z, d_x, d_y, d_z, 1000, false);
    printf("With CPU-GPU overlap (ratio = 1000)\n");
    timing(h_x, h_y, h_z, d_x, d_y, d_z, 1000, true);

    free(h_x);
    free(h_y);
    free(h_z);
    CHECK(cudaFree(d_x));
    CHECK(cudaFree(d_y));
    CHECK(cudaFree(d_z));

    return 0;
}

void cpu_sum(const real *x, const real *y, real *z, const int N_host)
{
    for (int n = 0; n < N_host; ++n)
    {
        z[n] = x[n] + y[n];
    }
}

void __global__ gpu_sum(const real *x, const real *y, real *z)
{
    const int n = blockDim.x * blockIdx.x + threadIdx.x;
    if (n < N)
    {
        z[n] = x[n] + y[n];
    }
}

void timing(const real *h_x,
            const real *h_y,
            real *h_z,
            const real *d_x,
            const real *d_y,
            real *d_z,
            const int ratio,
            bool overlap)
{
    float t_sum = 0;
    float t2_sum = 0;

    for (int repeat = 0; repeat <= NUM_REPEATS; ++repeat)
    {
        cudaEvent_t start, stop;
        CHECK(cudaEventCreate(&start));
        CHECK(cudaEventCreate(&stop));
        CHECK(cudaEventRecord(start));
        cudaEventQuery(start);

        // overlap == false, 意味不重叠计算, 所以先执行 cpu 计算, 设备会在此期间闲置等待
        // ratio 越高,  N / ratio 越小, cpu 执行的运算越少, 时间也就越短
        // ratio 可以用来调整 cpu 计算的时间长短, 以此对加速比进行实验
        if (!overlap)
        {
            cpu_sum(h_x, h_y, h_z, N / ratio);
        }

        gpu_sum<<<grid_size, block_size>>>(d_x, d_y, d_z);

        // overlap == true, 重叠主机计算与设备计算.
        if (overlap)
        {
            cpu_sum(h_x, h_y, h_z, N / ratio);
        }

        CHECK(cudaEventRecord(stop));
        CHECK(cudaEventSynchronize(stop));
        float elapsed_time;
        CHECK(cudaEventElapsedTime(&elapsed_time, start, stop));
        printf("Time = %g ms.\n", elapsed_time);

        if (repeat > 0)
        {
            t_sum += elapsed_time;
            t2_sum += elapsed_time * elapsed_time;
        }

        CHECK(cudaEventDestroy(start));
        CHECK(cudaEventDestroy(stop));
    }

    const float t_ave = t_sum / NUM_REPEATS;
    const float t_err = sqrt(t2_sum / NUM_REPEATS - t_ave * t_ave);
    printf("Time = %g +- %g ms.\n", t_ave, t_err);
}
