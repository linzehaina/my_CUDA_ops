#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>

// using namespace std;

//CUDA错误检测宏
#define CHECK_CUDA(call)                                \
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

template <int blockSize>
__device__ void warpSharedMemReduce(float *mem)
{
    float x = mem[threadIdx.x];
    if (blockSize >= 64)
    {
        x += mem[threadIdx.x + 32];
        __syncwarp();
        mem[threadIdx.x] = x;
        __syncwarp();
    }
    x += mem[threadIdx.x + 16];
    __syncwarp();
    mem[threadIdx.x] = x;
    __syncwarp();
    x += mem[threadIdx.x + 8];
    __syncwarp();
    mem[threadIdx.x] = x;
    __syncwarp();
    x += mem[threadIdx.x + 4];
    __syncwarp();
    mem[threadIdx.x] = x;
    __syncwarp();
    x += mem[threadIdx.x + 2];
    __syncwarp();
    mem[threadIdx.x] = x;
    __syncwarp();
    x += mem[threadIdx.x + 1];
    __syncwarp();
    mem[threadIdx.x] = x;
    __syncwarp();
}

template <int blockSize>
__device__ void blockReduce(float *mem)
{
    if (blockSize >= 1024)
    {
        // if (threadIdx.x < 512)
        // {
        //     mem[threadIdx.x] += mem[threadIdx.x + 512];
        // }
        mem[threadIdx.x] += mem[threadIdx.x ^ 512];
        __syncthreads();
    }
    if (blockSize >= 512)
    {
        // if (threadIdx.x < 256)
        // {
        //     mem[threadIdx.x] += mem[threadIdx.x + 256];
        // }
        mem[threadIdx.x] += mem[threadIdx.x ^ 256];
        __syncthreads();
    }
    if (blockSize >= 256)
    {
        // if (threadIdx.x < 128)
        // {
        //     mem[threadIdx.x] += mem[threadIdx.x + 128];
        // }
        mem[threadIdx.x] += mem[threadIdx.x ^ 128];
        __syncthreads();
    }
    if (blockSize >= 128)
    {
        // if (threadIdx.x < 64)
        // {
        //     mem[threadIdx.x] += mem[threadIdx.x + 64];
        // }
        mem[threadIdx.x] += mem[threadIdx.x ^ 64];
        __syncthreads();
    }
}

template <int blockSize>
__global__ void myReduce(float *in, float *out, int N)
{
    __shared__ float mem[blockSize];
    unsigned int tid = threadIdx.x;
    unsigned int gtid = blockIdx.x * blockDim.x + tid;
    unsigned int total_thread_sum = blockDim.x * gridDim.x;

    // 初步加数据
    float sum = 0.0f;
    for (unsigned int i = gtid; i < N; i += total_thread_sum)
    {
        sum = sum + in[i];
    }
    mem[tid] = sum;
    __syncthreads();

    // for (int i = blockSize / 2; i > 32; i >>= 1) {
    //     if(tid < i) {
    //         mem[tid] = mem[tid] + mem[tid + i];
    //     }
    //     __syncthreads();
    // }

    blockReduce<blockSize>(mem);
    // #pragma unroll
    // for (int s = 128; s > 32; s >>= 1) {
    //     if (tid < s) {
    //         mem[tid] += mem[tid + s];
    //     }
    //     __syncthreads();
    // }

    if (tid < 32)
    {
        warpSharedMemReduce<blockSize>(mem);
    }

    // //使用下面这段代码，可以观察到GPU-Util在一段时间内被用满
    // uint64_t start = clock64(); // 获取当前时钟周期
    // uint64_t delay = 50000000; // 延迟10000000个时钟周期

    // while (clock64() - start < delay) {
    //     // 空循环，等待延迟时间结束
    // }

    if (tid == 0)
    {
        atomicAdd(&out[0], mem[0]);
        // out[blockIdx.x] = mem[0];
        // if (N == 128)
        // {
        //     printf("%f\n\n", out[0]);
        // }
    }

}

bool checkResult(float *in, float *cuda_out, int N)
{
    double sum = 0.0f;
    for (int i = 0; i < N; i++)
    {
        // if (i % (N / 5) == 0)
        // {
        //     std::cout << sum << std::endl;
        // }
        sum += in[i];
    }
    // std ::cout << "sum:" << sum << std::endl;
    return sum == *cuda_out;
}

int main()
{
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    const int N = 25600000;
    const int blockSize = 256;
    // const int gridSize = min((N + blockSize - 1) / blockSize, prop.maxGridSize[0]) / 32;
    const int gridSize = 128;

    float *h_in = (float *)malloc(N * sizeof(float));
    float *d_in, *part_out, *d_out;
    cudaMalloc((void **)&d_in, N * sizeof(float));
    cudaMalloc((void **)&part_out, gridSize * sizeof(float));
    cudaMalloc((void **)&d_out, 1 * sizeof(float));
    float *cuda_out = (float *)malloc(sizeof(float));
    float *one_pass_out = (float *)malloc(1 * sizeof(float));

    *cuda_out = 0.0f;
    *one_pass_out = 0.0f;

    for (int i = 0; i < N; i++)
    {
        h_in[i] = 1.0f;
    }

    cudaMemcpy(d_in, h_in, N * sizeof(float), cudaMemcpyHostToDevice);

    dim3 grid(gridSize);
    dim3 block(blockSize);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    myReduce<blockSize><<<grid, block>>>(d_in, part_out, N);
    // myReduce<blockSize><<<1, block>>>(part_out, d_out, gridSize);

    //检测核函数有无错误,但是会严重破坏CUDA程序的性能
    // CHECK_CUDA(cudaGetLastError());
    // CHECK_CUDA(cudaDeviceSynchronize());

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    cudaMemcpy(one_pass_out, part_out, 1 * sizeof(float), cudaMemcpyDeviceToHost);
    // cudaMemcpy(cuda_out, d_out, 1 * sizeof(float), cudaMemcpyDeviceToHost);

    // printf("Mem BW= %f (GB/sec)\n", 1 * (float)N / milliseconds / 1e6);

    *cuda_out = *one_pass_out;
    // std::cout << "cuda:" << *cuda_out << std::endl;
    // std::cout << "one_pass_out:" << *one_pass_out << std::endl;

    if (checkResult(h_in, cuda_out, N))
    {
        printf("The result is correct\n");
    }
    else
    {
        printf("The result is incorrect\n");
    }
    printf("myReduce's latency is : %f ms\n", milliseconds);

    free(h_in);
    cudaFree(d_in);
    cudaFree(part_out);
    cudaFree(d_out);
    free(cuda_out);

    return 0;
}
