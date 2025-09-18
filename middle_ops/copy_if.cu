#include <algorithm>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <float.h>
#include <stdio.h>
#include <stdlib.h>
#include <cstdint>

#define WARP_SIZE 32
#define FLOAT4(value) (reinterpret_cast<float4 *>(&(value))[0])
#define INT4(value) (reinterpret_cast<int4 *>(&(value))[0])
#define HALF2(value) (reinterpret_cast<half2 *>(&(value))[0])
#define BFLOAT2(value) (reinterpret_cast<__nv_bfloat162 *>(&(value))[0])
#define LOAD128BIT(value) (reinterpret_cast<float4 *>(&(value))[0])

// 测试结果发现block_copy_if和用了warp聚合的copy_if性能差不多，应该是编译器自动做了warp聚合

__device__ int atomicAggInc(int * sum);

__global__ void copy_if(int * res, int * dst, int N, int *sum) {
    int global_tid = threadIdx.x + blockIdx.x * blockDim.x;
    for (int i = global_tid; i < N; i += gridDim.x * blockDim.x) {
        int val = res[i];
        if (val > 0) {
            dst[atomicAggInc(sum)] = val;
        }
    }
}

__device__ int atomicAggInc(int * sum) {
    unsigned int active = __activemask();
    int leader = __ffs(active) - 1;
    int num = __popc(active);
    int lane_mask_lt;
    asm("mov.u32 %0, %%lanemask_lt;" : "=r"(lane_mask_lt));
    int rank = __popc(active & lane_mask_lt);
    int pos;
    if (rank == 0) {
        pos = atomicAdd(sum, num);
    }
    pos = __shfl_sync(active, pos, leader);
    return pos + rank;
}

__global__ void block_copy_if(int * res, int * dst, int N, int *sum) {
    int global_tid = threadIdx.x + blockIdx.x * blockDim.x;
    __shared__ int block_sum;
    
    for (int i = global_tid; i < N; i += gridDim.x * blockDim.x) {
        if (threadIdx.x == 0)
            block_sum = 0;
        __syncthreads();

        int val = res[i];
        int pos = 0;
        if (val > 0) {
            pos = atomicAdd(&block_sum, 1);
        }
        __syncthreads();
        
        if (threadIdx.x == 0)
            block_sum = atomicAdd(sum, block_sum);
        __syncthreads();
        
        if (val > 0)
            dst[block_sum + pos] = val;
        __syncthreads();
    }
}

bool CheckResult(int *out, int groudtruth, int n){
    if (*out != groudtruth) {
        return false;
    }
    return true;
}

int main(){
    float milliseconds = 0;
    int N = 256001230;

    cudaSetDevice(0);
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);
    const int blockSize = 256;
    int GridSize = std::min((N + 256 - 1) / 256, deviceProp.maxGridSize[0]);

    int *src_h = (int *)malloc(N * sizeof(int));
    int *dst_h = (int *)malloc(N * sizeof(int));
    int *nres_h = (int *)malloc(1 * sizeof(int));
    int *dst, *nres;
    int *src;
    cudaMalloc((void **)&src, N * sizeof(int));
    cudaMalloc((void **)&dst, N * sizeof(int));
    cudaMalloc((void **)&nres, 1 * sizeof(int));

    for(int i = 0; i < N; i++){
        src_h[i] = 1;
    }

    int groudtruth = 0;
    for(int j = 0; j < N; j++){
        if (src_h[j] > 0) {
            groudtruth += 1;
        }
    }

    cudaMemcpy(src, src_h, N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemset(nres, 0, sizeof(int));

    dim3 Grid(GridSize);
    dim3 Block(blockSize);
    
    copy_if<<<Grid, Block>>>(src, dst, N, nres);
    copy_if<<<Grid, Block>>>(src, dst, N, nres);
    copy_if<<<Grid, Block>>>(src, dst, N, nres);
    copy_if<<<Grid, Block>>>(src, dst, N, nres);
    copy_if<<<Grid, Block>>>(src, dst, N, nres);
    copy_if<<<Grid, Block>>>(src, dst, N, nres);
    cudaMemset(nres, 0, sizeof(int));


    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    copy_if<<<Grid, Block>>>(src, dst, N, nres);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    cudaMemcpy(nres_h, nres, 1 * sizeof(int), cudaMemcpyDeviceToHost);
    bool is_right = CheckResult(nres_h, groudtruth, N);
    if(is_right) {
        printf("the ans is right\n");
    } else {
        printf("the ans is wrong\n");
        printf("%d ",*nres_h);
        printf("\n");
    }
    printf("copy_if latency = %f ms\n", milliseconds);    

    // test block_copy_if
    cudaMemset(nres, 0, sizeof(int));
    milliseconds = 0.0f;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    block_copy_if<<<Grid, Block>>>(src, dst, N, nres);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    cudaMemcpy(nres_h, nres, 1 * sizeof(int), cudaMemcpyDeviceToHost);
    is_right = CheckResult(nres_h, groudtruth, N);
    if(is_right) {
        printf("the ans is right\n");
    } else {
        printf("the ans is wrong\n");
        printf("%d ",*nres_h);
        printf("\n");
    }
    printf("block_copy_if latency = %f ms\n", milliseconds);

    cudaFree(src);
    cudaFree(dst);
    cudaFree(nres);
    free(src_h);
    free(dst_h);
    free(nres_h);
}

