#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdio>

__device__ __forceinline__ void myAtomicMax(int * address, int val) {
    int old_value = *address, assumed_old_val;
    do {
        assumed_old_val = old_value;
        old_value = atomicCAS(address, assumed_old_val, max(val, assumed_old_val));
    } while(old_value != assumed_old_val);
}


// --------------------------------------
// Test kernels and host test harness
// --------------------------------------

#define CUDA_CHECK(cmd) \
    do { \
        cudaError_t e = (cmd); \
        if (e != cudaSuccess) { \
            std::fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(e), __FILE__, __LINE__); \
            std::exit(EXIT_FAILURE); \
        } \
    } while (0)

__global__ void test_atomicMax_kernel(int* global_max, const int* values, int count) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < count) {
        // Use the custom atomicMax defined above
        myAtomicMax(global_max, values[idx]);
    }
}


int main() {
    // --------------------
    // Test for atomicMax
    // --------------------
    const int num_values = 1024;
    int* h_values = (int*)std::malloc(num_values * sizeof(int));
    for (int i = 0; i < num_values; ++i) {
        // A simple varying pattern with a known maximum
        h_values[i] = (i * 37) % 10000 - 5000; // range roughly [-5000, 4999]
    }
    int expected_max = h_values[0];
    for (int i = 1; i < num_values; ++i) expected_max = expected_max < h_values[i] ? h_values[i] : expected_max;

    int *d_values = nullptr, *d_max = nullptr;
    CUDA_CHECK(cudaMalloc(&d_values, num_values * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_max, sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_values, h_values, num_values * sizeof(int), cudaMemcpyHostToDevice));
    int init_min = INT_MIN;
    CUDA_CHECK(cudaMemcpy(d_max, &init_min, sizeof(int), cudaMemcpyHostToDevice));

    dim3 block(256);
    dim3 grid((num_values + block.x - 1) / block.x);
    test_atomicMax_kernel<<<grid, block>>>(d_max, d_values, num_values);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    int got_max = 0;
    CUDA_CHECK(cudaMemcpy(&got_max, d_max, sizeof(int), cudaMemcpyDeviceToHost));

    bool pass_max = (got_max == expected_max);
    std::printf("atomicMax test: got=%d expected=%d %s\n", got_max, expected_max, pass_max ? "[PASS]" : "[FAIL]");

    CUDA_CHECK(cudaFree(d_values));
    CUDA_CHECK(cudaFree(d_max));
    std::free(h_values);


    return 0;
}
