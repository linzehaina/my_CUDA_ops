#include <stdio.h>
#include <iostream>
#include <chrono>
#include <vector>
#include <cuda_runtime_api.h>
#include <algorithm>
#include <numeric>

// 示例kernel
__global__ void kernel_test1(float *result)
{
    unsigned int idx = threadIdx.x + blockDim.x * blockIdx.x;
    for (int i = 1234; i >= 0; --i)
    {
        if (idx == 0)
        {
            result[0] = i;
        }
    }

    uint64_t start = clock64(); // 获取当前时钟周期
    uint64_t delay = 50000000; // 延迟10000000个时钟周期

    while (clock64() - start < delay) {
        // 空循环，等待延迟时间结束
    }
}

__global__ void kernel_test2(float *result)
{
    unsigned int idx = threadIdx.x + blockDim.x * blockIdx.x;
    for (int i = 1234 * 2; i >= 0; --i)
    {
        if (idx == 0)
        {
            result[1] = i;
        }
    }

    uint64_t start = clock64(); // 获取当前时钟周期
    uint64_t delay = 100000000; // 延迟10000000个时钟周期

    while (clock64() - start < delay) {
        // 空循环，等待延迟时间结束
    }
}

std::vector<std::vector<cudaEvent_t>> initEvent(int m, int n)
{
    std::vector<std::vector<cudaEvent_t>> events(m, std::vector<cudaEvent_t>(n));
    for (int i = 0; i < m; i++)
    {
        for (int j = 0; j < n; j++)
        {
            cudaEventCreate(&events[i][j]);
        }
    }
    return events;
}

void syncEvent(const std::vector<std::vector<cudaEvent_t>> &events)
{
    for (auto &event_vec : events)
    {
        for (auto &event : event_vec)
        {
            cudaEventSynchronize(event);
        }
    }
}

std::vector<std::vector<float>> computeEventTime(const std::vector<std::vector<cudaEvent_t>> &events)
{
    std::vector<std::vector<float>> times(events.size() - 1, std::vector<float>(events[0].size()));
    for (int i = 0; i < events.size() - 1; i++)
    {
        for (int j = 0; j < events[i].size(); j++)
        {
            cudaEventElapsedTime(&times[i][j], events[i][j], events[i + 1][j]);
        }
    }
    return times;
}

void destoryEvent(std::vector<std::vector<cudaEvent_t>> &events)
{
    for (auto &event_vec : events)
    {
        for (auto &event : event_vec)
        {
            cudaEventDestroy(event);
        }
    }
}
void printTimeStatInfo(std::vector<float> data)
{
    std::sort(data.begin(), data.end());
    std::cout << "printTimeStatInfo:" << std::endl;
    std::cout << "Avg:" << std::accumulate(data.begin(), data.end(), 0.0) / (data.size()) << std::endl;
    std::cout << "Mid:" << data[data.size() / 2] << std::endl;
    std::cout << "Min:" << data[0] << std::endl;
    std::cout << "Max:" << data[data.size() - 1] << std::endl;
}

int main()
{
    const dim3 block_size(5, 6, 7);
    const dim3 grid_size(2, 3, 4);
    float *d_result;
    cudaMalloc((void **)&d_result, 2 * sizeof(float));

    // warm up
    for (int i = 0; i < 10; ++i)
    {
        kernel_test1<<<grid_size, block_size>>>(d_result);
        kernel_test2<<<grid_size, block_size>>>(d_result);
    }
    cudaDeviceSynchronize();

    // 初始化event
    auto events = initEvent(3, 100);

    // 对各kernel进行计时
    for (int i = 0; i < 100; ++i)
    {
      cudaEventRecord(events[0][i], nullptr);
      
      kernel_test1<<<grid_size, block_size>>>(d_result);
      
      cudaEventRecord(events[1][i], nullptr);
      
      kernel_test2<<<grid_size, block_size>>>(d_result);
      
      cudaEventRecord(events[2][i], nullptr);
    }

    // 等待各个event同步
    syncEvent(events);

    // 计算时间
    auto times = computeEventTime(events);

    // 打印时间统计信息
    printTimeStatInfo(times[0]);
    printTimeStatInfo(times[1]);

    // 销毁event
    destoryEvent(events);

    cudaFree(d_result);
    return 0;
}