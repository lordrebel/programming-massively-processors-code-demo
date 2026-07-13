#include <cuda.h>
#include <cstdio>
#include <random>

// 26 个字母，每 4 个字母打包到一个 bin，共 7 个 bin
#define NUM_BINS (26 / 4 + 1)

// 基于共享内存的直方图内核：
// 每个线程块在共享内存中维护一份私有直方图，块内线程通过原子操作更新共享内存，
// 计算完成后再将各块的局部直方图合并到全局内存。
__global__ void histo_shared_kernel(char* data, unsigned int length, unsigned int* histo) {
    // 在共享内存中声明私有直方图缓存（每个线程块一份）
    __shared__ unsigned int histo_s[NUM_BINS];

    // 协作式初始化。由于线程数可能和 bin 的数量不一致，使用跨步循环清零
    for (unsigned int bin = threadIdx.x; bin < NUM_BINS; bin += blockDim.x) {
        histo_s[bin] = 0u;
    }

    // 屏障同步，确保所有线程块内的线程都完成了共享内存的初始化
    __syncthreads();

    // 计算当前线程在全局数据中的索引
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    // 计算局部直方图
    if (i < length) {
        int alphabet_position = data[i] - 'a';
        if (alphabet_position >= 0 && alphabet_position < 26) {
            // 原子加法：此时操作的是低延迟的共享内存 histo_s，极大地减少了全局内存冲突
            atomicAdd(&(histo_s[alphabet_position / 4]), 1);
        }
    }

    // 屏障同步，确保本块内的所有线程都已将局部直方图计算完毕
    __syncthreads();

    // 将共享内存中的局部直方图结果，累加合并到全局内存的 histo 中
    for (unsigned int bin = threadIdx.x; bin < NUM_BINS; bin += blockDim.x) {
        unsigned int binValue = histo_s[bin];
        if (binValue > 0) {
            // 原子加法：将本块的累加值合并到最终的全局内存直方图
            atomicAdd(&(histo[bin]), binValue);
        }
    }
}

// CPU 端直方图计算，用于验证 GPU 结果
void cpu_histogram(char* data, unsigned int length, unsigned int* histo) {
    for (unsigned int i = 0; i < NUM_BINS; ++i) {
        histo[i] = 0u;
    }
    for (unsigned int i = 0; i < length; ++i) {
        int alphabet_position = data[i] - 'a';
        if (alphabet_position >= 0 && alphabet_position < 26) {
            histo[alphabet_position / 4]++;
        }
    }
}

int main() {
    // 随机数生成器
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> dist_int(0, 26);  // 0-26 包含越界测试

    int length = 1 << 20;  // 1M 数据点
    int blockSize = 256;
    int gridSize = (length + blockSize - 1) / blockSize;

    printf("=== Shared Memory Histogram ===\n");
    printf("Data length: %d, Grid: %d, Block: %d, Bins: %d\n", length, gridSize, blockSize, NUM_BINS);

    // 分配 Unified Memory
    char* data;
    unsigned int* histo;
    cudaMallocManaged(&data, length * sizeof(char));
    cudaMallocManaged(&histo, NUM_BINS * sizeof(unsigned int));

    // 初始化全局直方图为 0
    memset(histo, 0, NUM_BINS * sizeof(unsigned int));

    // 生成随机字母数据 (a-z)
    for (int i = 0; i < length; ++i) {
        data[i] = 'a' + dist_int(gen);
    }

    // 启动 GPU 内核
    printf("\nLaunching GPU kernel...\n");
    histo_shared_kernel<<<gridSize, blockSize>>>(data, length, histo);
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("CUDA Error: %s\n", cudaGetErrorString(err));
        cudaFree(data);
        cudaFree(histo);
        return -1;
    }
    printf("GPU kernel completed.\n");

    // CPU 端计算
    unsigned int* cpu_hist = new unsigned int[NUM_BINS];
    cpu_histogram(data, length, cpu_hist);

    // 对比结果
    printf("\n--- Results Comparison ---\n");
    printf("Bin | GPU      | CPU      | Match\n");
    printf("----+----------+----------+------\n");
    bool all_match = true;
    for (int i = 0; i < NUM_BINS; ++i) {
        bool match = (histo[i] == cpu_hist[i]);
        if (!match) all_match = false;
        printf("%3d | %8u | %8u | %s\n", i, histo[i], cpu_hist[i], match ? "OK" : "FAIL");
    }

    // 验证总数
    unsigned int gpu_total = 0, cpu_total = 0;
    for (int i = 0; i < NUM_BINS; ++i) {
        gpu_total += histo[i];
        cpu_total += cpu_hist[i];
    }
    printf("----+----------+----------+------\n");
    printf("Total: GPU=%u, CPU=%u, %s\n", gpu_total, cpu_total,
           (gpu_total == cpu_total) ? "OK" : "FAIL");

    if (all_match) {
        printf("\n✓ All bins match! GPU shared-memory histogram is correct.\n");
    } else {
        printf("\n✗ Mismatch detected!\n");
    }

    // 清理
    cudaFree(data);
    cudaFree(histo);
    delete[] cpu_hist;

    return all_match ? 0 : -1;
}
