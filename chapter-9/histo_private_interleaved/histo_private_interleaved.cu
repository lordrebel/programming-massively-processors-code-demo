#include <cuda.h>
#include <cstdio>
#include <cstring>
#include <random>

// 26 letters, packed into bins of 4 letters each → 7 bins
#define NUM_BINS (26 / 4 + 1)

// Privatized histogram kernel using shared memory (interleaved assignment).
// Each thread block maintains its own private histogram in shared memory;
// threads within a block update it via atomic operations.
// Unlike histo_shared, this uses interleaved assignment:
// every thread strides through data with a step of blockDim.x * gridDim.x,
// rather than processing just one element. This improves hardware utilization
// and reduces workload imbalance across thread blocks.
__global__ void histo_private_kernel(char* data, unsigned int length,
                                      unsigned int* histo) {
    // Declare a private histogram cache in shared memory (one per block)
    __shared__ unsigned int histo_s[NUM_BINS];

    // Cooperatively initialize the shared-memory histogram
    for (unsigned int bin = threadIdx.x; bin < NUM_BINS; bin += blockDim.x) {
        histo_s[bin] = 0u;
    }
    __syncthreads();

    // Compute the starting global index for this thread
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;

    // Interleaved assignment: each thread processes elements spaced by
    // blockDim.x * gridDim.x
    for (unsigned int i = tid; i < length; i += blockDim.x * gridDim.x) {
        int alphabet_position = data[i] - 'a';
        if (alphabet_position >= 0 && alphabet_position < 26) {
            // Atomic add on low-latency shared memory histo_s
            atomicAdd(&(histo_s[alphabet_position / 4]), 1);
        }
    }
    __syncthreads();

    // Merge the local shared-memory histogram into global memory
    for (unsigned int bin = threadIdx.x; bin < NUM_BINS; bin += blockDim.x) {
        unsigned int binValue = histo_s[bin];
        if (binValue > 0) {
            // Atomic add to accumulate this block's values into the final global histogram
            atomicAdd(&(histo[bin]), binValue);
        }
    }
}

// CPU histogram implementation for validating GPU results
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
    // Random number generator
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> dist_int(0, 26);  // 0-26 includes out-of-range values for robustness testing

    int length = 1 << 20;  // 1M data points
    int blockSize = 256;
    int gridSize = (length + blockSize - 1) / blockSize;

    printf("=== Private Interleaved Histogram ===\n");
    printf("Data length: %d, Grid: %d, Block: %d, Bins: %d\n",
           length, gridSize, blockSize, NUM_BINS);

    // Allocate Unified Memory
    char* data;
    unsigned int* histo;
    cudaMallocManaged(&data, length * sizeof(char));
    cudaMallocManaged(&histo, NUM_BINS * sizeof(unsigned int));

    // Initialize global histogram to zero
    memset(histo, 0, NUM_BINS * sizeof(unsigned int));

    // Generate random letter data (a-z), dist_int range 0-26 produces a few
    // out-of-range values to test robustness
    for (int i = 0; i < length; ++i) {
        data[i] = 'a' + dist_int(gen);
    }

    // Launch GPU kernel
    printf("\nLaunching GPU kernel (interleaved assignment)...\n");
    histo_private_kernel<<<gridSize, blockSize>>>(data, length, histo);
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("CUDA Error: %s\n", cudaGetErrorString(err));
        cudaFree(data);
        cudaFree(histo);
        return -1;
    }
    printf("GPU kernel completed.\n");

    // CPU computation
    unsigned int* cpu_hist = new unsigned int[NUM_BINS];
    cpu_histogram(data, length, cpu_hist);

    // Compare results
    printf("\n--- Results Comparison ---\n");
    printf("Bin | GPU      | CPU      | Match\n");
    printf("----+----------+----------+------\n");
    bool all_match = true;
    for (int i = 0; i < NUM_BINS; ++i) {
        bool match = (histo[i] == cpu_hist[i]);
        if (!match) all_match = false;
        printf("%3d | %8u | %8u | %s\n", i, histo[i], cpu_hist[i],
               match ? "OK" : "FAIL");
    }

    // Verify total count
    unsigned int gpu_total = 0, cpu_total = 0;
    for (int i = 0; i < NUM_BINS; ++i) {
        gpu_total += histo[i];
        cpu_total += cpu_hist[i];
    }
    printf("----+----------+----------+------\n");
    printf("Total: GPU=%u, CPU=%u, %s\n", gpu_total, cpu_total,
           (gpu_total == cpu_total) ? "OK" : "FAIL");

    if (all_match) {
        printf("\n✓ All bins match! GPU private interleaved histogram is correct.\n");
    } else {
        printf("\n✗ Mismatch detected!\n");
    }

    // Cleanup
    cudaFree(data);
    cudaFree(histo);
    delete[] cpu_hist;

    return all_match ? 0 : -1;
}