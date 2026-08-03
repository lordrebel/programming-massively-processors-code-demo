#include <cuda.h>
#include <cstdio>
#include <cstring>
#include <random>

// 26 letters, packed 4 letters per bin -> 7 bins total
#define NUM_BINS (26 / 4 + 1)

// Thread-local aggregation (RLE) based histogram kernel:
// Each thread keeps a register accumulator and locally accumulates elements
// that map to the same bin. It only commits the accumulated value to the
// shared-memory histogram with a single atomic add when the bin changes.
// This significantly reduces the number of atomic operations and alleviates
// shared-memory access conflicts.
__global__ void histo_private_kernel(
    char* data,
    unsigned int length,
    unsigned int* histo)
{
    // One private histogram per block
    __shared__ unsigned int histo_s[NUM_BINS];

    // -----------------------------
    // 1. Initialize the shared histogram
    // -----------------------------
    for (unsigned int bin = threadIdx.x;
         bin < NUM_BINS;
         bin += blockDim.x)
    {
        histo_s[bin] = 0;
    }

    __syncthreads();

    // -----------------------------
    // 2. Each thread processes its own data
    // -----------------------------
    unsigned int accumulator = 0;
    int prevBinIdx = -1;

    unsigned int tid =
        blockIdx.x * blockDim.x + threadIdx.x;

    // grid-stride loop
    for (unsigned int i = tid;
         i < length;
         i += blockDim.x * gridDim.x)
    {
        int alphabet_position =
            data[i] - 'a';

        if (alphabet_position >= 0 &&
            alphabet_position < 26)
        {
            int bin =
                alphabet_position / 4;

            /*
             * Thread-local RLE
             *
             * If the data a thread keeps touching falls into the
             * same bin, do not issue an atomic immediately.
             */
            if (bin == prevBinIdx)
            {
                ++accumulator;
            }
            else
            {
                // Commit the previously accumulated result
                if (accumulator > 0)
                {
                    atomicAdd(
                        &histo_s[prevBinIdx],
                        accumulator);
                }

                accumulator = 1;
                prevBinIdx = bin;
            }
        }
    }

    // Commit the final accumulated run
    if (accumulator > 0)
    {
        atomicAdd(
            &histo_s[prevBinIdx],
            accumulator);
    }

    __syncthreads();

    // -----------------------------
    // 3. Merge the block histogram into global memory
    // -----------------------------
    for (unsigned int bin = threadIdx.x;
         bin < NUM_BINS;
         bin += blockDim.x)
    {
        unsigned int binValue =
            histo_s[bin];

        if (binValue > 0)
        {
            atomicAdd(
                &histo[bin],
                binValue);
        }
    }
}

// CPU-side histogram computation, used to validate the GPU result
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

    printf("=== Private Aggregate (RLE) Histogram ===\n");
    printf("Data length: %d, Grid: %d, Block: %d, Bins: %d\n",
           length, gridSize, blockSize, NUM_BINS);

    // Allocate Unified Memory
    char* data;
    unsigned int* histo;
    cudaMallocManaged(&data, length * sizeof(char));
    cudaMallocManaged(&histo, NUM_BINS * sizeof(unsigned int));

    // Initialize the global histogram to zero
    memset(histo, 0, NUM_BINS * sizeof(unsigned int));

    // Generate random letter data (a-z); the 0-26 range yields a few
    // out-of-range characters to test robustness
    for (int i = 0; i < length; ++i) {
        data[i] = 'a' + dist_int(gen);
    }

    // Launch the GPU kernel
    printf("\nLaunching GPU kernel (thread-local aggregation)...\n");
    histo_private_kernel<<<gridSize, blockSize>>>(data, length, histo);
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("CUDA Error: %s\n", cudaGetErrorString(err));
        cudaFree(data);
        cudaFree(histo);
        return -1;
    }
    printf("GPU kernel completed.\n");

    // CPU-side computation
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
        printf("\n✓ All bins match! GPU aggregate histogram is correct.\n");
    } else {
        printf("\n✗ Mismatch detected!\n");
    }

    // Cleanup
    cudaFree(data);
    cudaFree(histo);
    delete[] cpu_hist;

    return all_match ? 0 : -1;
}
