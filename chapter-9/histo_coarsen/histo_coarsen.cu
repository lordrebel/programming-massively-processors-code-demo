#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <cuda.h>

#define NUM_BINS (26 / 4 + 1)   // 7 bins, each covers 4 letters
#define CFACTOR 4               // each thread processes CFACTOR elements
#define BLOCK_SIZE 256

__global__ void histo_private_kernel(char* data, unsigned int length,
                                      unsigned int* histo) {
    // Initialize privatized bins in shared memory
    __shared__ unsigned int histo_s[NUM_BINS];
    for (unsigned int bin = threadIdx.x; bin < NUM_BINS; bin += blockDim.x) {
        histo_s[bin] = 0u;
    }
    __syncthreads();

    // Histogram: each thread processes CFACTOR consecutive elements
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    for (unsigned int i = tid * CFACTOR; i < min((tid + 1) * CFACTOR, length); ++i) {
        int alphabet_position = data[i] - 'a';
        if (alphabet_position >= 0 && alphabet_position < 26) {
            atomicAdd(&(histo_s[alphabet_position / 4]), 1);
        }
    }
    __syncthreads();

    // Commit shared-memory bins to global memory
    for (unsigned int bin = threadIdx.x; bin < NUM_BINS; bin += blockDim.x) {
        unsigned int binValue = histo_s[bin];
        if (binValue > 0) {
            atomicAdd(&(histo[bin]), binValue);
        }
    }
}

int main() {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> dist_int(0, 26);  // 0..26, 26 -> non-letter

    int length = 1 << 20;

    // Total threads needed so that every element is covered
    unsigned int total_threads = (length + CFACTOR - 1) / CFACTOR;
    int grid_size = (total_threads + BLOCK_SIZE - 1) / BLOCK_SIZE;

    char* data;
    unsigned int* hist;
    unsigned int* cpu_hist = new unsigned int[NUM_BINS]();

    cudaMallocManaged(&data, length * sizeof(char));
    cudaMallocManaged(&hist, NUM_BINS * sizeof(unsigned int));
    memset(hist, 0, NUM_BINS * sizeof(unsigned int));

    for (int i = 0; i < length; ++i) {
        data[i] = 'a' + dist_int(gen);
    }

    histo_private_kernel<<<grid_size, BLOCK_SIZE>>>(data, length, hist);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Kernel launch error: %s\n", cudaGetErrorString(err));
        return -1;
    }
    cudaDeviceSynchronize();

    // CPU reference histogram
    for (int i = 0; i < length; ++i) {
        int alphabet_pos = data[i] - 'a';
        if (alphabet_pos >= 0 && alphabet_pos < 26) {
            cpu_hist[alphabet_pos / 4]++;
        }
    }

    // Compare results
    bool match = true;
    for (int i = 0; i < NUM_BINS; ++i) {
        printf("hist[%d] = %u, cpu_hist[%d] = %u\n", i, hist[i], i, cpu_hist[i]);
        if (hist[i] != cpu_hist[i]) {
            match = false;
        }
    }
    printf(match ? "Results match!\n" : "Results do NOT match!\n");

    cudaFree(data);
    cudaFree(hist);
    delete[] cpu_hist;
    return 0;
}