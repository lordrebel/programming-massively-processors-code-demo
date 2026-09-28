#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <random>
#include <algorithm>

// Number of threads per block. Each block scans a section of
// SECTION_SIZE = 2 * blockDim.x elements (two elements per thread).
const int SECTION_SIZE = 1024;
// Number of blocks in the grid.
const int GRID_DIM = 4;

// Brent-Kung (work-efficient) inclusive scan kernel.
// Each block scans its own SECTION_SIZE-element section in shared memory.
// Phase 1 (up-sweep / reduction): halving strides build partial sums so
//   that XY[SECTION_SIZE-1] holds the total of the section.
// Phase 2 (down-sweep): shrinking strides propagate the partial sums down
//   so every slot ends up holding the inclusive prefix sum of its section.
// Only O(N) adds are performed in total (work-efficient), at the cost of
// 2*log2(N) steps instead of log2(N) (higher latency than Kogge-Stone).
__global__ void Brent_Kung_scan_kernel(float *X, float *Y, unsigned int N) {
    __shared__ float XY[SECTION_SIZE];
    unsigned int i = 2*blockIdx.x*blockDim.x + threadIdx.x;
    if (i < N) XY[threadIdx.x] = X[i];
    if (i + blockDim.x < N) XY[threadIdx.x + blockDim.x] = X[i + blockDim.x];
    for (unsigned int stride = 1; stride <= blockDim.x; stride *= 2) {
        __syncthreads();
        unsigned int index = (threadIdx.x + 1)*2*stride - 1;
        if (index < SECTION_SIZE) {
            XY[index] += XY[index - stride];
        }
    }
    for (int stride = SECTION_SIZE/4; stride > 0; stride /= 2) {
        __syncthreads();
        unsigned int index = (threadIdx.x + 1)*stride*2 - 1;
        if (index + stride < SECTION_SIZE) {
            XY[index + stride] += XY[index];
        }
    }
    __syncthreads();
    if (i < N) Y[i] = XY[threadIdx.x];
    if (i + blockDim.x < N) Y[i + blockDim.x] = XY[threadIdx.x + blockDim.x];
}

int main() {
    // Total elements: GRID_DIM sections of SECTION_SIZE elements each.
    // N is an exact multiple of SECTION_SIZE so no padding is needed.
    const int N = GRID_DIM * SECTION_SIZE;  // 4096
    const int BLOCK_SIZE = SECTION_SIZE / 2;  // 512 threads per block

    float *h_input = (float *)malloc(N * sizeof(float));
    float *h_gpu = (float *)malloc(N * sizeof(float));
    float *h_ref = (float *)malloc(N * sizeof(float));
    float *d_input = nullptr;
    float *d_output = nullptr;

    // Initialize host data with random floats in [0, 1).
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    for (int i = 0; i < N; ++i) {
        h_input[i] = dist(gen);
    }

    // CPU reference: inclusive scan within each section (each block scans
    // its own section independently, so the reference must match that).
    for (int sec = 0; sec < GRID_DIM; ++sec) {
        float acc = 0.0f;
        for (int t = 0; t < SECTION_SIZE; ++t) {
            acc += h_input[sec * SECTION_SIZE + t];
            h_ref[sec * SECTION_SIZE + t] = acc;
        }
    }

    // Allocate device memory and copy input.
    cudaMalloc((void **)&d_input, N * sizeof(float));
    cudaMalloc((void **)&d_output, N * sizeof(float));
    cudaMemcpy(d_input, h_input, N * sizeof(float), cudaMemcpyHostToDevice);

    // Launch kernel with GRID_DIM blocks of BLOCK_SIZE threads.
    Brent_Kung_scan_kernel<<<GRID_DIM, BLOCK_SIZE>>>(d_input, d_output, N);

    // Check for launch errors and synchronize.
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Kernel launch error: %s\n", cudaGetErrorString(err));
        cudaFree(d_input);
        cudaFree(d_output);
        free(h_input);
        free(h_gpu);
        free(h_ref);
        return -1;
    }
    cudaDeviceSynchronize();

    // Copy result back.
    cudaMemcpy(h_gpu, d_output, N * sizeof(float), cudaMemcpyDeviceToHost);

    // Compare results element-wise.
    float max_abs_err = 0.0f;
    float max_val = 0.0f;
    for (int i = 0; i < N; ++i) {
        max_abs_err = std::max(max_abs_err, fabsf(h_gpu[i] - h_ref[i]));
        max_val = std::max(max_val, fabsf(h_ref[i]));
    }
    float rel_err = max_abs_err / max_val;

    printf("N = %d (grid = %d blocks x %d threads, section = %d)\n",
           N, GRID_DIM, BLOCK_SIZE, SECTION_SIZE);
    printf("Max abs err = %.6e\n", max_abs_err);
    printf("Rel err     = %.6e\n", rel_err);

    // Show a few prefix sums from the first section for a visual check.
    printf("First 8 prefix sums (GPU vs CPU):\n");
    for (int i = 0; i < 8; ++i) {
        printf("  Y[%d] = %10.4f  ref = %10.4f\n", i, h_gpu[i], h_ref[i]);
    }

    if (rel_err < 1e-5f) {
        printf("Result: MATCH\n");
    } else {
        printf("Result: MISMATCH\n");
    }

    cudaFree(d_input);
    cudaFree(d_output);
    free(h_input);
    free(h_gpu);
    free(h_ref);
    return 0;
}
