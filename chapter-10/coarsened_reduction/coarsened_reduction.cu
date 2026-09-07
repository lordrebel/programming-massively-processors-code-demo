#include <cuda.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <random>

// Number of threads per block.
const int BLOCK_DIM = 256;
// Each thread accumulates COARSE_FACTOR*2 input elements before the
// in-block tree reduction, so each block covers COARSE_FACTOR*2*BLOCK_DIM
// contiguous elements (its "segment").
const int COARSE_FACTOR = 8;
// Number of blocks in the grid.
const int GRID_DIM = 32;

// Coarsened sum reduction kernel.
// Phase 1 (coarsening): each thread sequentially accumulates
// COARSE_FACTOR*2 elements that are BLOCK_DIM apart, starting from
// segment + threadIdx.x. This amortizes the reduction cost over more
// data per thread and reduces the number of blocks (and therefore the
// number of atomicAdd operations on the output).
// Phase 2: convergent tree reduction in shared memory.
// Phase 3: thread 0 of each block atomically adds the block's partial
// sum into the global output.
__global__ void CoarsenedReductionKernel(float *input, float *output) {
    __shared__ float input_s[BLOCK_DIM];
    unsigned int segment = COARSE_FACTOR * 2 * blockDim.x * blockIdx.x;
    unsigned int i = segment + threadIdx.x;
    unsigned int t = threadIdx.x;

    // Phase 1: coarsened sequential accumulation into a register.
    float sum = input[i];
    for (unsigned int tile = 1; tile < COARSE_FACTOR * 2; ++tile) {
        sum += input[i + tile * BLOCK_DIM];
    }
    input_s[t] = sum;

    // Phase 2: convergent tree reduction in shared memory.
    for (unsigned int stride = blockDim.x / 2; stride >= 1; stride /= 2) {
        __syncthreads();
        if (t < stride) {
            input_s[t] += input_s[t + stride];
        }
    }

    // Phase 3: one atomicAdd per block.
    if (t == 0) {
        atomicAdd(output, input_s[0]);
    }
}

int main() {
    // Total elements: one segment per block.
    const int N = GRID_DIM * COARSE_FACTOR * 2 * BLOCK_DIM;  // 131072

    float *h_input = (float *)malloc(N * sizeof(float));
    float *d_input = nullptr;
    float *d_output = nullptr;

    // Initialize host data with random floats in [0, 1).
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    for (int i = 0; i < N; ++i) {
        h_input[i] = dist(gen);
    }

    // CPU reference sum.
    float cpu_sum = 0.0f;
    for (int i = 0; i < N; ++i) {
        cpu_sum += h_input[i];
    }

    // Allocate device memory and copy input.
    cudaMalloc((void **)&d_input, N * sizeof(float));
    cudaMalloc((void **)&d_output, sizeof(float));
    cudaMemcpy(d_input, h_input, N * sizeof(float), cudaMemcpyHostToDevice);

    // The kernel accumulates with atomicAdd, so the output must start at 0.
    cudaMemset(d_output, 0, sizeof(float));

    // Launch kernel with GRID_DIM blocks of BLOCK_DIM threads.
    CoarsenedReductionKernel<<<GRID_DIM, BLOCK_DIM>>>(d_input, d_output);

    // Check for launch errors and synchronize.
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Kernel launch error: %s\n", cudaGetErrorString(err));
        cudaFree(d_input);
        cudaFree(d_output);
        free(h_input);
        return -1;
    }
    cudaDeviceSynchronize();

    // Copy result back.
    float gpu_sum = 0.0f;
    cudaMemcpy(&gpu_sum, d_output, sizeof(float), cudaMemcpyDeviceToHost);

    // Compare results.
    printf("N = %d (grid = %d blocks x %d threads, coarse factor = %d)\n",
           N, GRID_DIM, BLOCK_DIM, COARSE_FACTOR);
    printf("GPU sum  = %.6f\n", gpu_sum);
    printf("CPU sum  = %.6f\n", cpu_sum);
    printf("Diff     = %.6f\n", fabsf(gpu_sum - cpu_sum));

    // Relative tolerance: GPU (coarsened + tree + atomic) vs CPU
    // (sequential) summation order differs, so a small difference is
    // expected and grows with N and magnitude.
    float rel_err = fabsf(gpu_sum - cpu_sum) / fabsf(cpu_sum);
    printf("Rel err  = %.6e\n", rel_err);

    if (rel_err < 1e-5f) {
        printf("Result: MATCH\n");
    } else {
        printf("Result: MISMATCH\n");
    }

    cudaFree(d_input);
    cudaFree(d_output);
    free(h_input);
    return 0;
}
