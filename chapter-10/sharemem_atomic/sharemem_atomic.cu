#include <cuda.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <random>

// Number of threads per block. Each block reduces a segment of
// 2 * BLOCK_DIM elements, so N must be a multiple of 2 * BLOCK_DIM.
const int BLOCK_DIM = 1024;

// Segmented shared-memory sum reduction kernel.
// Each block handles its own segment of 2 * BLOCK_DIM elements:
//   segment = 2 * blockDim.x * blockIdx.x
// Each thread loads two elements (input[i] + input[i + BLOCK_DIM]) into
// shared memory, performs a convergent tree reduction inside shared memory,
// then thread 0 atomically adds the block's partial sum to the global output.
__global__ void SegmentedSumReductionKernel(float *input, float *output) {
    __shared__ float input_s[BLOCK_DIM];
    unsigned int segment = 2 * blockDim.x * blockIdx.x;
    unsigned int i = segment + threadIdx.x;
    unsigned int t = threadIdx.x;

    // Load two elements per thread into shared memory.
    input_s[t] = input[i] + input[i + BLOCK_DIM];

    // Convergent tree reduction in shared memory.
    for (unsigned int stride = blockDim.x / 2; stride >= 1; stride /= 2) {
        __syncthreads();
        if (t < stride) {
            input_s[t] += input_s[t + stride];
        }
    }

    // Accumulate this block's partial sum into the global output.
    if (t == 0) {
        atomicAdd(output, input_s[0]);
    }
}

int main() {
    // N must be a multiple of 2 * BLOCK_DIM. Use 8 blocks -> 16384 elements.
    const int N = 1 << 14;              // 16384
    const int BLOCKS = N / (2 * BLOCK_DIM);  // 8 blocks

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

    // The output must be zeroed before launch because atomicAdd accumulates.
    cudaMemset(d_output, 0, sizeof(float));

    // Launch kernel with BLOCKS blocks of BLOCK_DIM threads.
    SegmentedSumReductionKernel<<<BLOCKS, BLOCK_DIM>>>(d_input, d_output);

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
    printf("N = %d, blocks = %d, threads/block = %d\n", N, BLOCKS, BLOCK_DIM);
    printf("GPU sum  = %.6f\n", gpu_sum);
    printf("CPU sum  = %.6f\n", cpu_sum);
    printf("Diff     = %.6f\n", fabsf(gpu_sum - cpu_sum));

    // Relative tolerance: GPU (segmented tree + atomic) vs CPU (sequential)
    // summation order differs, so a small absolute difference is expected.
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