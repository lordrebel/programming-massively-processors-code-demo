#include <cuda.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <random>

// Convergent (pair-wise) sum reduction kernel.
// At each step the active thread count halves: threads in [0, stride)
// add the element at distance `stride` into their own slot. This is a
// "convergent" pattern because the set of participating threads shrinks
// toward thread 0, improving warp convergence compared with a strided
// interleaved approach where active threads become scattered.
__global__ void ConvergentSumReductionKernel(float *input, float *output) {
    unsigned int i = threadIdx.x;

    for (unsigned int stride = blockDim.x; stride >= 1; stride /= 2) {
        if (threadIdx.x < stride) {
            input[i] += input[i + stride];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        *output = input[0];
    }
}

int main() {
    // Number of elements must be 2 * blockDim.x. Use 1024 threads -> 2048 elements.
    const int N = 1 << 11;          // 2048
    const int BLOCK_SIZE = N / 2;   // 1024 threads

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

    // Launch kernel with a single block of BLOCK_SIZE threads.
    ConvergentSumReductionKernel<<<1, BLOCK_SIZE>>>(d_input, d_output);

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
    printf("N = %d\n", N);
    printf("GPU sum  = %.6f\n", gpu_sum);
    printf("CPU sum  = %.6f\n", cpu_sum);
    printf("Diff     = %.6f\n", fabsf(gpu_sum - cpu_sum));

    // Use a relative tolerance: floating-point summation order differs
    // between GPU (tree reduction) and CPU (sequential), so a small
    // absolute difference is expected and grows with N and magnitude.
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
