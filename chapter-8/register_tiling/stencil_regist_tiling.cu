#include<cuda.h>
#include<cstdio>
#include<cmath>
#define OUT_TILE_DIM 16
#define IN_TILE_DIM (OUT_TILE_DIM + 2)
__global__ void stencil_kernel(float* in, float* out, unsigned int N) {
    int iStart = blockIdx.z * OUT_TILE_DIM;
    int j = blockIdx.y * OUT_TILE_DIM + threadIdx.y - 1;
    int k = blockIdx.x * OUT_TILE_DIM + threadIdx.x - 1;
    float inPrev;
    __shared__ float inCurr_s[IN_TILE_DIM][IN_TILE_DIM];
    float inCurr;
    float inNext;

    if (iStart - 1 >= 0 && iStart - 1 < N && j >= 0 && j < N && k >= 0 && k < N) {
        inPrev = in[(iStart - 1) * N * N + j * N + k];
    }

    if (iStart >= 0 && iStart < N && j >= 0 && j < N && k >= 0 && k < N) {
        inCurr = in[iStart * N * N + j * N + k];
        inCurr_s[threadIdx.y][threadIdx.x] = inCurr;
    }

    for (int i = iStart; i < iStart + OUT_TILE_DIM; ++i) {
        if (i + 1 >= 0 && i + 1 < N && j >= 0 && j < N && k >= 0 && k < N) {
            inNext = in[(i + 1) * N * N + j * N + k];
        }

        __syncthreads();

        if (i >= 1 && i < N - 1 && j >= 1 && j < N - 1 && k >= 1 && k < N - 1) {
            if (threadIdx.y >= 1 && threadIdx.y < IN_TILE_DIM - 1 
                && threadIdx.x >= 1 && threadIdx.x < IN_TILE_DIM - 1) {
                out[i * N * N + j * N + k] = (inCurr
                                           + inCurr_s[threadIdx.y][threadIdx.x - 1]
                                           + inCurr_s[threadIdx.y][threadIdx.x + 1]
                                           + inCurr_s[threadIdx.y + 1][threadIdx.x]
                                           + inCurr_s[threadIdx.y - 1][threadIdx.x]
                                           + inPrev
                                           + inNext)/7.0f;
            }
        }

        __syncthreads();
        inPrev = inCurr;
        inCurr = inNext;
        inCurr_s[threadIdx.y][threadIdx.x] = inNext; 
    }
}

int main(){
    unsigned int N = 64;
    float* in;
    float* out;
    cudaMallocManaged(&in, N * N * N * sizeof(float));
    cudaMallocManaged(&out, N * N * N * sizeof(float));

    for (unsigned int i = 0; i < N * N * N; ++i) {
        in[i] = static_cast<float>(i);
    }

    dim3 blockDim(IN_TILE_DIM, IN_TILE_DIM);
    dim3 gridDim((N + OUT_TILE_DIM - 1) / OUT_TILE_DIM, (N + OUT_TILE_DIM - 1) / OUT_TILE_DIM, (N + OUT_TILE_DIM - 1) / OUT_TILE_DIM);
    stencil_kernel<<<gridDim, blockDim>>>(in, out, N);
    cudaDeviceSynchronize();


    float maxError = 0.0f;
    unsigned int numInterior = 0;
    for (unsigned int i = 1; i < N - 1; ++i) {
        for (unsigned int j = 1; j < N - 1; ++j) {
            for (unsigned int k = 1; k < N - 1; ++k) {
                float expected = (in[i * N * N + j * N + k]
                               + in[i * N * N + j * N + k - 1]
                               + in[i * N * N + j * N + k + 1]
                               + in[i * N * N + (j - 1) * N + k]
                               + in[i * N * N + (j + 1) * N + k]
                               + in[(i - 1) * N * N + j * N + k]
                               + in[(i + 1) * N * N + j * N + k]) / 7.0f;
                float err = fabsf(out[i * N * N + j * N + k] - expected);
                if (err > maxError) maxError = err;
                ++numInterior;
            }
        }
    }
    printf("Max error: %e (interior points: %u)\n", maxError, numInterior);

    cudaFree(in);
    cudaFree(out);
    return 0;
}