#include<cstdio>
#include <cuda.h>
#include <cmath>

#define OUT_TILE_DIM 8
#define IN_TILE_DIM (OUT_TILE_DIM + 2)

#define CUDA_CHECK(call)                                                      \
    do {                                                                        \
        cudaError_t err = (call);                                                 \
        if (err != cudaSuccess) {                                                 \
            printf("CUDA error at %s:%d: %s\\n", __FILE__, __LINE__,               \
                         cudaGetErrorString(err));                                        \
            return 1;                                                               \
        }                                                                         \
    } while (0)

__global__ void stencil_kernel(float *in, float *out, unsigned int N) {
  int i = blockIdx.z * OUT_TILE_DIM + threadIdx.z - 1;
  int j = blockIdx.y * OUT_TILE_DIM + threadIdx.y - 1;
  int k = blockIdx.x * OUT_TILE_DIM + threadIdx.x - 1;
  __shared__ float tile[IN_TILE_DIM][IN_TILE_DIM][IN_TILE_DIM];
  if (i >= 0 && i < N && j >= 0 && j < N && k >= 0 && k < N) {
    tile[threadIdx.z][threadIdx.y][threadIdx.x] = in[i * N * N + j * N + k];
  } else {
    tile[threadIdx.z][threadIdx.y][threadIdx.x] = 0.0f;
  }
  __syncthreads();
  if (i >= 1 && i < N - 1 && j >= 1 && j < N - 1 && k >= 1 && k < N - 1) {
    if (threadIdx.x > 0 && threadIdx.x < IN_TILE_DIM - 1 && threadIdx.y > 0 &&
        threadIdx.y < IN_TILE_DIM - 1 && threadIdx.z > 0 &&
        threadIdx.z < IN_TILE_DIM - 1) {
      out[i * N * N + j * N + k] =
          (tile[threadIdx.z - 1][threadIdx.y][threadIdx.x] +
           tile[threadIdx.z + 1][threadIdx.y][threadIdx.x] +
           tile[threadIdx.z][threadIdx.y - 1][threadIdx.x] +
           tile[threadIdx.z][threadIdx.y + 1][threadIdx.x] +
           tile[threadIdx.z][threadIdx.y][threadIdx.x - 1] +
           tile[threadIdx.z][threadIdx.y][threadIdx.x + 1]) /
          6.0f;
    }
  }
}

int main(){
    unsigned int N = 33;
    float *h_in = (float*)malloc(N*N*N*sizeof(float));
    float *h_out = (float*)malloc(N*N*N*sizeof(float));
    float *h_out_host = (float*)malloc(N*N*N*sizeof(float));
    for(int i=0;i<N;i++){
        for(int j=0;j<N;j++){
            for(int k=0;k<N;k++){
                h_in[i*N*N+j*N+k] = i+j+k;
            }
        }
    }
    float *d_in,*d_out;
    CUDA_CHECK(cudaMalloc(&d_in,N*N*N*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_out,N*N*N*sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_in,h_in,N*N*N*sizeof(float),cudaMemcpyHostToDevice));
    dim3 blockSize(OUT_TILE_DIM+2,OUT_TILE_DIM+2,OUT_TILE_DIM+2);
    dim3 gridSize((N+OUT_TILE_DIM-1)/OUT_TILE_DIM,(N+OUT_TILE_DIM-1)/OUT_TILE_DIM,(N+OUT_TILE_DIM-1)/OUT_TILE_DIM);
    stencil_kernel<<<gridSize,blockSize>>>(d_in,d_out,N);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(h_out,d_out,N*N*N*sizeof(float),cudaMemcpyDeviceToHost));

    //cpu result
    for(int i=1;i<N-1;i++){
        for(int j=1;j<N-1;j++){
            for(int k=1;k<N-1;k++){
                h_out_host[i*N*N+j*N+k] = (h_in[(i-1)*N*N+j*N+k]+h_in[(i+1)*N*N+j*N+k]+
                                            h_in[i*N*N+(j-1)*N+k]+h_in[i*N*N+(j+1)*N+k]+
                                            h_in[i*N*N+j*N+(k-1)]+h_in[i*N*N+j*N+(k+1)])/6.0f;
            }
        }
    }
    //compare results
    for(int i=1;i<N-1;i++){
        for(int j=1;j<N-1;j++){
            for(int k=1;k<N-1;k++){
                if(fabsf(h_out[i*N*N+j*N+k]-h_out_host[i*N*N+j*N+k])>1e-5f){
                    printf("Mismatch at (%d,%d,%d): GPU=%f, CPU=%f\n",i,j,k,h_out[i*N*N+j*N+k],h_out_host[i*N*N+j*N+k]);
                }
            }
        }
    }
    cudaFree(d_in);
    cudaFree(d_out);
    free(h_in);
    free(h_out);
    free(h_out_host);       
}