#include "common.h"
#include <ctime>
#define TILE_WIDTH 32
#define COARSE_FACTOR 4
//a shape:[m,k]  b shape:[k,n]  c(result) shape:[m,n]
__global__ void matmul_kernel(float *a,float *b, float *c,size_t M,size_t N,size_t K){
    __shared__ float tile_a[TILE_WIDTH][TILE_WIDTH];
    __shared__ float tile_b[TILE_WIDTH][TILE_WIDTH * COARSE_FACTOR];
    float sum[COARSE_FACTOR];
    int bx= blockIdx.x;
    int by= blockIdx.y;
    int tx= threadIdx.x;
    int ty= threadIdx.y;
    int row=by * TILE_WIDTH + ty;
    int colstart=bx * TILE_WIDTH * COARSE_FACTOR + tx;
    for(int i=0; i<COARSE_FACTOR; i++){
        sum[i]=0;
    }
    for(int ph=0; ph<(K+TILE_WIDTH-1)/TILE_WIDTH; ph++){
        if(row < M && ph*TILE_WIDTH + tx < K){
            tile_a[ty][tx] = a[row * K + ph*TILE_WIDTH + tx];
        }else{
            tile_a[ty][tx] = 0;
        }
        for(int i=0; i<COARSE_FACTOR; i++){
            if(ph*TILE_WIDTH + ty < K && colstart + i*TILE_WIDTH < N){
                tile_b[ty][tx + i*TILE_WIDTH] = b[(ph*TILE_WIDTH + ty) * N + colstart + i*TILE_WIDTH];
            }else{
                tile_b[ty][tx + i*TILE_WIDTH] = 0;
            }
        }
        __syncthreads();
        for(int k=0; k<TILE_WIDTH; k++){
            for(int i=0; i<COARSE_FACTOR; i++){
                sum[i] += tile_a[ty][k] * tile_b[k][tx + i*TILE_WIDTH];
            }
        }
        __syncthreads();
    }
    for(int i=0; i<COARSE_FACTOR; i++){
        if(row < M && colstart + i*TILE_WIDTH < N){
            c[row * N + colstart + i*TILE_WIDTH] = sum[i];
        }
    }
}

int main(){
    float *a, *dev_a, *b, *dev_b, *c, *dev_c;
    size_t M=132, N=79, K=67;
    a = (float*)malloc(M*K*sizeof(float));
    b = (float*)malloc(K*N*sizeof(float));
    c = (float*)malloc(M*N*sizeof(float));
    cudaMalloc((void**)&dev_a, M*K*sizeof(float));
    cudaMalloc((void**)&dev_b, K*N*sizeof(float));
    cudaMalloc((void**)&dev_c, M*N*sizeof(float));
    srand(time(NULL));
    for(size_t i = 0; i < M*K; i++){
        a[i] = rand() % 10;
    }
    for(size_t i = 0; i < K*N; i++){
        b[i] = rand() % 10;
    }
    cudaMemcpy(dev_a, a, M*K*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, b, K*N*sizeof(float), cudaMemcpyHostToDevice);
    dim3 dimBlock(TILE_WIDTH, TILE_WIDTH);
    dim3 dimGrid((N+dimBlock.x*COARSE_FACTOR-1)/(dimBlock.x*COARSE_FACTOR), (M+dimBlock.y-1)/dimBlock.y);
    matmul_kernel<<<dimGrid, dimBlock>>>(dev_a, dev_b, dev_c, M, N, K);
    cudaMemcpy(c, dev_c, M*N*sizeof(float), cudaMemcpyDeviceToHost);

    //test correctness
    bool correct = true;
    for(size_t i = 0; i < M*N; i++){
        size_t row = i / N;
        size_t col = i % N;
        float sum = 0;
        for(size_t k = 0; k < K; k++){
            sum += a[row * K + k] * b[k * N + col];
        }
        if(fabs(c[i] - sum) > 1e-3){
            correct = false;
            printf("Mismatch at row %zu col %zu: expected %f but got %f\n", row, col, sum, c[i]);
        }
    }
    if(correct){
        printf("Result is correct.\n");
    }
    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_c);
    free(a);
    free(b);
    free(c);
}