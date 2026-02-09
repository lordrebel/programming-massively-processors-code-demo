#include "common.h"
#include "ctime"
#define TILE_WIDTH 16
//a shape:[m,k]  b shape:[k,n]  c(result) shape:[m,n]
__global__ void matmul_kernel(float *a,float *b, float *c,size_t M,size_t N,size_t K){
    __shared__ float tile_a[TILE_WIDTH][TILE_WIDTH];
    __shared__ float tile_b[TILE_WIDTH][TILE_WIDTH];
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int row = by * TILE_WIDTH + ty;
    int col = bx * TILE_WIDTH + tx;
    float sum = 0;
    for(int t = 0; t < (K+TILE_WIDTH-1)/TILE_WIDTH; t++){
        if(row < M && t*TILE_WIDTH + tx < K){
            tile_a[ty][tx] = a[row * K + t*TILE_WIDTH + tx];
        }else{
            tile_a[ty][tx] = 0;
        }
        if(t*TILE_WIDTH + ty < K && col < N){
            tile_b[ty][tx] = b[(t*TILE_WIDTH + ty) * N + col];
        }else{
            tile_b[ty][tx] = 0;
        }
        __syncthreads();
        for(int i = 0; i < TILE_WIDTH; i++){
            sum += tile_a[ty][i] * tile_b[i][tx];
        }
        __syncthreads();
    }
    if(row < M && col < N){
        c[row * N + col] = sum;
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
    dim3 dimGrid((N+dimBlock.x-1)/dimBlock.x, (M+dimBlock.y-1)/dimBlock.y);
    matmul_kernel<<<dimGrid, dimBlock>>>(dev_a, dev_b, dev_c, M, N, K);
    cudaMemcpy(c, dev_c, M*N*sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_c);
    for(size_t i = 0; i < 10; i++){
        printf("%f ", c[i]);
    }
    printf("\n");
    // Test correctness
    bool correct = true;
    for(size_t i = 0; i < M*N; i++){
        size_t row = i / N;  
        size_t col = i % N; 
        float sum = 0;
        for(size_t j = 0; j < K; j++){
            sum += a[row * K + j] * b[j * N + col];
        }
        if(fabs(c[i] - sum) > 1e-5){
            correct = false;
            printf("Mismatch at row %zu col %zu: expected %f but got %f\n", row, col, sum, c[i]);
            break;
        }
    }
    if(correct){
        printf("Result is correct!\n");
    }else{
        printf("Result is incorrect!\n");
    }
    free(a);
    free(b);
    free(c);
    return 0;
}