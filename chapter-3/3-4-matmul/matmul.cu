#include "common.h"
#include "ctime"
//a shape:[m,k]  b shape:[k,n]  c(result) shape:[m,n]
__global__ void matmul_kernel(float *a,float *b, float *c,size_t M,size_t N,size_t K){
    size_t row = blockIdx.y * blockDim.y + threadIdx.y;
    size_t col = blockIdx.x * blockDim.x + threadIdx.x;

    if(row < M && col < N){
        float sum = 0;
        for(size_t i = 0; i < K; i++){
            sum += a[row * K + i] * b[i * N + col];
        }
        c[row * N + col] = sum;
    }

}

int main(){
    // Initialize input matrices
    float *a, *dev_a, *b, *dev_b, *c, *dev_c;
    size_t M=132, N=79, K=67;
    a = (float*)malloc(M*K*sizeof(float));
    b = (float*)malloc(K*N*sizeof(float));
    c = (float*)malloc(M*N*sizeof(float));

    cudaMalloc((void**)&dev_a, M*K*sizeof(float));
    cudaMalloc((void**)&dev_b, K*N*sizeof(float));
    cudaMalloc((void**)&dev_c, M*N*sizeof(float));

    // Initialize input matrices with random values
    srand(time(NULL));
    for(size_t i = 0; i < M*K; i++){
        a[i] = rand() % 10;
    }
    for(size_t i = 0; i < K*N; i++){
        b[i] = rand() % 10;
    }

    // Copy input matrices to device
    cudaMemcpy(dev_a, a, M*K*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, b, K*N*sizeof(float), cudaMemcpyHostToDevice);

    // Launch kernel
    dim3 dimBlock(16, 16);
    dim3 dimGrid((N+dimBlock.x-1)/dimBlock.x, (M+dimBlock.y-1)/dimBlock.y);
    matmul_kernel<<<dimGrid, dimBlock>>>(dev_a, dev_b, dev_c, M, N, K);

    // Copy result from device to host
    cudaMemcpy(c, dev_c, M*N*sizeof(float), cudaMemcpyDeviceToHost);

    // Free device memory
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
        for(size_t k = 0; k < K; k++){
            sum += a[row * K + k] * b[k * N + col];
        }
        if(std::abs(c[i] - sum) > 1e-6){
            correct = false;
            break;
        }
    }
    if(correct){
        printf("Result is correct.\n");
    } else {
        printf("Result is incorrect.\n");
    }

    // Clean up
    cudaDeviceReset();

    // Free host memory
    free(a);
    free(b);
    free(c);

    return 0;
   

}