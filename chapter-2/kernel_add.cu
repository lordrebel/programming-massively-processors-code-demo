#include"common.h"

__global__ void vectorAddKernel(float *A, float *B, float *C, int N){
    int curIdx=blockDim.x*blockIdx.x+threadIdx.x;
    int stride=blockDim.x*gridDim.x;
    for(int i=curIdx;i<N;i+=stride){
        C[i]=A[i]+B[i];
    }


}

void vectorAdd(float *A, float *B, float *C, int N){
    float *d_A,*d_B,*d_C;
    size_t size=N*sizeof(float);
    checkCudaErrors(cudaMalloc((void**)&d_A,size));
    checkCudaErrors(cudaMalloc((void**)&d_B,size));
    checkCudaErrors(cudaMalloc((void**)&d_C,size));  
    checkCudaErrors(cudaMemcpy(d_A,A,size,cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_B,B,size,cudaMemcpyHostToDevice));
    vectorAddKernel<<<ceil(N/256.0),256>>>(d_A,d_B,d_C,N);
    checkCudaErrors(cudaMemcpy(C,d_C,size,cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaFree(d_A));
    checkCudaErrors(cudaFree(d_B));
    checkCudaErrors(cudaFree(d_C));
  
}
int main(){
    int N=10000000;
    float *A=(float*)malloc(N*sizeof(float));
    float *B=(float*)malloc(N*sizeof(float));
    float *C=(float*)malloc(N*sizeof(float));
    for(int i=0;i<N;i++){
        A[i]=i+1;
        B[i]=i+1;
    }
    vectorAdd(A,B,C,N);
    for(int i=0;i<10;i++){
        printf("C[%d]=%f\n",i,C[i]);
    }
    free(A);
    free(B);
    free(C);
    return 0;

}