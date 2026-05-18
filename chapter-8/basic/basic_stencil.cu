#include<cuda.h>
#include<cstdio>
__global__ void stencil_kernel(float * in,float * out,unsigned int N){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j= blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;
    if(i>0 && i<N-1 && j>0 && j<N-1 && k>0 && k<N-1){
        out[i*N*N+j*N+k] = (in[(i-1)*N*N+j*N+k]+in[(i+1)*N*N+j*N+k]+
                            in[i*N*N+(j-1)*N+k]+in[i*N*N+(j+1)*N+k]+
                            in[i*N*N+j*N+(k-1)]+in[i*N*N+j*N+(k+1)])/6.0f;
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
    cudaMalloc(&d_in,N*N*N*sizeof(float));
    cudaMalloc(&d_out,N*N*N*sizeof(float));
    cudaMemcpy(d_in,h_in,N*N*N*sizeof(float),cudaMemcpyHostToDevice);
    dim3 blockSize(8,8,8);
    dim3 gridSize((N+blockSize.x-1)/blockSize.x,(N+blockSize.y-1)/blockSize.y,(N+blockSize.z-1)/blockSize.z);
    stencil_kernel<<<gridSize,blockSize>>>(d_in,d_out,N);
    cudaMemcpy(h_out,d_out,N*N*N*sizeof(float),cudaMemcpyDeviceToHost);

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
                if(abs(h_out[i*N*N+j*N+k]-h_out_host[i*N*N+j*N+k])>1e-5){
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