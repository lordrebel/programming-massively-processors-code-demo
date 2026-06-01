#include<cuda.h>
#include<cstdio>
#define OUT_TILE_DIM 8
#define IN_TILE_DIM (OUT_TILE_DIM + 2)

__global__ void stencil_kernel(float *in, float *out, unsigned int N) {
    int iStart = blockIdx.z * OUT_TILE_DIM;
    int j=blockIdx.y * OUT_TILE_DIM + threadIdx.y-1;
    int k=blockIdx.x * OUT_TILE_DIM + threadIdx.x-1;
    __shared__ float inPrev_s[IN_TILE_DIM][IN_TILE_DIM];
    __shared__ float inCurr_s[IN_TILE_DIM][IN_TILE_DIM];
    __shared__ float inNext_s[IN_TILE_DIM][IN_TILE_DIM];
    if(iStart-1>=0 && iStart-1<N && j>=0 && j<N && k>=0 && k<N){
        inPrev_s[threadIdx.y][threadIdx.x] = in[(iStart-1)*N*N+j*N+k];
    }
    if(iStart>=0 && iStart<N && j>=0 && j<N && k>=0 && k<N){
        inCurr_s[threadIdx.y][threadIdx.x] = in[iStart*N*N+j*N+k];
    }
    for(int i=iStart;i<iStart+OUT_TILE_DIM;i++){
      if(i+1>=0 && i+1<N && j>=0 && j<N && k>=0 && k<N){
          inNext_s[threadIdx.y][threadIdx.x] = in[(i+1)*N*N+j*N+k];
      }
      __syncthreads();
      if(i>=1 && i<N-1 && j>=1 && j<N-1 && k>=1 && k<N-1){
          if(threadIdx.x>0 && threadIdx.x<IN_TILE_DIM-1 && threadIdx.y>0 && threadIdx.y<IN_TILE_DIM-1){
              out[i*N*N+j*N+k] = (inPrev_s[threadIdx.y][threadIdx.x]+inNext_s[threadIdx.y][threadIdx.x]+
                                  inCurr_s[threadIdx.y-1][threadIdx.x]+inCurr_s[threadIdx.y+1][threadIdx.x]+
                                  inCurr_s[threadIdx.y][threadIdx.x-1]+inCurr_s[threadIdx.y][threadIdx.x+1])/6.0f;
          }
      }
      __syncthreads();
      inPrev_s[threadIdx.y][threadIdx.x] = inCurr_s[threadIdx.y][threadIdx.x];
      inCurr_s[threadIdx.y][threadIdx.x] = inNext_s[threadIdx.y][threadIdx.x];
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
    dim3 blockDim(IN_TILE_DIM,IN_TILE_DIM);
    dim3 gridDim((N+OUT_TILE_DIM-1)/OUT_TILE_DIM,(N+OUT_TILE_DIM-1)/OUT_TILE_DIM,(N+OUT_TILE_DIM-1)/OUT_TILE_DIM);
    stencil_kernel<<<gridDim,blockDim>>>(d_in,d_out,N);
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
    //check result
    for(int i=1;i<N-1;i++){
        for(int j=1;j<N-1;j++){
            for(int k=1;k<N-1;k++){
                if(fabs(h_out[i*N*N+j*N+k]-h_out_host[i*N*N+j*N+k])>1e-5){
                    printf("Mismatch at (%d,%d,%d): GPU=%f, CPU=%f\n",i,j,k,h_out[i*N*N+j*N+k],h_out_host[i*N*N+j*N+k]);
                }
            }
        }
    }
    //free memory
    free(h_in);
    free(h_out);
    free(h_out_host);
    cudaFree(d_in);
    cudaFree(d_out);
    return 0;
}