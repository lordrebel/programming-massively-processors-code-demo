#include <cuda.h>
#include<cstdio>
#define RADIUS 2
__constant__ float d_filter[2 * RADIUS + 1][2 * RADIUS + 1];

__global__ void conv2d_constant_kernel(float *input, float *output, int height,
                                       int width) {
  int outCol = blockDim.x * blockIdx.x + threadIdx.x;
  int outRow = blockDim.y * blockIdx.y + threadIdx.y;
  float sum = 0.0f;
  if (outCol < width && outRow < height) {
    for (int i = -RADIUS; i <= RADIUS; i++) {
      for (int j = -RADIUS; j <= RADIUS; j++) {
        int inRow = outRow + i;
        int inCol = outCol + j;
        if (inRow >= 0 && inRow < height && inCol >= 0 && inCol < width) {
          sum +=
              input[inRow * width + inCol] * d_filter[i + RADIUS][j + RADIUS];
        }
      }
    }
    output[outRow * width + outCol] = sum;
  }
}

int main() {

    int height = 5;
    int width = 5;
    float input[25] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11,
                        12, 13, 14, 15, 16, 17, 18, 19, 20, 21,
                        22, 23, 24, 25};
    float filter[25] = {0.04f, 0.04f, 0.04f, 0.04f,
                        0.04f, 0.04f, 0.04f, 0.04f,
                        0.04f, 0.04f,
                        0.04f, 0.04f,
                        0.04f};
    float output[25] = {0};
    
    float *d_input, *d_output;
    cudaMalloc(&d_input, height * width * sizeof(float));
    cudaMalloc(&d_output, height * width * sizeof(float));
    
    cudaMemcpy(d_input, input, height * width * sizeof(float),
                 cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(d_filter, filter,
                         (2 * RADIUS + 1) * (2 * RADIUS + 1) * sizeof(float));
    
    dim3 blockSize(16,16);
        dim3 gridSize((width+blockSize.x-1)/blockSize.x,(height+blockSize.y-1)/blockSize.y);
        conv2d_constant_kernel<<<gridSize,blockSize>>>(d_input,d_output,height,width);
    
        cudaMemcpy(output,d_output,height*width*sizeof(float),cudaMemcpyDeviceToHost);
         //compare with CPU result
    for(int i=0;i<height;i++){
        for(int j=0;j<width;j++){
            float sum=0.0f;
            for(int m=-RADIUS;m<=RADIUS;m++){
                for(int n=-RADIUS;n<=RADIUS;n++){
                    int inRow=i+m;
                    int inCol=j+n;
                    if(inRow>=0 && inRow<height && inCol>=0 && inCol<width){
                        sum+=input[inRow*width+inCol]*filter[(m+RADIUS)*((int)2*RADIUS+1)+(n+RADIUS)];
                    }
                }
            }
            printf("CPU: %f, GPU: %f\n",sum,output[i*width+j]);
        }
    }
    cudaFree(d_input);
    cudaFree(d_output);
}