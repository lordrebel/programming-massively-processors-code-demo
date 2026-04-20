#include <cstdio>
#include<cuda.h>
#define RADIUS 2
#define IN_TILED_DIM 32
#define OUT_TILED_DIM (IN_TILED_DIM - 2 * RADIUS)
__constant__ float d_filter[2 * RADIUS + 1][2 * RADIUS + 1];

__global__ void tiled_conv2d_kernel(float *input, float *output, int height,
                                int width) {
  __shared__ float tile[IN_TILED_DIM][IN_TILED_DIM];
  int col = blockIdx.x * OUT_TILED_DIM + threadIdx.x;
  int row = blockIdx.y * OUT_TILED_DIM + threadIdx.y;
  int inCol = col - RADIUS;
  int inRow = row - RADIUS;
  if (inCol >= 0 && inCol < width && inRow >= 0 && inRow < height) {
    tile[threadIdx.y][threadIdx.x] = input[inRow * width + inCol];
  } else {
    tile[threadIdx.y][threadIdx.x] = 0.0f;
  }
  __syncthreads();
  
  float sum = 0.0f;
  if (threadIdx.x < OUT_TILED_DIM && threadIdx.y < OUT_TILED_DIM &&
      col < width && row < height) {
    for (int i = -RADIUS; i <= RADIUS; i++) {
      for (int j = -RADIUS; j <= RADIUS; j++) {
        sum += tile[threadIdx.y + i + RADIUS][threadIdx.x + j + RADIUS] *
               d_filter[i + RADIUS][j + RADIUS];
      }
    }
    output[row * width + col] = sum;
  }
}

int main(){
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
    dim3 blockSize(IN_TILED_DIM, IN_TILED_DIM);
    dim3 gridSize((width + OUT_TILED_DIM - 1) / OUT_TILED_DIM, (height + OUT_TILED_DIM - 1) / OUT_TILED_DIM);
    tiled_conv2d_kernel<<<gridSize, blockSize>>>(d_input, d_output, height, width);
    cudaMemcpy(output, d_output, height * width * sizeof(float), cudaMemcpyDeviceToHost);
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
        printf("\n");
    }
    cudaFree(d_input);
    cudaFree(d_output);
    return 0;
}