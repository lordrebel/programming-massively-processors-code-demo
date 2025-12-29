#include "common.h"
#include<ctime>
#define  BLURSIZE 4
#define KERNEL_SIZE 16
using uchar=unsigned char;


__global__ void blurKernel(uchar *d_input, uchar *d_output, int width, int height){
    int curRol=blockDim.y*blockIdx.y+threadIdx.y;
    int curCol=blockDim.x*blockIdx.x+threadIdx.x;
    if(curCol<width && curRol<height){
        int pixval=0;
        int numPixels=0;
        for(int i=-BLURSIZE;i<BLURSIZE+1;i++){
            for(int j=-BLURSIZE;j<BLURSIZE+1;j++){
                int x=curCol+j;
                int y=curRol+i;
                if(x>=0 && x<width && y>=0 && y<height){
                    pixval+=d_input[y*width+x];
                    numPixels++;
                }
            }
            d_output[curRol*width+curCol]=pixval/numPixels;
        }

    }
}
int main(){
    srand(time(0));
    // Create input and output arrays
    int width=1024;
    int height=768;
    size_t size=width*height*sizeof(uchar);
    uchar *h_input=new uchar[size];
    uchar *h_output=new uchar[size];
    uchar *d_input, *d_output;

    // Initialize input array with random values
    for(int i=0;i<size;i++){
        h_input[i]=rand()%256;
    }

    // Allocate device memory for input and output arrays
    cudaMalloc((void**)&d_input, size);
    cudaMalloc((void**)&d_output, size);

    // Copy input array to device
    cudaMemcpy(d_input, h_input, size, cudaMemcpyHostToDevice);
    // Create a CUDA kernel
    dim3 blockDim(KERNEL_SIZE, KERNEL_SIZE);
    dim3 gridDim((width+KERNEL_SIZE-1)/blockDim.x, (height+KERNEL_SIZE-1)/blockDim.y);
    blurKernel<<<gridDim, blockDim>>>(d_input, d_output, width, height);

    // Copy output array from device to host
    cudaMemcpy(h_output, d_output, size, cudaMemcpyDeviceToHost);
    // Print output array
    for(int i=0;i<10;i++){
        for(int j=0;j<10;j++){
            printf("%d ", h_output[i*width+j]);
        }
        printf("\n");
    }

    // Free device memory
    cudaFree(d_input);
    cudaFree(d_output);

    // Free host memory
    delete[] h_input;
    
}