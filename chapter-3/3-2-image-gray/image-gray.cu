#include "common.h"
#include<ctime>
#define CHANNELS 3
#define HIGHT 67
#define WIDTH 39
__global__ void colorToGrayScaleConvertionKernel(unsigned char* pin, unsigned char* pout, int width, int height) {
    int col = threadIdx.x+blockDim.x*blockIdx.x;
    int row = threadIdx.y+blockDim.y*blockIdx.y;

    if (col < width && row < height) {
        int grayOffset=row*width+col;
        int rgbOffset=grayOffset*CHANNELS;
        int r = pin[rgbOffset];
        int g = pin[rgbOffset+1];
        int b = pin[rgbOffset+2];
        pout[grayOffset] = r*0.21f + g*0.71f + b*0.07f;
    }

}

int main(){
    srand((unsigned)time(NULL)); 
    
    unsigned char* pin = (unsigned char*)malloc(sizeof(unsigned char)*HIGHT*WIDTH*CHANNELS);
    unsigned char* pout = (unsigned char*)malloc(sizeof(unsigned char)*HIGHT*WIDTH);

    // Fill pin with random data
    for(int i=0; i<HIGHT; i++) {
        for(int j=0; j<WIDTH; j++) {
            for(int k=0; k<CHANNELS; k++) {
                pin[i*WIDTH*CHANNELS+j*CHANNELS+k] = rand()%256;
            }
        }
    }
    
     unsigned char *devPin, *devPout;
     cudaMalloc((void**)&devPin, sizeof(unsigned char)*HIGHT*WIDTH*CHANNELS);
     cudaMalloc((void**)&devPout, sizeof(unsigned char)*HIGHT*WIDTH);

     // Copy data to device
     cudaMemcpy(devPin, pin, sizeof(unsigned char)*HIGHT*WIDTH*CHANNELS, cudaMemcpyHostToDevice);
    // Set block and grid sizes
    int blockSize = 16;
    int gridSizeX = (WIDTH+blockSize-1)/blockSize;
    int gridSizeY = (HIGHT+blockSize-1)/blockSize;
    dim3 grid(gridSizeX, gridSizeY);
    // Set number of threads per block
    dim3 block(blockSize, blockSize);

    // Launch kernel
    colorToGrayScaleConvertionKernel<<<grid,block>>>(devPin, devPout, WIDTH, HIGHT);
    cudaDeviceSynchronize();

    // Copy data from device to host
    cudaMemcpy(pout, devPout, sizeof(unsigned char)*HIGHT*WIDTH, cudaMemcpyDeviceToHost);
    // Free device memory
    cudaFree(devPin);
    cudaFree(devPout);

    // show first few pixels of the output image
    for(int i=0; i<5; i++) {
        for(int j=0; j<5; j++) {
            printf("%d ", pout[i*WIDTH+j]);
        }
        printf("\n");
    }

    // Free host memory
    free(pin);
    free(pout);

    return 0;

}