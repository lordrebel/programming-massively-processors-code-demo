#include<cstdio>
#include<cuda.h>

__global__ void conv2d_basc_kernel(float * input,float *filter,float*output,
                                    int height,int width,int radius){
int outCol=blockDim.x*blockIdx.x+threadIdx.x;
int outRow=blockDim.y*blockIdx.y+threadIdx.y;
float sum=0.0f;
if(outCol<width && outRow<height){
    for(int i=-radius;i<=radius;i++){
        for(int j=-radius;j<=radius;j++){
            int inRow=outRow+i;
            int inCol=outCol+j;
            if(inRow>=0 && inRow<height && inCol>=0 && inCol<width){
                sum+=input[inRow*width+inCol]*filter[(i+radius)*((int)2*radius+1)+(j+radius)];
            }
        }
    }
    output[outRow*width+outCol]=sum;
}
}

int main(){
    int height=5;
    int width=5;
    int radius=1;
    float input[25]={1,2,3,4,5,
                     6,7,8,9,10,
                     11,12,13,14,15,
                     16,17,18,19,20,
                     21,22,23,24,25};
    float filter[9]={0.111f,0.111f,0.111f,
                     0.111f,0.111f,0.111f,
                     0.111f,0.111f,0.111f};
    float output[25]={0};

    float *d_input,*d_filter,*d_output;
    cudaMalloc(&d_input,height*width*sizeof(float));
    cudaMalloc(&d_filter,(2*radius+1)*(2*radius+1)*sizeof(float));
    cudaMalloc(&d_output,height*width*sizeof(float));

    cudaMemcpy(d_input,input,height*width*sizeof(float),cudaMemcpyHostToDevice);
    cudaMemcpy(d_filter,filter,(2*radius+1)*(2*radius+1)*sizeof(float),cudaMemcpyHostToDevice);

    dim3 blockSize(16,16);
    dim3 gridSize((width+blockSize.x-1)/blockSize.x,(height+blockSize.y-1)/blockSize.y);
    conv2d_basc_kernel<<<gridSize,blockSize>>>(d_input,d_filter,d_output,height,width,radius);

    cudaMemcpy(output,d_output,height*width*sizeof(float),cudaMemcpyDeviceToHost);
    //compare with CPU result
    for(int i=0;i<height;i++){
        for(int j=0;j<width;j++){
            float sum=0.0f;
            for(int m=-radius;m<=radius;m++){
                for(int n=-radius;n<=radius;n++){
                    int inRow=i+m;
                    int inCol=j+n;
                    if(inRow>=0 && inRow<height && inCol>=0 && inCol<width){
                        sum+=input[inRow*width+inCol]*filter[(m+radius)*((int)2*radius+1)+(n+radius)];
                    }
                }
            }
            printf("CPU: %f, GPU: %f\n",sum,output[i*width+j]);
        }
    }
    cudaFree(d_input);
    cudaFree(d_filter);
    cudaFree(d_output);

    return 0;
}