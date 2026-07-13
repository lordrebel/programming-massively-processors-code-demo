#include<cuda.h>
#include <random>
#define NUM_BINS (26/4 + 1)

__global__ void histo_private_kernel(char * data, unsigned int* hist, unsigned int length) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < length) {
        int alphabet_pos = data[idx] - 'a';
        if(alphabet_pos >= 0 && alphabet_pos < 26)
            atomicAdd(&(hist[blockIdx.x*NUM_BINS + alphabet_pos/4]), 1);
    }
    __syncthreads();
    if(blockIdx.x>0){
        for(unsigned int bin=threadIdx.x; bin<NUM_BINS; bin+=blockDim.x){
            auto binValue=hist[blockIdx.x*NUM_BINS + bin];
            if(binValue>0){
                atomicAdd(&(hist[bin]), binValue);
            }
        }
    }
}

int main(){
    std::random_device rd;                   
    std::mt19937 gen(rd());                 

    std::uniform_int_distribution<int> dist_int(0, 26);
    
    int length = 1 << 20;
    int grid_size = (length + 255) / 256;
    char* data;
    unsigned int* hist;
    unsigned int*cpu_hist = new unsigned int[NUM_BINS]();
    cudaMallocManaged(&data, length * sizeof(char));
    cudaMallocManaged(&hist, grid_size * NUM_BINS * sizeof(unsigned int));
    memset(hist, 0, grid_size * NUM_BINS * sizeof(unsigned int));
    for (int i = 0; i < length; ++i) {
        data[i] = 'a' + dist_int(gen);
    }
    histo_private_kernel<<<(length + 255) / 256, 256>>>(data, hist, length);
    cudaDeviceSynchronize();
    for (int i = 0; i < length; ++i) {
        int alphabet_pos = data[i] - 'a';
        if(alphabet_pos >= 0 && alphabet_pos < 26)
            cpu_hist[alphabet_pos/4]++;
    }
    //compare results
    for (int i = 0; i < NUM_BINS; ++i) {
        printf("hist[%d] = %u, cpu_hist[%d] = %u\n", i, hist[i], i, cpu_hist[i]);
    }
    cudaFree(data);
    cudaFree(hist); 
    delete[] cpu_hist;
}