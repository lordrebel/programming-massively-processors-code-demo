#include<cuda.h>
#include<cstdio>
#include<random>

__global__ void histogram_kernel(char * data, unsigned int* hist, unsigned int length) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < length) {
        int alphabet_pos = data[idx] - 'a';
        if(alphabet_pos >= 0 && alphabet_pos < 26)
            atomicAdd(&hist[alphabet_pos/4], 1);
    }
}

int main(){
    std::random_device rd;                   
    std::mt19937 gen(rd());                 

    std::uniform_int_distribution<int> dist_int(0, 26);
    
    int length = 1 << 20;
    char* data;
    unsigned int* hist;
    unsigned int*cpu_hist = new unsigned int[26/4]();
    memset(cpu_hist, 0, 26/4 * sizeof(unsigned int));
    cudaMallocManaged(&data, length * sizeof(char));
    cudaMallocManaged(&hist, 26/4 * sizeof(unsigned int));
    for (int i = 0; i < length; ++i) {
        data[i] = 'a' + dist_int(gen);
    }
    histogram_kernel<<<(length + 255) / 256, 256>>>(data, hist, length);
    cudaDeviceSynchronize();
    for (int i = 0; i < length; ++i) {
        int alphabet_pos = data[i] - 'a';
        if(alphabet_pos >= 0 && alphabet_pos < 26)
            cpu_hist[alphabet_pos/4]++;
    }
    //compare results
    for (int i = 0; i < 26/4; ++i) {
        printf("hist[%d] = %u, cpu_hist[%d] = %u\n", i, hist[i], i, cpu_hist[i]);
    }
    cudaFree(data);
    cudaFree(hist); 
    delete[] cpu_hist;
}