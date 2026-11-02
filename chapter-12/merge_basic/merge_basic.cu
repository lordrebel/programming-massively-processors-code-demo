#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <algorithm>

// Block / grid configuration for the merge kernel.
const int BLOCK_SIZE = 128;
const int GRID_SIZE = 8;  // 1024 threads in total

// Sequential merge of two sorted sub-arrays A[0..m) and B[0..n) into C.
// Ties are broken in favor of A (A[i] <= B[j] takes from A first), which
// keeps the merge stable and must match the co_rank tie-breaking below.
__device__ void merge_sequential(int *A, int m, int *B, int n, int *C) {
    int i = 0, j = 0, k = 0;
    while (i < m && j < n) {
        if (A[i] <= B[j]) {
            C[k++] = A[i++];
        } else {
            C[k++] = B[j++];
        }
    }
    while (i < m) C[k++] = A[i++];
    while (j < n) C[k++] = B[j++];
}

// co_rank(k): find the cut point (i, j) with i + j = k such that
// A[0..i) and B[0..j) together form exactly the first k elements of the
// merged sequence. Returns i (then j = k - i).
//
// Validity conditions (ties broken in favor of A):
//   (i == 0 || j == n || A[i-1] <= B[j])  -- not too many elements from A
//   (j == 0 || i == m || B[j-1] <  A[i])  -- not too few elements from A
//
// Binary search over i in [max(0, k-n), min(k, m)]:
//   A[i-1] >  B[j]  ->  too many from A, decrease i
//   B[j-1] >= A[i]  ->  too few  from A, increase i
__device__ int co_rank(int k, int *A, int m, int *B, int n) {
    int i_low = max(0, k - n);   // j = k - i <= n  =>  i >= k - n
    int i_high = min(k, m);      // i <= k and i <= m
    while (i_low < i_high) {
        int i = i_low + (i_high - i_low + 1) / 2;  // upper-biased midpoint
        int j = k - i;
        if (i > 0 && j < n && A[i - 1] > B[j]) {
            i_high = i - 1;      // too many elements from A
        } else if (j > 0 && i < m && B[j - 1] >= A[i]) {
            i_low = i + 1;       // too few elements from A
        } else {
            return i;            // both co-rank conditions hold
        }
    }
    return i_low;
}

// Basic parallel merge kernel (PMPP ch.12): each thread is responsible for
// a contiguous chunk of the output C. A thread first uses co_rank to find
// the A/B input ranges that correspond to its output chunk, then merges
// those ranges sequentially. No thread ever touches another thread's
// output range, so no synchronization is needed.
__global__ void merge_basic_kernel(int *A, int m, int *B, int n, int *C) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int totalThreads = blockDim.x * gridDim.x;
    // ceil((m+n) / totalThreads) in integer arithmetic
    int elementsPerThread = (m + n + totalThreads - 1) / totalThreads;
    int k_curr = tid * elementsPerThread;                    // start output index
    int k_next = min((tid + 1) * elementsPerThread, m + n);  // end output index
    if (k_curr >= m + n) return;  // this thread has no output elements
    int i_curr = co_rank(k_curr, A, m, B, n);
    int i_next = co_rank(k_next, A, m, B, n);
    int j_curr = k_curr - i_curr;
    int j_next = k_next - i_next;
    merge_sequential(&A[i_curr], i_next - i_curr,
                     &B[j_curr], j_next - j_curr,
                     &C[k_curr]);
}

// CPU reference: plain sequential merge with the same tie-breaking rule.
static void merge_cpu(const int *A, int m, const int *B, int n, int *C) {
    int i = 0, j = 0, k = 0;
    while (i < m && j < n) {
        if (A[i] <= B[j]) C[k++] = A[i++];
        else              C[k++] = B[j++];
    }
    while (i < m) C[k++] = A[i++];
    while (j < n) C[k++] = B[j++];
}

int main() {
    // Non-equal, non-power-of-two sizes to exercise boundary cases.
    const int m = 1033;  // elements in A
    const int n = 777;   // elements in B
    const int total = m + n;

    int *h_A = (int *)malloc(m * sizeof(int));
    int *h_B = (int *)malloc(n * sizeof(int));
    int *h_C = (int *)malloc(total * sizeof(int));
    int *h_ref = (int *)malloc(total * sizeof(int));
    int *d_A = nullptr, *d_B = nullptr, *d_C = nullptr;

    // Generate two sorted arrays with overlapping value ranges so that
    // ties between A and B (and duplicates within each) actually occur,
    // which stresses the co_rank tie-breaking logic.
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> step(0, 5);
    int v = 0;
    for (int i = 0; i < m; ++i) { v += step(gen); h_A[i] = v; }
    v = 0;
    for (int i = 0; i < n; ++i) { v += step(gen); h_B[i] = v; }

    // CPU reference result.
    merge_cpu(h_A, m, h_B, n, h_ref);

    // Allocate device memory and copy inputs.
    cudaMalloc((void **)&d_A, m * sizeof(int));
    cudaMalloc((void **)&d_B, n * sizeof(int));
    cudaMalloc((void **)&d_C, total * sizeof(int));
    cudaMemcpy(d_A, h_A, m * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, n * sizeof(int), cudaMemcpyHostToDevice);

    // Launch kernel.
    merge_basic_kernel<<<GRID_SIZE, BLOCK_SIZE>>>(d_A, m, d_B, n, d_C);

    // Check for launch errors and synchronize.
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Kernel launch error: %s\n", cudaGetErrorString(err));
        cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
        free(h_A); free(h_B); free(h_C); free(h_ref);
        return -1;
    }
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("Kernel execution error: %s\n", cudaGetErrorString(err));
        cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
        free(h_A); free(h_B); free(h_C); free(h_ref);
        return -1;
    }

    // Copy result back (checked).
    err = cudaMemcpy(h_C, d_C, total * sizeof(int), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        printf("cudaMemcpy (D2H) error: %s\n", cudaGetErrorString(err));
        cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
        free(h_A); free(h_B); free(h_C); free(h_ref);
        return -1;
    }

    // Compare GPU result against the CPU reference element-wise.
    int mismatches = 0;
    int first_bad = -1;
    for (int i = 0; i < total; ++i) {
        if (h_C[i] != h_ref[i]) {
            ++mismatches;
            if (first_bad < 0) first_bad = i;
        }
    }

    printf("m = %d, n = %d, total = %d (grid = %d blocks x %d threads)\n",
           m, n, total, GRID_SIZE, BLOCK_SIZE);
    printf("First 10 merged elements (GPU vs CPU):\n");
    for (int i = 0; i < 10; ++i) {
        printf("  C[%d] = %6d  ref = %6d\n", i, h_C[i], h_ref[i]);
    }
    if (mismatches == 0) {
        printf("Result: MATCH (all %d elements identical)\n", total);
    } else {
        printf("Result: MISMATCH (%d elements differ, first at index %d)\n",
               mismatches, first_bad);
    }

    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    free(h_A); free(h_B); free(h_C); free(h_ref);
    return mismatches == 0 ? 0 : -1;
}
