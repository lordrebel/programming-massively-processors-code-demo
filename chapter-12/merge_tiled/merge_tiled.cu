#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <random>

// Block / grid / tile configuration for the tiled merge kernel.
// Constraint: each block merges a chunk of C of length
// ceil((m+n)/gridDim.x), so tile_size must be >= that chunk length
// (the whole A-part and B-part of the chunk are staged in shared memory).
const int BLOCK_SIZE = 128;
const int GRID_SIZE = 8;
const int TILE_SIZE = 512;  // >= ceil((m+n)/GRID_SIZE) for the test sizes below

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

// Tiled parallel merge kernel (PMPP ch.12).
//
// Phase 1: thread 0 computes the block's global co-rank values and publishes
//          them via shared memory so every thread knows the block's input
//          ranges without redoing the (expensive) binary search.
// Phase 2: the block cooperatively loads its A-tile and B-tile into shared
//          memory.
// Phase 3: each thread finds its personal cut points inside the tiles with
//          co_rank (now cheap, operating on shared memory) and merges its
//          sub-range sequentially into its own private section of C.
__global__ void merge_tiled_kernel(int* A, int m, int* B, int n, int* C, int tile_size) {
    /* shared memory allocation */
    extern __shared__ int shareAB[];
    int * A_S = &shareAB[0];                  // shareA is first half of shareAB
    int * B_S = &shareAB[tile_size];          // shareB is second half of shareAB
    int total = m + n;
    int chunk = (total + gridDim.x - 1) / gridDim.x;   // ceil((m+n)/gridDim.x)
    int C_curr = min(blockIdx.x * chunk, total);       // start point of block's C subarray
    int C_next = min((blockIdx.x + 1) * chunk, total); // ending point

    if (threadIdx.x == 0) {
        A_S[0] = co_rank(C_curr, A, m, B, n); // Make block-level co-rank values visible
        A_S[1] = co_rank(C_next, A, m, B, n); // to other threads in the block
    }
    __syncthreads();
    int A_curr = A_S[0];
    int A_next = A_S[1];
    int B_curr = C_curr - A_curr;
    int B_next = C_next - A_next;
    __syncthreads();

    /* Phase 2: cooperatively load this block's tiles of A and B */
    int A_length = A_next - A_curr;
    int B_length = B_next - B_curr;
    for (int i = threadIdx.x; i < A_length; i += blockDim.x) {
        A_S[i] = A[A_curr + i];
    }
    for (int i = threadIdx.x; i < B_length; i += blockDim.x) {
        B_S[i] = B[B_curr + i];
    }
    __syncthreads();

    /* Phase 3: each thread merges its private sub-range of the block chunk */
    int C_length = C_next - C_curr;
    if (C_length <= 0) return;   // this block owns no output elements
    int elementsPerThread = (C_length + blockDim.x - 1) / blockDim.x;
    int C_thread = C_curr + threadIdx.x * elementsPerThread;        // start output index
    int C_thread_next = min(C_thread + elementsPerThread, C_next);  // end output index
    if (C_thread >= C_thread_next) return;  // this thread has no output elements

    // Cut points inside the tiles (co-rank on shared memory, cheap).
    int k_curr = C_thread - C_curr;                       // offset within block chunk
    int k_next = k_curr + (C_thread_next - C_thread);
    int i_thread = co_rank(k_curr, A_S, A_length, B_S, B_length);
    int i_thread_next = co_rank(k_next, A_S, A_length, B_S, B_length);
    int j_thread = k_curr - i_thread;
    int j_thread_next = k_next - i_thread_next;

    merge_sequential(&A_S[i_thread], i_thread_next - i_thread,
                     &B_S[j_thread], j_thread_next - j_thread,
                     &C[C_thread]);
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

    // The whole per-block chunk must fit in the shared-memory tile.
    const int chunk = (total + GRID_SIZE - 1) / GRID_SIZE;
    if (chunk > TILE_SIZE) {
        printf("Configuration error: chunk %d exceeds tile size %d\n",
               chunk, TILE_SIZE);
        return -1;
    }

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

    // Launch kernel with dynamically sized shared memory (A-tile + B-tile).
    size_t sharedMemSize = 2 * TILE_SIZE * sizeof(int);
    merge_tiled_kernel<<<GRID_SIZE, BLOCK_SIZE, sharedMemSize>>>(
        d_A, m, d_B, n, d_C, TILE_SIZE);

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

    printf("m = %d, n = %d, total = %d (grid = %d blocks x %d threads, tile = %d)\n",
           m, n, total, GRID_SIZE, BLOCK_SIZE, TILE_SIZE);
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
