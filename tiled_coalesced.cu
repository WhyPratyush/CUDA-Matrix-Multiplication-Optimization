%%writefile tiled_coalesced.cu
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <math.h>

#define TILE 32

__global__ void matMulTiled_Coalesced(float *A, float *B, float *C, int N) {
    int cRow = blockIdx.y;
    int cCol = blockIdx.x;

    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE+1];

    int threadCol = threadIdx.x;
    int threadRow = threadIdx.y;

    int row = cRow * TILE + threadRow;
    int col = cCol * TILE + threadCol;

    float tmp = 0.0f;

    for (int t = 0; t < N/TILE; t++) {
        int aCol = t * TILE + threadCol;
        int bRow = t * TILE + threadRow;

        As[threadRow][threadCol] = A[row * N + aCol];
        Bs[threadCol][threadRow] = B[bRow * N + col];

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < TILE; ++k) {
            tmp += As[threadRow][k] * Bs[threadCol][k];
        }

        __syncthreads();
    }

    if (row < N && col < N) C[row * N + col] = tmp;
}

int main() {
    const int N = 1024;
    const size_t size = (size_t)N * N * sizeof(float);
    float *h_A = (float*)malloc(size);
    float *h_B = (float*)malloc(size);
    float *h_C = (float*)malloc(size);

    for (int i = 0; i < N * N; i++) {
        h_A[i] = 1.0f;
        h_B[i] = 2.0f;
    }

    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);

    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(TILE, TILE);
    dim3 numBlocks(N/TILE,N/TILE);

    matMulTiled_Coalesced<<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_C, N);
    cudaDeviceSynchronize();

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start, 0);
    matMulTiled_Coalesced<<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_C, N);
    cudaEventRecord(stop, 0);

    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Kernel finished in: %f ms\n", milliseconds);

    double gflops = (2.0 * (double)N * N * N) / ((double)milliseconds / 1000.0) / 1e9;
    printf("GFLOPS: %f\n", gflops);

    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    float expected = 1.0f * 2.0f * N;
    bool success = true;
    for (int i = 0; i < N * N; i++) {
        if (fabs(h_C[i] - expected) > 1e-3) {
            printf("Verification FAILED at %d: Expected %f, Got %f\n", i, expected, h_C[i]);
            success = false;
            break;
        }
    }

    if (success)
        printf("Verification PASSED! All elements are %f\n", expected);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    free(h_A);
    free(h_B);
    free(h_C);

    return 0;
}
