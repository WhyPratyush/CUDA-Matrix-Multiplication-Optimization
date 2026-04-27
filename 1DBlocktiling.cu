%%writefile 1DBlocktiling.cu
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <math.h>

#define TILE 64
#define BK 8    // No. of blocks along K dimension
#define TM 8

__global__ void matMul1D(float *A, float *B, float *C, int N) {
    int cRow = blockIdx.y;
    int cCol = blockIdx.x;
    __shared__ float As[TILE * BK];
    __shared__ float Bs[BK * TILE];

    int tid = threadIdx.x;
    int threadCol = tid % TILE;
    int threadRow = tid / TILE;
    int innerColA = tid % BK;
    int innerRowA = tid / BK;
    int innerColB = tid % TILE;
    int innerRowB = tid / TILE;

    float threadResults[TM] = {0.0f};
    int baseRow = cRow * TILE;
    int baseCol = cCol * TILE;

    int numTiles = N/BK;
    for (int t = 0; t < numTiles; ++t) {
        int kBase = t * BK;

        int aRow = baseRow + innerRowA;
        int aCol = kBase + innerColA;
        As[innerRowA * BK + innerColA] = A[aRow * N + aCol];

        int bRow = kBase + innerRowB;
        int bCol = baseCol + innerColB;
        Bs[innerRowB * TILE + innerColB] = B[bRow * N + bCol];

        __syncthreads();

        #pragma unroll
        for (int dotIdx = 0; dotIdx < BK; ++dotIdx) {
            float tmpB = Bs[dotIdx * TILE + threadCol];
            for (int resIdx = 0; resIdx < TM; ++resIdx) {
                int aRow = threadRow * TM + resIdx;
                float aVal = As[aRow * BK + dotIdx];
                threadResults[resIdx] += aVal * tmpB;
            }
        }
        __syncthreads();
    }

    for (int resIdx = 0; resIdx < TM; ++resIdx) {
        int outRow = baseRow + (threadRow * TM + resIdx);
        int outCol = baseCol + threadCol;
        if (outRow < N && outCol < N) {
            C[outRow * N + outCol] = threadResults[resIdx];
        }
    }
}

int main() {
    const int N = 1024;
    const size_t size = (size_t)N * N * sizeof(float);
    float *h_A = (float *)malloc(size);
    float *h_B = (float *)malloc(size);
    float *h_C = (float *)malloc(size);

    for (int i = 0; i < N * N; i++) {
        h_A[i] = 1.0f;
        h_B[i] = 2.0f;
        h_C[i] = 0.0f;
    }

    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);

    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(TILE * BK);
    dim3 numBlocks(N/TILE,N/TILE);

    matMul1D<<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_C, N);
    cudaDeviceSynchronize();

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    matMul1D<<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_C, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0.0f;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Kernel finished in: %f ms\n", milliseconds);

    double gflops = (2.0 * (double)N * N * N) / ((double)milliseconds / 1000.0) / 1e9;
    printf("GFLOPS: %f\n", gflops);

    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    float expected = 1.0f * 2.0f * N;
    bool success = true;
    for (int i = 0; i < N * N; i++) {
        if (fabs(h_C[i] - expected) > 1e-2) {
            printf("Verification FAILED at %d: Expected %f, Got %f\n", i, expected, h_C[i]);
            success = false;
            break;
        }
    }
    if (success) printf("Verification PASSED! All elements are %f\n", expected);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C);
    return 0;
}
