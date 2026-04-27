%%writefile 2DBlocktiling.cu
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <math.h>

#define BK 8
#define TILE 128
#define TM 8
#define TN 8

__global__ void matMul2D(float *A, float *B, float *C,int N) {
  const int cRow = blockIdx.y;
  const int cCol = blockIdx.x;

  const int totalResultsBlocktile = TILE * TILE;
  const int numThreadsBlocktile = totalResultsBlocktile / (TM * TN);

  const int threadCol = threadIdx.x % (TILE / TN);
  const int threadRow = threadIdx.x / (TILE / TN);

  __shared__ float As[TILE * BK];
  __shared__ float Bs[BK * TILE];

  A += cRow * TILE * N;
  B += cCol * TILE;
  C += cRow * TILE * N + cCol * TILE;

  const int innerRowA = threadIdx.x / BK;
  const int innerColA = threadIdx.x % BK;
  const int strideA = numThreadsBlocktile / BK;
  const int innerRowB = threadIdx.x / TILE;
  const int innerColB = threadIdx.x % TILE;
  const int strideB = numThreadsBlocktile / TILE;

  float threadResults[TM * TN] = {0.0};
  float regM[TM] = {0.0};
  float regN[TN] = {0.0};

  for (int t = 0; t < N; t += BK) {
    for (int i = 0; i < TILE; i += strideA) {
      As[(innerRowA + i) * BK + innerColA] =
          A[(innerRowA + i) * N + innerColA];
    }
    for (int j = 0; j < BK; j += strideB) {
      Bs[(innerRowB + j) * TILE + innerColB] =
          B[(innerRowB + j) * N + innerColB];
    }
    __syncthreads();

    A += BK;
    B += BK * N;

    for (int k = 0; k < BK; ++k) {

      for (int i = 0; i < TM; ++i) {
        regM[i] = As[(threadRow * TM + i) * BK + k];
      }
      for (int i = 0; i < TN; ++i) {
        regN[i] = Bs[k * TILE + threadCol * TN + i];
      }
      for (int i = 0; i < TM; ++i) {
        for (int j = 0; j < TN; ++j) {
          threadResults[i * TN + j] += regM[i] * regN[j];
        }
      }
    }
    __syncthreads();
  }

  for (int i = 0; i < TM; ++i) {
    for (int j = 0; j < TN; ++j) {
      C[(threadRow * TM + i) * N + threadCol * TN + j] = threadResults[i * TN + j];
    }
  }
}
int main() {
  const int N = 1024;
  const size_t size = (size_t)N * N * sizeof(float);

  float *h_A = (float*)malloc(size);
  float *h_B = (float*)malloc(size);
  float *h_C = (float*)malloc(size);

  for (int i = 0; i < N * N; ++i) {
    h_A[i] = 1.0f;
    h_B[i] = 2.0f;
  }

  float *d_A=nullptr, *d_B=nullptr, *d_C=nullptr;
  cudaMalloc(&d_A, size);
  cudaMalloc(&d_B, size);
  cudaMalloc(&d_C, size);

  cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_C, h_C, size, cudaMemcpyHostToDevice);

  dim3 gridDim(N/TILE, N/TILE);
  dim3 blockDim((TILE * TILE) / (TM * TN));

  matMul2D<<<gridDim, blockDim>>>(d_A, d_B, d_C, N);
  cudaDeviceSynchronize();

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start, 0);
  matMul2D<<<gridDim, blockDim>>>(d_A, d_B, d_C, N);
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);

  float ms = 0.0f;
  cudaEventElapsedTime(&ms, start, stop);
  double gflops = (2.0 * (double)N * N * N) / (ms / 1000.0) / 1e9;
  printf("Kernel finished in: %f ms\n", ms);
  printf("GFLOPS: %f\n", gflops);

  cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

  float expected = 1.0f * 2.0f * (float)N;
  bool ok = true;
  for (int i = 0; i < N * N; ++i) {
    if (fabs(h_C[i] - expected) > 1e-2) {
      printf("Verification FAILED at %d: expected %f got %f\n", i, expected, h_C[i]);
      ok = false;
      break;
    }
  }
  if (ok) printf("Verification PASSED! All elements are %f\n", expected);

  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);
  free(h_A);
  free(h_B);
  free(h_C);
  return ok ? 0 : 1;
}
