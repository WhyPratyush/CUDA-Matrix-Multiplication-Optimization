%%writefile vectorized.cu
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <math.h>

#define BK 8
#define TILE 128
#define TM 8
#define TN 8
#define PAD 4

__global__ void matMul2D(float *A, float *B, float *C,int N) {
  const int cRow = blockIdx.y;
  const int cCol = blockIdx.x;

  const int threadCol = threadIdx.x % (TILE / TN);
  const int threadRow = threadIdx.x / (TILE / TN);

  __shared__ float As[BK * (TILE+PAD)];
  __shared__ float Bs[BK * TILE];

  A += cRow * TILE * N;
  B += cCol * TILE;
  C += cRow * TILE * N + cCol * TILE;

  const int innerRowA = threadIdx.x / (BK/4);
  const int innerColA = threadIdx.x % (BK/4);
  const int innerRowB = threadIdx.x / (TILE/4);
  const int innerColB = threadIdx.x % (TILE/4);

  float threadResults[TM * TN] = {0.0};
  float regM[TM] = {0.0};
  float regN[TN] = {0.0};

  for (int t = 0; t < N; t += BK) {
    float4 tmpA = reinterpret_cast<float4 *>(&A[innerRowA * N + innerColA * 4])[0];
    As[(innerColA * 4 + 0) * (TILE+PAD) + innerRowA] = tmpA.x;
    As[(innerColA * 4 + 1) * (TILE+PAD) + innerRowA] = tmpA.y;
    As[(innerColA * 4 + 2) * (TILE+PAD) + innerRowA] = tmpA.z;
    As[(innerColA * 4 + 3) * (TILE+PAD) + innerRowA] = tmpA.w;
    
    reinterpret_cast<float4 *>(&Bs[innerRowB * TILE + innerColB * 4])[0] =
        reinterpret_cast<float4 *>(&B[innerRowB * N + innerColB * 4])[0];

    __syncthreads();

    A += BK;
    B += BK * N;

    for (int k = 0; k < BK; ++k) {
      for (int i = 0; i < TM; ++i) {
        regM[i] = As[k * (TILE+PAD) + threadRow * TM + i];
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
    for (int j = 0; j < TN; j += 4) {
      float4 tmpC;
      tmpC.x = threadResults[i * TN + j + 0];
      tmpC.y = threadResults[i * TN + j + 1];
      tmpC.z = threadResults[i * TN + j + 2];
      tmpC.w = threadResults[i * TN + j + 3];
      reinterpret_cast<float4 *>(
          &C[(threadRow * TM + i) * N + threadCol * TN + j])[0] = tmpC;
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
