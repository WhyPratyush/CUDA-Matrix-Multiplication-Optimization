%%writefile naive.cu
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>


__global__ void matMulNaive(float *A, float *B, float *C,int N) {
  int j = blockDim.x * blockIdx.x + threadIdx.x;
  int i = blockDim.y * blockIdx.y + threadIdx.y;

  if(i < N && j < N) {
    float C_val = 0.0f;
    for(int k = 0; k < N; k++) {
      C_val += A[i * N + k] * B[k * N + j];
    }
    C[i * N + j] = C_val;
  }
}

int main() {
  const int N = 1024;
  const size_t size = N * N * sizeof(float);
  float *h_A = (float*)malloc(size);
  float *h_B = (float*)malloc(size);
  float *h_C = (float*)malloc(size);

  for(int i = 0; i < N * N; i++) {
    h_A[i] = 1.0f;
    h_B[i] = 2.0f;
  }

  float* d_A;
  cudaMalloc(&d_A, size);
  float* d_B;
  cudaMalloc(&d_B, size);
  float* d_C;
  cudaMalloc(&d_C, size);

  cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  dim3 threadsPerBlock(32, 32);
  dim3 numBlocks(N/32,N/32);

  cudaEventRecord(start,0);
  matMulNaive<<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_C, N);
  cudaEventRecord(stop,0);

  cudaEventSynchronize(stop);

  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);
  printf("Kernel finished in: %f ms\n", milliseconds);

  double gflops = (2.0 * N * N * N) / (milliseconds / 1000.0) / 1e9;
  printf("GFLOPS: %f\n", gflops);

  cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);

  free(h_A);
  free(h_B);
  free(h_C);

  return 0;
}