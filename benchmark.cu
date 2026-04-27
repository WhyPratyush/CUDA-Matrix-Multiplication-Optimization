%%writefile benchmark.cu
#include <stdio.h>
#include <stdlib.h>
#include <math.h> 
#include <cuda_runtime.h>
#include <cublas_v2.h> 

const int N = 1024;
const int TILE = 32;


void initializeMatrices(float *h_A, float *h_B, int n) {
    for(int i = 0; i < n * n; i++) {
        h_A[i] = 1.0f;
        h_B[i] = 2.0f;
    }
}


void verify(float *h_C, float expected_value, int n, const char* kernel_name) {
    bool success = true;
    for (int i = 0; i < n * n; ++i) {
        if (fabs(h_C[i] - expected_value) > 1e-5) {
            printf("[%s] Verification FAILED at index %d! Expected: %f, Got: %f\n", 
                   kernel_name, i, expected_value, h_C[i]);
            success = false;
            break;
        }
    }
    if (success) {
        printf("[%s] Verification PASSED!\n", kernel_name);
    }
}

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


__global__ void matMulTiled(float *A, float *B, float *C, int N) {
    int cRow = blockIdx.y;
    int cCol = blockIdx.x;

    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];

    int threadCol = threadIdx.x;
    int threadRow = threadIdx.y;

    int row = cRow * TILE + threadRow;
    int col = cCol * TILE + threadCol;

    float tmp = 0.0f;

    for (int t = 0; t < N/TILE; t++) {
        int aCol = t * TILE + threadCol;
        int bRow = t * TILE + threadRow;

        As[threadRow][threadCol] = A[row * N + aCol];
        Bs[threadRow][threadCol] = B[bRow * N + col];

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < TILE; ++k) {
            tmp += As[threadRow][k] * Bs[k][threadCol];
        }

        __syncthreads();
    }

    if (row < N && col < N) C[row * N + col] = tmp;
}

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

#define TILE_1D 64
#define BK 8    // No. of blocks along K dimension
#define TM 8

__global__ void matMul1D(float *A, float *B, float *C, int N) {
    int cRow = blockIdx.y;
    int cCol = blockIdx.x;
    __shared__ float As[TILE_1D * BK];
    __shared__ float Bs[BK * TILE_1D];

    int tid = threadIdx.x;
    int threadCol = tid % TILE_1D;
    int threadRow = tid / TILE_1D;
    int innerColA = tid % BK;
    int innerRowA = tid / BK;
    int innerColB = tid % TILE_1D;
    int innerRowB = tid / TILE_1D;

    float threadResults[TM] = {0.0f};
    int baseRow = cRow * TILE_1D;
    int baseCol = cCol * TILE_1D;

    int numTiles = N/BK;
    for (int t = 0; t < numTiles; ++t) {
        int kBase = t * BK;

        int aRow = baseRow + innerRowA;
        int aCol = kBase + innerColA;
        As[innerRowA * BK + innerColA] = A[aRow * N + aCol];

        int bRow = kBase + innerRowB;
        int bCol = baseCol + innerColB;
        Bs[innerRowB * TILE_1D + innerColB] = B[bRow * N + bCol];

        __syncthreads();

        #pragma unroll
        for (int i = 0; i < BK; ++i) {
            float bVal = Bs[i * TILE_1D + threadCol];
            for (int j = 0; j < TM; ++j) {
                int aRow = threadRow * TM + j;
                float aVal = As[aRow * BK + i];
                threadResults[j] += aVal * bVal;
            }
        }
        __syncthreads();
    }

    for (int i = 0; i < TM; ++i) {
        int outRow = baseRow + (threadRow * TM + i);
        int outCol = baseCol + threadCol;
        if (outRow < N && outCol < N) {
            C[outRow * N + outCol] = threadResults[i];
        }
    }
}


#define TILE_2D 128
#define TN 8

__global__ void matMul2D(float *A, float *B, float *C,int N) {
  const int cRow = blockIdx.y;
  const int cCol = blockIdx.x;

  const int totalResultsBlocktile = TILE_2D * TILE_2D;
  const int numThreadsBlocktile = totalResultsBlocktile / (TM * TN);

  const int threadCol = threadIdx.x % (TILE_2D / TN);
  const int threadRow = threadIdx.x / (TILE_2D / TN);

  __shared__ float As[TILE_2D * BK];
  __shared__ float Bs[BK * TILE_2D];

  A += cRow * TILE_2D * N;
  B += cCol * TILE_2D;
  C += cRow * TILE_2D * N + cCol * TILE_2D;

  const int innerRowA = threadIdx.x / BK;
  const int innerColA = threadIdx.x % BK;
  const int strideA = numThreadsBlocktile / BK;
  const int innerRowB = threadIdx.x / TILE_2D;
  const int innerColB = threadIdx.x % TILE_2D;
  const int strideB = numThreadsBlocktile / TILE_2D;

  float threadResults[TM * TN] = {0.0};
  float regM[TM] = {0.0};
  float regN[TN] = {0.0};

  for (int t = 0; t < N; t += BK) {
    for (int i = 0; i < TILE_2D; i += strideA) {
      As[(innerRowA + i) * BK + innerColA] =
          A[(innerRowA + i) * N + innerColA];
    }
    for (int j = 0; j < BK; j += strideB) {
      Bs[(innerRowB + j) * TILE_2D + innerColB] =
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
        regN[i] = Bs[k * TILE_2D + threadCol * TN + i];
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
    const size_t size = N * N * sizeof(float);
    float *h_A, *h_B, *h_C;
    float *d_A, *d_B, *d_C;


    h_A = (float*)malloc(size);
    h_B = (float*)malloc(size);
    h_C = (float*)malloc(size);

    initializeMatrices(h_A, h_B, N);
    const float expected_value = 1.0f * 2.0f * N;

    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);

    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cublasHandle_t handle;
    cublasCreate(&handle);
    float alpha = 1.0f;
    float beta = 0.0f;

    float naive_ms = 0.0f;
    float non_coalesced_ms = 0.0f;
    float tiled_2d_ms = 0.0f;
    float tiled_1d_ms = 0.0f;
    float reg_blocked_ms = 0.0f;
    float cublas_ms = 0.0f;

    printf("Running Matrix Multiplication Benchmarks (N=%d)\n", N);
    printf("===================================================\n");

    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &alpha, d_B, N, d_A, N, &beta, d_C, N);
    cudaDeviceSynchronize();

    printf("\n--- 1. Naive ---\n");
    cudaMemset(d_C, 0, size);
    dim3 threadsPerBlockNaive(32, 32);
    dim3 numBlocksNaive(N/32,N/32);
    matMulNaive<<<numBlocksNaive, threadsPerBlockNaive>>>(d_A, d_B, d_C, N);
    cudaDeviceSynchronize();

    cudaEventRecord(start, 0);
    matMulNaive<<<numBlocksNaive, threadsPerBlockNaive>>>(d_A, d_B, d_C, N);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&naive_ms, start, stop);
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);
    verify(h_C, expected_value, N, "Naive");
    printf("Time: %f ms\n", naive_ms);

    printf("\n--- 2. Tiled ---\n");
    cudaMemset(d_C, 0, size);
    dim3 threadsPerBlockTiled(TILE, TILE);
    dim3 numBlocksTiled(N / TILE, N / TILE);
    matMulTiled<<<numBlocksTiled, threadsPerBlockTiled>>>(d_A, d_B, d_C, N);
    cudaDeviceSynchronize();

    cudaEventRecord(start, 0);
    matMulTiled<<<numBlocksTiled, threadsPerBlockTiled>>>(d_A, d_B, d_C, N);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&non_coalesced_ms, start, stop);
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);
    verify(h_C, expected_value, N, "Tiled");
    printf("Time: %f ms\n", non_coalesced_ms);

    printf("\n--- 3. Tiled + Coalesced ---\n");
    cudaMemset(d_C, 0, size);
    matMulTiled_Coalesced<<<numBlocksTiled, threadsPerBlockTiled>>>(d_A, d_B, d_C, N);
    cudaDeviceSynchronize();
    cudaEventRecord(start, 0);
    matMulTiled_Coalesced<<<numBlocksTiled, threadsPerBlockTiled>>>(d_A, d_B, d_C, N);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&tiled_2d_ms, start, stop);
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);
    verify(h_C, expected_value, N, "Tiled + Coalesced");
    printf("Time: %f ms\n", tiled_2d_ms);
    
    printf("\n--- 4. 1D Blocktiling ---\n");
    cudaMemset(d_C, 0, size);
    dim3 threadsPerBlockTiled1D(TILE_1D * BK);
    dim3 numBlocksTiled1D(N/TILE_1D, N/TILE_1D);
    matMul1D<<<numBlocksTiled1D, threadsPerBlockTiled1D>>>(d_A, d_B, d_C, N);
    cudaDeviceSynchronize();
    cudaEventRecord(start, 0);
    matMul1D<<<numBlocksTiled1D, threadsPerBlockTiled1D>>>(d_A, d_B, d_C, N);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&tiled_1d_ms, start, stop);
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);
    verify(h_C, expected_value, N, "1D Blocktiling");
    printf("Time: %f ms\n", tiled_1d_ms);

    printf("\n--- 5. Tiled 2D + Register Blocking ---\n");
    cudaMemset(d_C, 0, size);
    dim3 threadsPerBlockReg((TILE_2D * TILE_2D)/(TM*TN)); 
    dim3 numBlocksReg(N / TILE_2D, N / TILE_2D);
    matMul2D<<<numBlocksReg, threadsPerBlockReg>>>(d_A, d_B, d_C, N);
    cudaDeviceSynchronize();
    cudaEventRecord(start, 0);
    matMul2D<<<numBlocksReg, threadsPerBlockReg>>>(d_A, d_B, d_C, N);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&reg_blocked_ms, start, stop);
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);
    verify(h_C, expected_value, N, "2D Blocktiling");
    printf("Time: %f ms\n", reg_blocked_ms);
    
    printf("\n--- 6. cuBLAS ---\n");
    cudaMemset(d_C, 0, size);
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &alpha, d_B, N, d_A, N, &beta, d_C, N);
    cudaDeviceSynchronize();
    cudaEventRecord(start, 0);
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &alpha, d_B, N, d_A, N, &beta, d_C, N);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&cublas_ms, start, stop);
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);
    verify(h_C, expected_value, N, "cuBLAS");
    printf("Time: %f ms\n", cublas_ms);

    printf("\n\n--- Performance Summary Table ---\n\n");
    
    double ops = 2.0 * (double)N * (double)N * (double)N;
    double gflops_naive = (ops / (naive_ms / 1000.0)) / 1e9;
    double gflops_non_coalesced = (ops / (non_coalesced_ms / 1000.0)) / 1e9;
    double gflops_tiled_2d = (ops / (tiled_2d_ms / 1000.0)) / 1e9;
    double gflops_tiled_1d = (ops / (tiled_1d_ms / 1000.0)) / 1e9;
    double gflops_reg_blocked = (ops / (reg_blocked_ms / 1000.0)) / 1e9;
    double gflops_cublas = (ops / (cublas_ms / 1000.0)) / 1e9;

    printf("| Implementation               | Time (ms) | GFLOPS  | %% of cuBLAS  | Speedup vs Naive  |\n");
    printf("|------------------------------|-----------|---------|--------------|-------------------|\n");
    printf("| Naive                        | %9.3f | %7.2f | %11.1f%% | 1.00x            |\n", 
           naive_ms, gflops_naive, (gflops_naive / gflops_cublas) * 100.0);
    printf("| Tiled                        | %9.3f | %7.2f | %11.1f%% | %.2fx            |\n", 
           non_coalesced_ms, gflops_non_coalesced, (gflops_non_coalesced / gflops_cublas) * 100.0, gflops_non_coalesced / gflops_naive);
    printf("| Tiled + Coalesced            | %9.3f | %7.2f | %11.1f%% | %.2fx            |\n", 
           tiled_2d_ms, gflops_tiled_2d, (gflops_tiled_2d / gflops_cublas) * 100.0, gflops_tiled_2d / gflops_naive);
    printf("| 1D Blocktiling               | %9.3f | %7.2f | %11.1f%% | %.2fx            |\n", 
           tiled_1d_ms, gflops_tiled_1d, (gflops_tiled_1d / gflops_cublas) * 100.0, gflops_tiled_1d / gflops_naive);
    printf("| 2D Blocktiling               | %9.3f | %7.2f | %11.1f%% | %.2fx            |\n", 
           reg_blocked_ms, gflops_reg_blocked, (gflops_reg_blocked / gflops_cublas) * 100.0, gflops_reg_blocked / gflops_naive);
    printf("| cuBLAS                       | %9.3f | %7.2f | %11.1f%% | %.2fx            |\n", 
           cublas_ms, gflops_cublas, (gflops_cublas / gflops_cublas) * 100.0, gflops_cublas / gflops_naive);

    cublasDestroy(handle);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    free(h_A);
    free(h_B);
    free(h_C);

    return 0;
}