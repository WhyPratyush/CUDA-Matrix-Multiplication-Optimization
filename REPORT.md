### IMPLEMENTATION DETAILS:

This project measures the differnet optimization techniques for CUDA Matrix Multiplication. 

- Naive: Each thread computes a single element of matrix C (C = A x B). Since the kernel accesses global memory everytime, it causes high latency.

    Key code snippet:

        int j = blockDim.x * blockIdx.x + threadIdx.x;
        int i = blockDim.y * blockIdx.y + threadIdx.y;

        if(i < N && j < N) {
            float C_val = 0.0f;
            for(int k = 0; k < N; k++) {
            C_val += A[i * N + k] * B[k * N + j];
            }
            C[i * N + j] = C_val;
        }

- Tiled: This is the first optimization which uses shared memory. It stores all the elements of A and B in submatrices As and Bs which are stored in shared memory, causing very less latency as compared to global memory.

    Key code snippet:

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

    Expected: 2-3x improvement as compared to naive
    Actual: 1.29x, due to memory bank conflicts

- Tiled + Coalesced: Coalescing is the process of the consecutive threads reading consecutive memory addresses. I tried to implement this by transposing the Bs matrix in the shared memory.

    Key code snippet:

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

    Expected: ~3x improvement compared to naive
    Actual: 1.27x, Although the global memory coalescing improved, it didnt do much for the shared memory and we still have one output per thread. The real reason that coalescing didn't work was the memory was already coalesced, the real bottleneck as seen from the ncu's scheduler, is the MIO throttle stalls, the threads filled up the shared memory queue so many had to wait for their turn causing the bottleneck.

- 1D Blocktiling: A single thread now computes multiple results instead of 1.

    Key Code Snippet:

        float threadResults[TM] = {0.0f};
        for (int i = 0; i < BK; ++i) {
            float bVal = Bs[i * TILE + threadCol];
            for (int j = 0; j < TM; ++j) {
                int aRow = threadRow * TM + j;
                float aVal = As[aRow * BK + i];
                threadResults[j] += aVal * bVal;
            }
        }

    This resulted in a significant increase in performance as we each thread does the work of multiple threads.

    Expected: Significant improvement compared to the previous optimizations
    Actual: 2.78x compared to naive

- 2D Blocktiling: Instead of having one array of results, we calculate a submatrix of C at a time, increasing the data reuse per thread.

    Key Code Snippet:

        float threadResults[TM * TN] = {0.0};
        float regM[TM] = {0.0};
        float regN[TN] = {0.0};

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

    Expected: Significant improvement compared to 1D blocktiling
    Actual: 4.05x Naive

- Vectorized: Instead of loading global memory one float at a time, each thread loads 4 consecutive floats in a single 128-bit transaction using float4. This cuts the number of shared-memory load/store instructions by 4x, directly reducing pressure on the MIO instruction pipeline.

    Key Code Snippet:

        float4 tmpA = reinterpret_cast<float4*>(&A[innerRowA * N + innerColA * 4])[0];
        As[(innerColA * 4 + 0) * (TILE + PAD) + innerRowA] = tmpA.x;
        As[(innerColA * 4 + 1) * (TILE + PAD) + innerRowA] = tmpA.y;
        As[(innerColA * 4 + 2) * (TILE + PAD) + innerRowA] = tmpA.z;
        As[(innerColA * 4 + 3) * (TILE + PAD) + innerRowA] = tmpA.w;

        reinterpret_cast<float4*>(&Bs[innerRowB * TILE + innerColB * 4])[0] =
            reinterpret_cast<float4*>(&B[innerRowB * N + innerColB * 4])[0];

        float4 tmpC = { threadResults[i*TN+j], threadResults[i*TN+j+1],
                        threadResults[i*TN+j+2], threadResults[i*TN+j+3] };
        reinterpret_cast<float4*>(&C[(threadRow*TM+i)*N + threadCol*TN+j])[0] = tmpC;

Expected: Meaningful improvement over 2D Blocktiling by relieving MIO pipe pressure, since profiling identified this as the dominant stall reason (not bank conflicts or bandwidth).
Actual: 4.95x Naive (61.3% of cuBLAS), up from 4.29x (53.1%) for 2D Blocktiling. Store-side bank conflicts introduced by the A transpose were confirmed via ncu (l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum) and fully eliminated with PAD=4; load-side conflicts (...op_ld.sum) remain unresolved

Compilation commands:

- Naive :   

            nvcc -O3 -arch=sm_75 naive.cu -o naive
            ./naive

- Tiled :

            nvcc -O3 -arch=sm_75 tiled.cu -o tiled
            ./tiled

- Tiled + Coalesced :

            nvcc -O3 -arch=sm_75 tiled_coalesced.cu -o tiled_coalesced
            ./tiled_coalesced

- 1DBlocktiling :

            nvcc -O3 -arch=sm_75 1DBlocktiling.cu -o 1DBlocktiling
            ./1DBlocktiling

- 2DBlocktiling :

            nvcc -O3 -arch=sm_75 2DBlocktiling.cu -o 2DBlocktiling 
            ./2DBlocktiling 

- Vectorized :
            nvcc -O3 -arch=sm_75 vectorized.cu -o vectorized
            ./vectorized

### PERFORMANCE ANALYSIS:

| Implementation               | Time (ms) | GFLOPS  | % of cuBLAS  | Speedup vs Naive  |
|------------------------------|-----------|---------|--------------|-------------------|
| Naive                        |     4.268 |  503.16 |        12.4% | 1.00x             |
| Tiled                        |     3.277 |  655.36 |        16.1% | 1.30x             |
| Tiled + Coalesced            |     3.313 |  648.18 |        15.9% | 1.29x             |
| 1D Blocktiling               |     1.497 | 1434.50 |        35.3% | 2.85x             |
| 2D Blocktiling               |     0.995 | 2157.84 |        53.1% | 4.29x             |
| Vectorized (float4 + padded) |     0.862 | 2490.68 |        61.3% | 4.95x             |
| cuBLAS                       |     0.528 | 4064.49 |       100.0% | 8.08x             |



![PERFORMANCE PROGRESSION](PerformanceAnalysis.png)

### Diminishing Returns

As optimization levels increase memory bandwidth becomes less of a bottleneck and the improvements taper off unless we use specific techniques like vectorization or wraptiling.

