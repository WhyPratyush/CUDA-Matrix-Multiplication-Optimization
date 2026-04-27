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

- Benchmark :

            nvcc -O3 -arch=sm_75 benchmark.cu -o benchmark -lcublas
            ./benchmark
