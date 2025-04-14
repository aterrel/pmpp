#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <math.h>

#define FILTER_RADIUS 2
#define IN_TILE_DIM 32
#define OUT_TILE_DIM ((IN_TILE_DIM - 2*FILTER_RADIUS))
#define TILE_DIM 32
__constant__ float c_F[2*FILTER_RADIUS+1][2*FILTER_RADIUS+1];

// CUDA error checking macro
#define checkCudaErrors(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            printf("CUDA error at %s:%d: %s\n", \
                   __FILE__, __LINE__, \
                   cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)


__global__ void convolution_2D_basic_kernel(float *N, float *F, float *P, 
                                            int r, int width, int height) {
    int outCol = blockIdx.x * blockDim.x + threadIdx.x;
    int outRow = blockIdx.y * blockDim.y + threadIdx.y;
    float Pvalue = 0.0f;
    int inRow, inCol;
    for (int fRow = 0; fRow < 2*r+1; fRow++) {
        for (int fCol = 0; fCol < 2*r+1; fCol++) {
            inRow = outRow - r + fRow;
            inCol = outCol - r + fCol;
            if (inRow >= 0 && inRow < height && inCol >= 0 && inCol < width) {
                Pvalue += F[fRow * (2*r+1) + fCol] * N[inRow * width + inCol];
            }
        }
    }
    P[outRow + outCol * width] = Pvalue;
}

__global__ void convolution_2D_constant_mem_kernel(float *N, float *P, int r, int width, int height) {
    int outCol = blockIdx.x * blockDim.x + threadIdx.x;
    int outRow = blockIdx.y * blockDim.y + threadIdx.y;
    float Pvalue = 0.0f;
    int inRow, inCol;
    for (int fRow = 0; fRow < 2*r+1; fRow++) {
        for (int fCol = 0; fCol < 2*r+1; fCol++) {
            inRow = outRow - r + fRow;
            inCol = outCol - r + fCol;
            if (inRow >= 0 && inRow < height && inCol >= 0 && inCol < width) {
                Pvalue += c_F[fRow][fCol] * N[inRow * width + inCol];
            }
        }
    }
    P[outRow + outCol * width] = Pvalue;
}

__global__ void convolution_tiled_2D_const_mem_kernel(float *N, float *P, int width, int height) {
    int col = blockIdx.x * OUT_TILE_DIM + threadIdx.x - FILTER_RADIUS;
    int row = blockIdx.y * OUT_TILE_DIM + threadIdx.y - FILTER_RADIUS;
    // loading the input tile
    __shared__ float N_s[IN_TILE_DIM][IN_TILE_DIM];
    if (row>=0 && row<height && col>=0 && col<width) {
        N_s[threadIdx.y][threadIdx.x] = N[row*width + col];
    } else {
        N_s[threadIdx.y][threadIdx.x] = 0.0f;
    }
    __syncthreads();
    // Calculating the output elements
    int tileCol = threadIdx.x - FILTER_RADIUS;
    int tileRow = threadIdx.y - FILTER_RADIUS;
    // turn off the threads at the edges of the block
    if (col>=0 && col<width && row>=0 && row<height) {
        if (tileCol>=0 && tileCol<OUT_TILE_DIM && tileRow>=0 && tileRow<OUT_TILE_DIM) {
            float Pvalue = 0.0f;
            for (int fRow = 0; fRow < 2*FILTER_RADIUS+1; fRow++) {
                for (int fCol = 0; fCol < 2*FILTER_RADIUS+1; fCol++) {
                    Pvalue += c_F[fRow][fCol]*N_s[tileRow+fRow][tileCol+fCol];
                }
            }
            P[row*width+col] = Pvalue;
        }
    }
}

__global__ void convolution_cached_tiled_2D_const_mem_kernel(float *N, float *P, int width, int height) {
    int col = blockIdx.x*TILE_DIM + threadIdx.x;
    int row = blockIdx.y*TILE_DIM + threadIdx.y;
    __shared__ float N_s[TILE_DIM][TILE_DIM];
    if (row<height && col<width) {
        N_s[threadIdx.y][threadIdx.x] = N[row*width+col];
    } else {
        N_s[threadIdx.y][threadIdx.x] = 0.0f;
    }
    __syncthreads();

    // Calculating the output elements
    int input_x = threadIdx.x - FILTER_RADIUS;
    int input_y = threadIdx.y - FILTER_RADIUS;
    if (col<width && row<height) {
        float Pvalue = 0.0f;
        for (int fRow = 0; fRow < 2*FILTER_RADIUS+1; fRow++) {
            for (int fCol = 0; fCol < 2*FILTER_RADIUS+1; fCol++) {
                if (input_x+fCol >= 0 &&
                    input_x+fCol < TILE_DIM &&
                    input_y+fRow >= 0 &&
                    input_y+fRow < TILE_DIM){
                        Pvalue += c_F[fRow][fCol]*N_s[input_y+fRow][input_x+fCol];
                } else {
                    if (row-FILTER_RADIUS+fRow >= 0 &&
                        row-FILTER_RADIUS+fRow < height &&
                        col-FILTER_RADIUS+fCol >= 0 &&
                        col-FILTER_RADIUS+fCol < width){
                            Pvalue += c_F[fRow][fCol]*N[(row-FILTER_RADIUS+fRow)*width + col-FILTER_RADIUS+fCol];
                        }
                }
            }
        }
        P[row*width+col] = Pvalue;
    }
}

int main() {
    int width = 8192;
    int height = 8192;
    int r = 1;
    int nBytes = width * height * sizeof(float);
    float *h_N = (float*)malloc(nBytes);
    float *h_P = (float*)malloc(nBytes);

    float h_F[25] = {
        1, 2, 1, 2, 1, 
        2, 4, 2, 4, 2, 
        1, 2, 1, 2, 1, 
        2, 4, 2, 4, 2, 
        1, 2, 1, 2, 1
    };

    srand(time(NULL));
    for (int i = 0; i < width * height; ++i) {
        h_N[i] = rand() / (float)RAND_MAX;
    }

    float *d_N, *d_F, *d_P;
    checkCudaErrors(cudaMalloc(&d_N, nBytes));
    checkCudaErrors(cudaMalloc(&d_F, 2*r+1 * 2*r+1 * sizeof(float)));
    checkCudaErrors(cudaMalloc(&d_P, nBytes));

    checkCudaErrors(cudaMemcpy(d_N, h_N, nBytes, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_F, h_F, 2*r+1 * 2*r+1 * sizeof(float), cudaMemcpyHostToDevice));

    dim3 block(TILE_DIM, TILE_DIM);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);

    convolution_2D_basic_kernel<<<grid, block>>>(d_N, d_F, d_P, r, width, height);
     // Check for kernel launch errors
    checkCudaErrors(cudaGetLastError());
    // Wait for kernel to complete and check for errors
    checkCudaErrors(cudaDeviceSynchronize());
    
    checkCudaErrors(cudaMemcpyToSymbol(c_F, h_F, (2*FILTER_RADIUS+1) * (2*FILTER_RADIUS+1) * sizeof(float)));
    convolution_2D_constant_mem_kernel<<<grid, block>>>(d_N, d_P, r, width, height);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
    
    dim3 block_tiled(IN_TILE_DIM, IN_TILE_DIM);
    dim3 grid_tiled((width + OUT_TILE_DIM - 1) / OUT_TILE_DIM, (height + OUT_TILE_DIM - 1) / OUT_TILE_DIM);

    convolution_tiled_2D_const_mem_kernel<<<grid_tiled, block_tiled>>>(d_N, d_P, width, height);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    convolution_cached_tiled_2D_const_mem_kernel<<<grid, block>>>(d_N, d_P, width, height);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());


    checkCudaErrors(cudaMemcpy(h_P, d_P, nBytes, cudaMemcpyDeviceToHost));
    

    checkCudaErrors(cudaFree(d_N));
    checkCudaErrors(cudaFree(d_F));
    checkCudaErrors(cudaFree(d_P));
}