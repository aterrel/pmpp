#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <math.h>

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


#define TILE_WIDTH 32
#define COARSE_FACTOR 4

__global__ void MatrixMulKernel(float* M, float* N,
                                 float* P, int Width)                                
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if ((row < Width) && (col < Width)) {
        float Pvalue = 0;
        for (int k = 0; k < Width; ++k) {
            Pvalue += M[row*Width+k] * N[k*Width + col];
        }
        P[row*Width + col] = Pvalue;
    }
}


__global__ void MatrixMulKernel_roworder(float* M, float* N,
                                         float* P, int Width)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col < Width) {
        for (int row = 0; row < Width; ++row) {
            float Pvalue = 0;
            for (int k = 0; k < Width; ++k) {
                Pvalue += M[row*Width + k] * N[k*Width + col];
            }
            P[row*Width + col] = Pvalue;
        }
    }
}

__global__ void MatrixMulKernel_colorder(float* M, float* N,
                                         float* P, int Width)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row < Width) {
        for (int col = 0; col < Width; ++col) {
            float Pvalue = 0;
            for (int k = 0; k < Width; ++k) {
                Pvalue += M[row*Width + k] * N[k*Width + col];
            }
            P[row*Width + col] = Pvalue;
        }
    }
}

__global__ void MatrixMulKernel_tiled(float *M, float *N, float *P, int Width) {
    __shared__ float Mds[TILE_WIDTH][TILE_WIDTH];
    __shared__ float Nds[TILE_WIDTH][TILE_WIDTH];

    int bx = blockIdx.x; int by = blockIdx.y;
    int tx = threadIdx.x; int ty = threadIdx.y;

    // Identify the row and column of the P element to work on
    int Row = by * TILE_WIDTH + ty;
    int Col = bx * TILE_WIDTH + tx;

    // Loop over the M and N tiles required to compute the P element
    float Pvalue =  0;
    for (int ph = 0; ph < (int)ceilf(((float)Width/TILE_WIDTH)); ++ph) {
        // Collaborative loading of M and N tiles into shared memory
        if ((Row < Width) && (ph*TILE_WIDTH + tx) < Width) 
            Mds[ty][tx] = M[Row*Width + ph*TILE_WIDTH + tx];
        else
            Mds[ty][tx] = 0.0f;
        if ((Col < Width) && (ph*TILE_WIDTH + ty) < Width)
            Nds[ty][tx] = N[(ph*TILE_WIDTH + ty)*Width + Col];
        else
            Nds[ty][tx] = 0.0f;
        __syncthreads();

        // Matrix multiplication on the small tiles
        for (int k = 0; k < TILE_WIDTH; ++k) {
            Pvalue += Mds[ty][k] * Nds[k][tx];
        }
        __syncthreads();
    }
    if ((Row < Width) && (Col < Width))
        P[Row*Width + Col] = Pvalue;
}

__global__ void MatrixMulKernel_tiled_dynamic(float *M, float *N, float *P, int Width,
                                             unsigned Mdz_sz, unsigned Ndz_sz) {

    extern __shared__ char Mds_Nds[];

    float *Mds = (float*)Mds_Nds;
    float *Nds = (float*)(Mds_Nds + Mdz_sz);

    int bx = blockIdx.x; int by = blockIdx.y;
    int tx = threadIdx.x; int ty = threadIdx.y;

    // Identify the row and column of the P element to work on
    int Row = by * TILE_WIDTH + ty;
    int Col = bx * TILE_WIDTH + tx;

    // Loop over the M and N tiles required to compute the P element
    float Pvalue =  0;
    for (int ph = 0; ph < (int)ceilf(((float)Width/TILE_WIDTH)); ++ph) {
        // Collaborative loading of M and N tiles into shared memory
        if ((Row < Width) && (ph*TILE_WIDTH + tx) < Width) 
            Mds[ty*TILE_WIDTH + tx] = M[Row*Width + ph*TILE_WIDTH + tx];
        else
            Mds[ty*TILE_WIDTH + tx] = 0.0f;
        if ((Col < Width) && (ph*TILE_WIDTH + ty) < Width)
            Nds[tx*TILE_WIDTH + ty] = N[(ph*TILE_WIDTH + ty)*Width + Col];
        else
            Nds[tx*TILE_WIDTH + ty] = 0.0f;
        __syncthreads();

        // Matrix multiplication on the small tiles
        for (int k = 0; k < TILE_WIDTH; ++k) {
            Pvalue += Mds[ty * TILE_WIDTH + k] * Nds[k + tx * TILE_WIDTH];
        }
        __syncthreads();
    }
    if ((Row < Width) && (Col < Width))
        P[Row*Width + Col] = Pvalue;
}

__global__ void MatrixMulKernel_thread_coarsened(float *M, float *N, float *P, int width) {

    __shared__ float Mds[TILE_WIDTH][TILE_WIDTH];
    __shared__ float Nds[TILE_WIDTH][TILE_WIDTH];

    int bx = blockIdx.x;  int by = blockIdx.y;
    int tx = threadIdx.x; int ty = threadIdx.y;

    // Identify the row and column of the P element to work on.
    int row = by*TILE_WIDTH + ty;
    int colStart = bx*TILE_WIDTH*COARSE_FACTOR + tx;

    // Initialize Pvalue for all output elements
    float Pvalue[COARSE_FACTOR];
    for (int c = 0; c < COARSE_FACTOR; ++c) {
        Pvalue[c] = 0.0f;
    }

    // Loop over the M and N tiles required to compute P element
    for (int ph = 0; ph < width/TILE_WIDTH; ++ph) {

        // Collaborative loading of M tile into shared memory
        Mds[ty][tx] = M[row*width + ph*TILE_WIDTH + tx];

        for (int c = 0; c < COARSE_FACTOR; ++c) {

            int col = colStart + c*TILE_WIDTH;

            // Collaborative loading of N tile into shared memory
            Nds[ty][tx] = N[(ph*TILE_WIDTH + ty)*width + col];
            __syncthreads();

            for (int k = 0; k < TILE_WIDTH; ++k) {
                Pvalue[c] += Mds[ty][k]*Nds[k][tx];
            }
            __syncthreads();
        }
    }

    for (int c = 0; c < COARSE_FACTOR; ++c) {
        int col = colStart + c*TILE_WIDTH;
        P[row*width + col] = Pvalue[c];
    }
}




void MatrixMulCPU(float* M, float* N, float* P, int Width) {
    for (int row = 0; row < Width; ++row) {
        for (int col = 0; col < Width; ++col) {
            float Pvalue = 0;
            for (int k = 0; k < Width; ++k) {
                Pvalue += M[row*Width + k] * N[k*Width + col];
            }
            P[row*Width + col] = Pvalue;
        }
    }
}

bool allclose(float* A, float* B, int Width, float rtol=1e-5, float atol=1e-8) {
    for (int i = 0; i < Width * Width; ++i) {
        float diff = fabs(A[i] - B[i]);
        float tol = atol + rtol * fabs(B[i]);
        if (diff > tol) {
            printf("Arrays differ at index %d: %f != %f\n", i, A[i], B[i]);
            return false;
        }
    }
    return true;
}

size_t calculate_approprate_SM_usage(size_t total_shared_mem, int block_x, int block_y) {
    // Calculate size needed for one tile of shared memory (2 tiles needed total)
    size_t single_tile_size = block_x * block_y * sizeof(float);
    
    // We need two tiles (Mds and Nds), so double the size
    size_t required_size = 2 * single_tile_size;
    
    // Check if we have enough shared memory
    if (required_size > total_shared_mem) {
        printf("Warning: Required shared memory (%lu bytes) exceeds available shared memory (%lu bytes)\n",
               required_size, total_shared_mem);
        // Return the maximum possible size that fits in shared memory
        return (total_shared_mem / 2) & ~(sizeof(float) - 1); // Ensure alignment
    }
    
    return required_size;
}


void callCudaMatmul(float* h_M, float* h_N, float* h_P, int Width, char* order) {
    float *d_M, *d_N, *d_P;
    int nBytes = Width * Width * sizeof(float);

    checkCudaErrors(cudaMalloc(&d_M, nBytes));
    checkCudaErrors(cudaMalloc(&d_N, nBytes));
    checkCudaErrors(cudaMalloc(&d_P, nBytes));

    checkCudaErrors(cudaMemcpy(d_M, h_M, nBytes, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_N, h_N, nBytes, cudaMemcpyHostToDevice));


    if (strcmp(order, "row") == 0) {
        dim3 block(1024, 1);
        dim3 grid((Width + block.x - 1) / block.x, 1);

        MatrixMulKernel_roworder<<<grid, block>>>(d_M, d_N, d_P, Width);
    } else if (strcmp(order, "tiled") == 0) {
        dim3 block(TILE_WIDTH, TILE_WIDTH);
        dim3 grid((Width + block.x - 1) / block.x, (Width + block.y - 1) / block.y);

        MatrixMulKernel_tiled<<<grid, block>>>(d_M, d_N, d_P, Width);
    } else if (strcmp(order, "tiled_dynamic") == 0) {
        dim3 block(TILE_WIDTH, TILE_WIDTH);
        dim3 grid((Width + block.x - 1) / block.x, (Width + block.y - 1) / block.y);

        cudaDeviceProp devProp;
        cudaGetDeviceProperties(&devProp, 0);  // 0 is the device ID
        size_t size = calculate_approprate_SM_usage(devProp.sharedMemPerBlock, block.x, block.y);
        MatrixMulKernel_tiled_dynamic<<<grid, block, size>>>(d_M, d_N, d_P, Width, size/2, size/2);
    } else if (strcmp(order, "col") == 0) {
        dim3 block(1, 1024);
        dim3 grid(1, (Width + block.y - 1) / block.y);

        MatrixMulKernel_colorder<<<grid, block>>>(d_M, d_N, d_P, Width);
    } else if (strcmp(order, "thread") == 0) {
        dim3 block(32, 32);
        dim3 grid((Width + block.x - 1) / block.x, (Width + block.y - 1) / block.y);

        MatrixMulKernel<<<grid, block>>>(d_M, d_N, d_P, Width);
    } else if (strcmp(order, "thread_coarsened") == 0) {
        dim3 block(16, 16);
        dim3 grid((Width + block.x - 1) / block.x, (Width + block.y - 1) / block.y);

        MatrixMulKernel_thread_coarsened<<<grid, block>>>(d_M, d_N, d_P, Width);
    } else {
        printf("Invalid order: %s\n", order);
        exit(EXIT_FAILURE);
    }
    
    // Check for kernel launch errors
    checkCudaErrors(cudaGetLastError());
    // Wait for kernel to complete and check for errors
    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaMemcpy(h_P, d_P, nBytes, cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaFree(d_M));
    checkCudaErrors(cudaFree(d_N));
    checkCudaErrors(cudaFree(d_P));
}


int main(int argc, char** argv)
{
    int Width = 1024;
    int nBytes = Width * Width * sizeof(float);
    float *h_M = (float*)malloc(nBytes);
    float *h_N = (float*)malloc(nBytes);
    float *h_P = (float*)malloc(nBytes);
    float *h_P_ref = (float*)malloc(nBytes);

    srand(time(NULL));
    for (int i = 0; i < Width * Width; ++i) {
        h_M[i] = rand() / (float)RAND_MAX;
        h_N[i] = rand() / (float)RAND_MAX;
    }

    MatrixMulCPU(h_M, h_N, h_P_ref, Width);

    callCudaMatmul(h_M, h_N, h_P, Width, "thread");
    if (allclose(h_P, h_P_ref, Width)) {
        printf("Results are close!\n");
    } else {
        printf("Results are NOT close!\n");
    }
    callCudaMatmul(h_M, h_N, h_P, Width, "row");
    if (allclose(h_P, h_P_ref, Width)) {
        printf("Results are close!\n");
    } else {
        printf("Results are NOT close!\n");
    }
    callCudaMatmul(h_M, h_N, h_P, Width, "col");
    if (allclose(h_P, h_P_ref, Width)) {
        printf("Results are close!\n");
    } else {
        printf("Results are NOT close!\n");
    }
    callCudaMatmul(h_M, h_N, h_P, Width, "tiled");
    if (allclose(h_P, h_P_ref, Width)) {
        printf("Results are close!\n");
    } else {
        printf("Results are NOT close!\n");
    }
    callCudaMatmul(h_M, h_N, h_P, Width, "tiled_dynamic");
    if (allclose(h_P, h_P_ref, Width)) {
        printf("Results are close!\n");
    } else {
        printf("Results are NOT close!\n");
    }
    callCudaMatmul(h_M, h_N, h_P, Width, "thread_coarsened");
    if (allclose(h_P, h_P_ref, Width)) {
        printf("Results are close!\n");
    } else {
        printf("Results are NOT close!\n");
    }

    free(h_M);
    free(h_N);
    free(h_P);


    return 0;
}