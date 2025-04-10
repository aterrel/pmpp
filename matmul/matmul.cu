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

#define TILE_WIDTH 16
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
    } else if (strcmp(order, "col") == 0) {
        dim3 block(1, 1024);
        dim3 grid(1, (Width + block.y - 1) / block.y);

        MatrixMulKernel_colorder<<<grid, block>>>(d_M, d_N, d_P, Width);
    } else if (strcmp(order, "thread") == 0) {
        dim3 block(32, 32);
        dim3 grid((Width + block.x - 1) / block.x, (Width + block.y - 1) / block.y);

        MatrixMulKernel<<<grid, block>>>(d_M, d_N, d_P, Width);
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

    free(h_M);
    free(h_N);
    free(h_P);


    return 0;
}