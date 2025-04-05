#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

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


// Compute vector sum C = A + B
// Each thread performs one pair-wize addition
__device__ __host__
void vecAddUniversalKernel(float* A, float* B, float* C, int n,int i) {
    if (i < n) {
        C[i] = A[i] + B[i];
    }
}

// Compute vector sum C = A + B
// Each thread performs one pair-wize addition
__global__
void vecAddKernel(float* A, float* B, float* C, int n) {
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    vecAddUniversalKernel(A, B, C, n,i);
}


// Compute vector sum C_h = A_h + B_h
void vecAdd(float* A_h, float* B_h, float* C_h, int n) {
    for (int i = 0; i < n; ++i) {
        vecAddUniversalKernel(A_h, B_h, C_h, n, i);
    }
}

// GPU vector addition kernel
void vecAddGPU(float* A_h, float *B_h, float *C_h, int n) {
    int size = n * sizeof(float);
    float *A_d, *B_d, *C_d;

    // Part 1: Allocate device memory for A, B, and C
    // Copy A and B to device memory
    checkCudaErrors(cudaMalloc((void**) &A_d, size));
    checkCudaErrors(cudaMalloc((void**) &B_d, size));
    checkCudaErrors(cudaMalloc((void**) &C_d, size));

    checkCudaErrors(cudaMemcpy(A_d, A_h, size, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(B_d, B_h, size, cudaMemcpyHostToDevice));

    // Part 2: Call kernel - to launch a grid of threads
    // to perform the actual vecotr addition
    vecAddKernel<<<ceil(n/256.0), 256>>>(A_d, B_d, C_d, n);
    
    // Check for kernel launch errors
    checkCudaErrors(cudaGetLastError());
    // Wait for kernel to complete and check for errors
    checkCudaErrors(cudaDeviceSynchronize());

    // Part 3: Copy C from the device memory
    // Free device memory

    checkCudaErrors(cudaMemcpy(C_h, C_d, size, cudaMemcpyDeviceToHost));

    checkCudaErrors(cudaFree(A_d));
    checkCudaErrors(cudaFree(B_d));
    checkCudaErrors(cudaFree(C_d));
}

double sum(float* A, int n) {
    double sum = 0;
    for (int i = 0; i < n; ++i) {
        sum += A[i];
    }
    return sum;
}

int main(int argc, char* argv[]) {
    int N = 1024;
    // Memory allocation for arrays A, B, and C
    float* A_h = (float*)malloc(N * sizeof(float));
    float* B_h = (float*)malloc(N * sizeof(float));
    float* C_h = (float*)malloc(N * sizeof(float));

    // Initialize arrays A and B
    for (int i = 0; i < N; ++i) {
        A_h[i] = i;
        B_h[i] = i;
    }

    // Compute C = A + B on the host
    vecAdd(A_h, B_h, C_h, N);
    printf("cpu: sum(C_h) = %f\n", sum(C_h, N));
    // Compute C = A + B on the GPU
    vecAddGPU(A_h, B_h, C_h, N);
    printf("gpu: sum(C_h) = %f\n", sum(C_h, N));

    // Free allocated memory
    free(A_h);
    free(B_h);
    free(C_h);

    return 0;
}