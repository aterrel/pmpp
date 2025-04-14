#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <math.h>

#define STENCIL_POINTS 7
__constant__ float c[STENCIL_POINTS];

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

__global__ void stencil_kernel(float *in, float *out, unsigned int N) {
    unsigned int i = blockIdx.z * blockDim.z + threadIdx.z;
    unsigned int j = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int k = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= 1 && i < N-1 && j >= 1 && j < N-1 && k >= 1 && k < N-1) {
        out[i*N*N + j*N + k] = c[0]*in[i*N*N + j*N +k]
                             + c[1]*in[i*N*N + j*N + (k - 1)]
                             + c[2]*in[i*N*N + j*N + (k + 1)]
                             + c[3]*in[i*N*N + (j-1)*N +k]   
                             + c[4]*in[i*N*N + (j+1)*N +k]
                             + c[5]*in[(i-1)*N*N + j*N +k]
                             + c[6]*in[(i+1)*N*N + j*N +k];
    }
}

int main() {
    int N = 128;
    float h = 1.0 / (N -1);
    int nBytes = N * N * N * sizeof(float);
    float *h_in = (float*)malloc(nBytes);
    float *h_out = (float*)malloc(nBytes);

    float h_2 = h*h;
    float h_c[STENCIL_POINTS] = {
        -6.0f / h_2,
        1.0f / h_2,
        1.0f / h_2,
        1.0f / h_2,
        1.0f / h_2,
        1.0f / h_2,
        1.0f / h_2,        
    };

    srand(time(NULL));
    for (int i = 0; i < N*N*N; ++i) {
        float x = h * (i % N);
        float y = h * ((i / N) % N);
        float z = h * ((i / (N*N)) % N);
        float noise = .05*h*(rand() / (float)RAND_MAX);
        float signal = sin(x) * y * z;
        h_in[i] = signal + noise; 
    }

    float *d_in, *d_out;
    checkCudaErrors(cudaMalloc(&d_in, nBytes));
    checkCudaErrors(cudaMalloc(&d_out, nBytes));

    checkCudaErrors(cudaMemcpy(d_in, h_in, nBytes, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpyToSymbol(c, h_c, STENCIL_POINTS * sizeof(float)));

    dim3 block(8,8,8);
    dim3 grid((N + block.x - 1) / block.x, (N + block.y - 1) / block.y, (N + block.z - 1) / block.z);

    stencil_kernel<<<grid, block>>>(d_in, d_out, N);
     // Check for kernel launch errors
    checkCudaErrors(cudaGetLastError());
    // Wait for kernel to complete and check for errors
    checkCudaErrors(cudaDeviceSynchronize());   

}