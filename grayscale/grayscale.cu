#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <math.h>

#define STB_IMAGE_IMPLEMENTATION
#include "include/stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "include/stb_image_write.h"


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

struct CommandLineOptions {
    bool blurred;
    const char* inputFile;
    
    CommandLineOptions() : blurred(false), inputFile(nullptr) {}
};


const int CHANNELS = 3;
const int BLUR_SIZE = 5;

// The input image in encoded as unsigned char (0-255)
// Each pixel is 3 consecutive chars for the 3 channels (RGB)
__global__
void colorToGrayscaleConvertion(unsigned char* Pout, unsigned char* Pin, int width, int height) {
    int col = blockIdx.x*blockDim.x + threadIdx.x;
    int row = blockIdx.y*blockDim.y + threadIdx.y;
    if (col < width && row < height) {
        // Get Id offset for the graycale image
        int grayOffset = row*width + col;
        // One can think of the GRB image having CHANNEL
        // times more columns than the gray scale image
        int rgbOffset = grayOffset*CHANNELS;
        unsigned char r = Pin[rgbOffset]; // Red value
        unsigned char g = Pin[rgbOffset + 1]; // Green value
        unsigned char b = Pin[rgbOffset + 2]; // Blue value
        // Perform the rescaling and store it
        // We multiply by floating point constants
        Pout[grayOffset] = 0.21f*r + 0.71f*g + 0.07f*b;
    }
}

__global__
void blurKernel(unsigned char *in, unsigned char *out, int w, int h) {
    int col = blockIdx.x*blockDim.x + threadIdx.x;
    int row = blockIdx.y*blockDim.y + threadIdx.y;
    if (col < w && row < h) {
        int pixVal = 0;
        int pixels = 0;
        // Get average of the surrounding BLUR_SIZE x BLUR_SIZE box
        for (int blurRow = -BLUR_SIZE; blurRow < BLUR_SIZE + 1; ++blurRow) {
            for (int blurCol = -BLUR_SIZE; blurCol < BLUR_SIZE + 1; ++blurCol) {
                int curRow = row + blurRow;
                int curCol = col + blurCol;
                // Verify we have a valid image pixel
                if (curRow >= 0 && curRow < h && curCol >= 0 && curCol < w) {
                    pixVal += in[curRow*w + curCol];
                    ++pixels; // Keep track of the number of pixels in the avg
                }
            }
        }
        // Write out new pixel value ot
        out[row*w + col] = (unsigned char) (pixVal/pixels);
    }
}

CommandLineOptions parseCommandLine(int argc, char** argv) {
    CommandLineOptions options;
    bool invalid_cmd = false;

    if (argc == 2) { // case with no options
        options.inputFile = argv[1];
        options.blurred = false;
        printf("Performing grayscale conversion on %s\n", options.inputFile);
    }
    else if (argc == 3) { // case with options
        char* opt = argv[1];
        if (opt[1] == 'b') { // Using cmd[1] to get the character after '-'          
            options.inputFile = argv[2];
            options.blurred = true;
            printf("Performing grayscale and blur operation on %s\n", options.inputFile);
        }
        else {
            invalid_cmd = true;
        }            
    }
    else {
        invalid_cmd = true;
    }

    if (invalid_cmd) {
        printf("Usage: %s <options> <input_image>\n", argv[0]);
        printf("  options: \n");
        printf("    -b for blur\n");
        exit(1);
    }
    if (argc == 2) {
        options.inputFile = argv[1];
    }
    return options;
}

int main(int argc, char** argv) {
    CommandLineOptions options = parseCommandLine(argc, argv);

    // Load the input image
    int width, height, channels;
    unsigned char* imageData = stbi_load(options.inputFile, &width, &height, &channels, CHANNELS);
    if (imageData == NULL) {
        printf("Error loading image %s\n", argv[1]);
        exit(1);
    }

    // Allocate host memory for output grayscale image
    int grayImageSize = width * height;
    int colorImageSize = width * height * CHANNELS;
    unsigned char* h_Pout = (unsigned char*)malloc(grayImageSize);
    
    // Allocate device memory
    unsigned char *d_Pin, *d_Pout;
    checkCudaErrors(cudaMalloc((void**)&d_Pin, colorImageSize));
    checkCudaErrors(cudaMalloc((void**)&d_Pout, grayImageSize));

    // Copy input image to device
    checkCudaErrors(cudaMemcpy(d_Pin, imageData, colorImageSize, cudaMemcpyHostToDevice));

    dim3 dimGrid(ceil(width/32.0), ceil(height/32.0));
    dim3 dimBlock(32, 32);
    colorToGrayscaleConvertion<<<dimGrid, dimBlock>>>(d_Pout, d_Pin, width, height);

    if (options.blurred) {
        unsigned char *d_Pout_blurred;
        checkCudaErrors(cudaMalloc((void**)&d_Pout_blurred, grayImageSize));    
        blurKernel<<<dimGrid, dimBlock>>>(d_Pout, d_Pout_blurred, width, height);
        // Copy output image to host
        checkCudaErrors(cudaMemcpy(h_Pout, d_Pout_blurred, grayImageSize, cudaMemcpyDeviceToHost));
        checkCudaErrors(cudaFree(d_Pout_blurred));
    } else {
        // Copy output image to host
        checkCudaErrors(cudaMemcpy(h_Pout, d_Pout, grayImageSize, cudaMemcpyDeviceToHost));
    }
    checkCudaErrors(cudaFree(d_Pin));
    checkCudaErrors(cudaFree(d_Pout));

    // Save the output image
    stbi_write_png("output.jpg", width, height, 1, h_Pout, width);

    // Free allocated memory    
    free(imageData);
    free(h_Pout);

    return 0;
}   

