#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        printf("CUDA error: %s\n", cudaGetErrorString(err)); \
        exit(1); \
    } \
}

__global__ void Conv2d(
    float *input,
    float *kernel,
    float *output,
    int width,
    int height,
    int inchannels,
    int outchannels,
    int kernelSize,
    int batchSize) {
    
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int outChannel = blockIdx.z % outchannels;
    int batchIdx = blockIdx.z / outchannels;
    
    if(x < width && y < height && outChannel < outchannels && batchIdx < batchSize) {
        float sum = 0.0f;
        int halfkernel = kernelSize / 2;
        
        for(int inChannel = 0; inChannel < inchannels; inChannel++) {
            for(int ky = -halfkernel; ky <= halfkernel; ky++) {
                for(int kx = -halfkernel; kx <= halfkernel; kx++) {
                    int ix = x + kx;
                    int iy = y + ky;
                    
                    float pixelValue = (ix >= 0 && ix < width && iy >= 0 && iy < height)
                        ? input[((batchIdx * inchannels + inChannel) * height + iy) * width + ix] 
                        : 0.0f;
                    
                    int kernelIdx = ((outChannel * inchannels + inChannel) * kernelSize + (ky + halfkernel)) * kernelSize + (kx + halfkernel);
                    sum += pixelValue * kernel[kernelIdx];
                }
            }
        }
        
        int outputIdx = ((batchIdx * outchannels + outChannel) * height + y) * width + x;
        output[outputIdx] = sum;
    }
}

int main() {
    // Problem dimensions
    const int width = 4;
    const int height = 4;
    const int kernel_size = 3;
    const int inchannel = 1;
    const int outchannel = 1;
    const int batch_size = 1;
    const int input_size = width * height * inchannel * batch_size;
    const int out_size = width * height * outchannel * batch_size;
    const int kernelFilter = kernel_size * kernel_size * inchannel * outchannel;
    
    // Input and kernel data
    float input_values[] = {
        1, 2, 3, 4,
        5, 6, 7, 8,
        9, 10, 11, 12,
        13, 14, 15, 16,
    };
    
    float kernel_values[] = {
        1, 2, 3,
        4, 5, 6,
        7, 8, 9,
    };
    
    // Timing variables
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));
    float milliseconds = 0;
    
    // Start timing host-to-device transfer
    CHECK_CUDA(cudaEventRecord(start));
    
    // Allocate device memory
    float *d_input, *d_kernel, *d_output;
    CHECK_CUDA(cudaMalloc(&d_input, input_size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_kernel, kernelFilter * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_output, out_size * sizeof(float)));
    
    // Copy data to device
    CHECK_CUDA(cudaMemcpy(d_input, input_values, input_size * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_kernel, kernel_values, kernelFilter * sizeof(float), cudaMemcpyHostToDevice));
    
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));
    CHECK_CUDA(cudaEventElapsedTime(&milliseconds, start, stop));
    printf("Host to Device Transfer Time: %.3f ms\n", milliseconds);
    
    // Kernel launch configuration
    dim3 blockSize(4, 4);
    dim3 gridSize(
        (width + blockSize.x - 1) / blockSize.x,
        (height + blockSize.y - 1) / blockSize.y,
        outchannel * batch_size
    );
    
    // Warmup runs
    const int warmupRuns = 5;
    printf("Performing %d warmup runs...\n", warmupRuns);
    for(int i = 0; i < warmupRuns; i++) {
        Conv2d<<<gridSize, blockSize>>>(
            d_input, d_kernel, d_output,
            width, height, inchannel, outchannel,
            kernel_size, batch_size
        );
    }
    CHECK_CUDA(cudaDeviceSynchronize());
    
    // Benchmark runs
    const int benchmarkRuns = 20;
    float totalTime = 0.0f;
    
    printf("Performing %d benchmark runs...\n", benchmarkRuns);
    for(int i = 0; i < benchmarkRuns; i++) {
        CHECK_CUDA(cudaEventRecord(start));
        
        Conv2d<<<gridSize, blockSize>>>(
            d_input, d_kernel, d_output,
            width, height, inchannel, outchannel,
            kernel_size, batch_size
        );
        
        CHECK_CUDA(cudaEventRecord(stop));
        CHECK_CUDA(cudaEventSynchronize(stop));
        CHECK_CUDA(cudaEventElapsedTime(&milliseconds, start, stop));
        totalTime += milliseconds;
    }
    
    float averageTime = totalTime / benchmarkRuns;
    printf("Average Kernel Execution Time: %.3f ms\n", averageTime);
    
    // Time device-to-host transfer
    float *h_output = (float *)malloc(out_size * sizeof(float));
    CHECK_CUDA(cudaEventRecord(start));
    CHECK_CUDA(cudaMemcpy(h_output, d_output, out_size * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));
    CHECK_CUDA(cudaEventElapsedTime(&milliseconds, start, stop));
    printf("Device to Host Transfer Time: %.3f ms\n", milliseconds);
    
    // Cleanup
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    CHECK_CUDA(cudaFree(d_input));
    CHECK_CUDA(cudaFree(d_kernel));
    CHECK_CUDA(cudaFree(d_output));
    free(h_output);
    
    return 0;
}