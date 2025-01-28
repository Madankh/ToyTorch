#include<stdio.h>
#include<stdlib.h>
#include<cuda_runtime.h>

__global__ void ReLU(float *Z, float *A, int size){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < size){
        A[idx] = fmaxf(0.0f, Z[idx]);
    }
}

__global__ void ReLUBackward(float *dZ, float *dA, int size){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < size){
        dZ[idx] = dA[idx] > 0 ? dA[idx] : 0.0f;
    }

