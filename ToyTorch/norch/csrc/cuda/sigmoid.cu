#include<stdio.h>
#include<stdlib.h>
#include<cuda_runtime.h>

__device__ float sigmoid(float x){
    return 1.0f / (1.0f + expf(-x))
}

__global__ void Sigmoid(float *Z, float *A, int size){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < size){
        A[idx] = sigmoid(-Z[idx]);
    }
}

__global__ void SigmoidBackward(float *Z , float *dA, float *dZ, int size){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < size){
        dZ[idx] = dA[idx] * sigmoid(Z[idx]) * (1 - sigmoid(Z[idx]));
    }
}

