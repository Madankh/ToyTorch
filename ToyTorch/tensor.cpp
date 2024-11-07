#include<stdio.h>
#include<stdlib.h>
#include<string.h>
#include<math.h>


typedef struct{
    float* data;
    int strides;
    int* shape;
    int ndim;
    int size;
    char* device;
} Tensor;

Tensor* create_tensor(float* data, int* shape, int ndim){
    Tensor* tensor = (Tensor*)malloc(sizeof(Tensor));
    if(tensor == NULL){
        fprintf(stderr, "Memory allocation falied\n");
        exit(1);
    }
    tensor->data = data;
    tensor->shape = shape;
    tensor->ndim = ndim;

    tensor->size = 1;
    
}