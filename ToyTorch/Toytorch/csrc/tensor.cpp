#include<stdio.h>
#include<stdlib.h>
#include<string.h>
#include<math.h>


typedef struct{
    float* data;
    int *strides;
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
    for(int i=0; i<ndim; i++){
        tensor->size *= shape[i];
    }

    tensor->strides = (int*)malloc(ndim * sizeof(int));
    if(tensor->strides == NULL){
        fprintf(stderr, "Memory allocation failed\n");
        exit(1);
    }
    int stride = 1;
    for(int i=ndim-1; i>=0; i--){
        tensor->strides[i] = stride;
        stride *= shape[i];
    }
    return tensor;
}

// In order to access some element, we can take advantage of strides, as we learned before:
float get_item(Tensor* tensor, int* indices){
    int index = 0;
    for(int i=0; i<tensor->ndim; i++){
        index+= indices[i] * tensor->strides[i];
    }
    float result;
    result = tensor->data[index];

    return result;
}

Tensor* add_tensor(Tensor* tensor1, Tensor* tensor2){
    if(tensor1->ndim != tensor2->ndim){
        fprintf(stderr, "Tensors must have the same number of dimensions  %d and %d for addition\n", tensor1->ndim, tensor2->ndim);
        exit(1);
    }

    int ndim = tensor1->ndim;
    int* shape = (int*)malloc(ndim * sizeof(int));

    if (shape == NULL){
        fprintf(stderr, "Memory allocation failed\n");
        exit(1);
    }
}

