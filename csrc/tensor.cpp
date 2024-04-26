#include <stdio.h>
#include <stdlib.h>
#include "tensor.h"

extern "C" {

    Tensor* create_tensor(float* data, int* shape, int ndim) {
        Tensor* tensor = (Tensor*)malloc(sizeof(Tensor));
        if (tensor == NULL) {
            fprintf(stderr, "Memory allocation failed\n");
            exit(1);
        }
        tensor->data = data;
        tensor->shape = shape;
        tensor->ndim = ndim;
        tensor->size = 1;

        for (int i = 0; i < ndim; i++) {
            tensor->size *= shape[i];
        }

        tensor->strides = (int*)malloc(ndim * sizeof(int));
        if (tensor->strides == NULL) {
            fprintf(stderr, "Memory allocation failed\n");
            exit(1);
        }
        int stride = 1;
        for (int i = ndim - 1; i >= 0; i--) {
            tensor->strides[i] = stride;
            stride *= shape[i];
        }

        printf("Tensor created successfully\n");
        printf("Tensor information:\n");
        printf("Number of dimensions: %d\n", tensor->ndim);
        printf("Number size: %d\n", tensor->size);
        printf("Shape: [");
        for (int i = 0; i < ndim; i++) {
            printf("%d", tensor->shape[i]);
            if (i < ndim - 1) {
                printf(", ");
            }
        }
        printf("]\n");

        printf("Data:\n[");
        for (int i = 0; i < stride; i++) {
            printf("%.2f", tensor->data[i]);
            if (i < stride - 1) {
                printf(", ");
            }
        }
        printf("]\n\n\n");

        
        return tensor;
    }

    float get_item(Tensor* tensor, int* indices) {
        int index = 0;
        for (int i = 0; i < tensor->ndim; i++) {
            index += indices[i] * tensor->strides[i];
        }
        return tensor->data[index];
    }

    Tensor* add_tensor(Tensor* tensor1, Tensor* tensor2) {
        if (tensor1->ndim != tensor2->ndim) {
            fprintf(stderr, "Tensors must have the same number of dimensions %d and %d for addition\n", tensor1->ndim, tensor2->ndim);
            exit(1);
        }

        int ndim = tensor1->ndim;
        int* shape = (int*)malloc(ndim * sizeof(int));
        if (shape == NULL) {
            fprintf(stderr, "Memory allocation failed\n");
            exit(1);
        }

        printf("Size: %d\n", tensor1->size);
        printf("Data: [");
        for (int i = 0; i < tensor1->size; i++) {
            printf("%.2f", tensor1->data[i]);
            if (i < tensor1->size - 1) {
                printf(", ");
            }
        }
        printf("]\n");

        printf("Size: %d\n", tensor2->size);
        printf("Data: [");
        for (int i = 0; i < tensor2->size; i++) {
            printf("%.2f", tensor2->data[i]);
            if (i < tensor2->size - 1) {
                printf(", ");
            }
        }
        printf("]\n");

        printf("Shapes : [");
        for (int i = 0; i < tensor1->ndim; i++) {
            printf("%d", tensor1->shape[i]);
            if (i < tensor1->ndim - 1) {
                printf(", ");
            }
        }
        printf("]\n");

        printf("Shapes : [");
        for (int i = 0; i < tensor2->ndim; i++) {
            printf("%d", tensor2->shape[i]);
            if (i < tensor2->ndim - 1) {
                printf(", ");
            }
        }
        printf("]\n");

        for (int i = 0; i < ndim; i++) {
            if (tensor1->shape[i] != tensor2->shape[i]) {
                fprintf(stderr, "Tensors must have the same shape %d and %d at index %d for addition\n", tensor1->shape[i], tensor2->shape[i], i);
                exit(1);
            }
            shape[i] = tensor1->shape[i];
        }

        float* result_data = (float*)malloc(tensor1->size * sizeof(float));
        if (result_data == NULL) {
            fprintf(stderr, "Memory allocation failed\n");
            exit(1);
        }

        for (int i = 0; i < tensor1->size; i++) {
            result_data[i] = tensor1->data[i] + tensor2->data[i];
        }

        return create_tensor(result_data, shape, ndim);
    }
}
