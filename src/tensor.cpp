#include <stdio.h>
#include <stdlib.h>
#include "tensor.h"

extern "C" {

    Tensor *create_tensor(float *data, int *shape, int ndim) {
        Tensor *tensor = (Tensor *)malloc(sizeof(Tensor));
        if (tensor == NULL) {
            fprintf(stderr, "Memory allocation failed\n");
            exit(1);
        }
        tensor->data = data;
        tensor->shape = shape;
        tensor->ndim = ndim;

        tensor->strides = (int *)malloc(ndim * sizeof(int));
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
        printf("]\n");

        
        return tensor;
    }

    float get_item(Tensor *tensor, int *indices) {
        int index = 0;
        for (int i = 0; i < tensor->ndim; i++) {
            index += indices[i] * tensor->strides[i];
        }
        return tensor->data[index];
    }

    void free_tensor(Tensor *tensor) {
        free(tensor->data);
        free(tensor->shape);
        free(tensor->strides);
        free(tensor);
    }
}
