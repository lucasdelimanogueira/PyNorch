#include "tensor.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void add_tensor_cpu(Tensor* tensor1, Tensor* tensor2, float* result_data) {
    result_data = (float*)malloc(tensor1->size * sizeof(float));

    if (result_data == NULL) {
        fprintf(stderr, "Memory allocation failed\n");
        exit(1);
    }

    for (int i = 0; i < tensor1->size; i++) {
        result_data[i] = tensor1->data[i] + tensor2->data[i];
    }
}
