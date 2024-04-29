#include "tensor.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void add_tensor_cpu(Tensor* tensor1, Tensor* tensor2, float* result_data) {
    
    for (int i = 0; i < tensor1->size; i++) {
        result_data[i] = tensor1->data[i] + tensor2->data[i];
    }
}
