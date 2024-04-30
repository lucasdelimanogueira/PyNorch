#include "tensor.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

void add_tensor_cpu(Tensor* tensor1, Tensor* tensor2, float* result_data) {
    
    for (int i = 0; i < tensor1->size; i++) {
        result_data[i] = tensor1->data[i] + tensor2->data[i];
    }
}

void sub_tensor_cpu(Tensor* tensor1, Tensor* tensor2, float* result_data) {
    
    for (int i = 0; i < tensor1->size; i++) {
        result_data[i] = tensor1->data[i] - tensor2->data[i];
    }
}

void elementwise_mul_tensor_cpu(Tensor* tensor1, Tensor* tensor2, float* result_data) {
    
    for (int i = 0; i < tensor1->size; i++) {
        result_data[i] = tensor1->data[i] * tensor2->data[i];
    }
}

void matmul_tensor_cpu(Tensor* tensor1, Tensor* tensor2, float* result_data) {
    
    for (int i = 0; i < tensor1->shape[0]; i++) {
        for (int j = 0; j < tensor2->shape[1]; j++) {
            float sum = 0.0;
            for (int k = 0; k < tensor1->shape[1]; k++) {
                sum += tensor1->data[i * tensor1->shape[1] + k] * tensor2->data[k * tensor2->shape[1] + j];
            }
            result_data[i * tensor2->shape[1] + j] = sum;
        }
    }
}

void pow_tensor_cpu(Tensor* tensor, float power, float* result_data) {
    
    for (int i = 0; i < tensor->size; i++) {
        result_data[i] = powf(tensor->data[i], power);
    }
}
