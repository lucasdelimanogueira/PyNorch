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

void scalar_mul_tensor_cpu(Tensor* tensor, float scalar, float* result_data) {
    
    for (int i = 0; i < tensor->size; i++) {
        result_data[i] = scalar * tensor->data[i];
    }
}

void scalar_div_tensor_cpu(float scalar, Tensor* tensor, float* result_data) {
    
    for (int i = 0; i < tensor->size; i++) {
        result_data[i] = scalar / tensor->data[i];
    }
}

void tensor_div_scalar_cpu(Tensor* tensor, float scalar, float* result_data) {
    
    for (int i = 0; i < tensor->size; i++) {
        result_data[i] = tensor->data[i] / scalar;
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


void broadcasted_batched_matmul_tensor_cpu(Tensor* tensor1, Tensor* tensor2, float* result_data) {

    int tensor2_offset = tensor2->shape[1] * tensor2->shape[2];
    int result_data_offset = tensor1->shape[0] * tensor2->shape[2];

    for (int batch = 0; batch < tensor2->shape[0]; batch++) {
    
        for (int i = 0; i < tensor1->shape[0]; i++) {
            for (int j = 0; j < tensor2->shape[2]; j++) {
                float sum = 0.0;
                for (int k = 0; k < tensor1->shape[1]; k++) {
                    sum += tensor1->data[i * tensor1->shape[1] + k] * tensor2->data[batch*tensor2_offset + (k * tensor2->shape[2] + j)];
                }
                result_data[(batch * result_data_offset) + (i * tensor2->shape[2] + j)] = sum;
            }
        }
    }
} 

void batched_matmul_tensor_cpu(Tensor* tensor1, Tensor* tensor2, float* result_data) {

    int tensor1_offset = tensor1->shape[1] * tensor1->shape[2];
    int tensor2_offset = tensor2->shape[1] * tensor2->shape[2];
    int result_data_offset = tensor1->shape[1] * tensor2->shape[2];

    for (int batch = 0; batch < tensor2->shape[0]; batch++) {
    
        for (int i = 0; i < tensor1->shape[1]; i++) {
            for (int j = 0; j < tensor2->shape[2]; j++) {
                float sum = 0.0;
                for (int k = 0; k < tensor1->shape[2]; k++) {
                    sum += tensor1->data[(batch * tensor1_offset) + i * tensor1->shape[2] + k] * tensor2->data[batch*tensor2_offset + (k * tensor2->shape[2] + j)];
                }
                result_data[(batch * result_data_offset) + (i * tensor2->shape[2] + j)] = sum;
            }
        }
    }
} 

void scalar_pow_tensor_cpu(float base, Tensor* tensor, float* result_data) {
    
    for (int i = 0; i < tensor->size; i++) {
        result_data[i] = powf(base, tensor->data[i]);
    }
}

void tensor_pow_scalar_cpu(Tensor* tensor, float exponent, float* result_data) {
    
    for (int i = 0; i < tensor->size; i++) {
        result_data[i] = powf(tensor->data[i], exponent);
    }
}

void log_tensor_cpu(Tensor* tensor, float* result_data) {
    for (int i = 0; i < tensor->size; i++) {
        result_data[i] = logf(tensor->data[i]);
    }
}

void sum_tensor_cpu(Tensor* tensor, float* result_data) {
    float sum = 0.0;

    for (int i = 0; i < tensor->size; i++) {
        sum += tensor->data[i];
    }

    *result_data = sum;
}

void ones_like_tensor_cpu(Tensor* tensor, float* result_data) {
    
    for (int i = 0; i < tensor->size; i++) {
        result_data[i] = 1.0;
    }
}

void zeros_like_tensor_cpu(Tensor* tensor, float* result_data) {
    
    for (int i = 0; i < tensor->size; i++) {
        result_data[i] = 0.0;
    }
}

void transpose_1D_tensor_cpu(Tensor* tensor, float* result_data) {

    for (int i = 0; i < tensor->shape[0]; i++) {
        result_data[i] = tensor->data[i];
    }
}

void transpose_2D_tensor_cpu(Tensor* tensor, float* result_data) {
    int rows = tensor->shape[0];
    int cols = tensor->shape[1];

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            result_data[j * rows + i] = tensor->data[i * cols + j];
        }
    }
}

void transpose_3D_tensor_cpu(Tensor* tensor, float* result_data) {
    int depth = tensor->shape[0];
    int rows = tensor->shape[1];
    int cols = tensor->shape[2];

    for (int i = 0; i < depth; i++) {
        for (int j = 0; j < rows; j++) {
            for (int k = 0; k < cols; k++) {
                result_data[k * rows * depth + j * depth + i] = tensor->data[i * rows * cols + j * cols + k];
            }
        }
    }
}

void assign_tensor_cpu(Tensor* tensor, float* result_data) {

    for (int i = 0; i < tensor->size; i++) {
        result_data[i] = tensor->data[i];
    }
}
