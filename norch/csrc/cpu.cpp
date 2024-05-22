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

void add_broadcasted_tensor_cpu(Tensor* tensor1, Tensor* tensor2, float* result_data, int* broadcasted_shape, int broadcasted_size) {
    int max_ndim = tensor1->ndim > tensor2->ndim ? tensor1->ndim : tensor2->ndim;

    // Calculate strides for broadcasting
    int* strides1 = (int*)malloc(max_ndim * sizeof(int));
    int* strides2 = (int*)malloc(max_ndim * sizeof(int));
    if (strides1 == NULL || strides2 == NULL) {
        fprintf(stderr, "Memory allocation failed\n");
        exit(1);
    }

    int stride1 = 1, stride2 = 1;
    for (int i = max_ndim - 1; i >= 0; i--) {
        int dim1 = i < tensor1->ndim ? tensor1->shape[tensor1->ndim - max_ndim + i] : 1;
        int dim2 = i < tensor2->ndim ? tensor2->shape[tensor2->ndim - max_ndim + i] : 1;
        strides1[i] = dim1 == broadcasted_shape[i] ? stride1 : 0;
        strides2[i] = dim2 == broadcasted_shape[i] ? stride2 : 0;
        stride1 *= (dim1 == broadcasted_shape[i]) ? dim1 : 1;
        stride2 *= (dim2 == broadcasted_shape[i]) ? dim2 : 1;
    }

    // Perform element-wise addition with broadcasting
    for (int i = 0; i < broadcasted_size; i++) {
        int index1 = 0, index2 = 0;
        int linear_index = i;
        for (int j = max_ndim - 1; j >= 0; j--) {
            int pos = linear_index % broadcasted_shape[j];
            linear_index /= broadcasted_shape[j];
            if (strides1[j] != 0) index1 += pos * strides1[j];
            if (strides2[j] != 0) index2 += pos * strides2[j];
        }
        result_data[i] = tensor1->data[index1] + tensor2->data[index2];
    }

    // Free strides
    free(strides1);
    free(strides2);
}


void sub_tensor_cpu(Tensor* tensor1, Tensor* tensor2, float* result_data) {
    
    for (int i = 0; i < tensor1->size; i++) {
        result_data[i] = tensor1->data[i] - tensor2->data[i];
    }
}

void sub_broadcasted_tensor_cpu(Tensor* tensor1, Tensor* tensor2, float* result_data, int* broadcasted_shape, int broadcasted_size) {
    int max_ndim = tensor1->ndim > tensor2->ndim ? tensor1->ndim : tensor2->ndim;

    // Calculate strides for broadcasting
    int* strides1 = (int*)malloc(max_ndim * sizeof(int));
    int* strides2 = (int*)malloc(max_ndim * sizeof(int));
    if (strides1 == NULL || strides2 == NULL) {
        fprintf(stderr, "Memory allocation failed\n");
        exit(1);
    }

    int stride1 = 1, stride2 = 1;
    for (int i = max_ndim - 1; i >= 0; i--) {
        int dim1 = i < tensor1->ndim ? tensor1->shape[tensor1->ndim - max_ndim + i] : 1;
        int dim2 = i < tensor2->ndim ? tensor2->shape[tensor2->ndim - max_ndim + i] : 1;
        strides1[i] = dim1 == broadcasted_shape[i] ? stride1 : 0;
        strides2[i] = dim2 == broadcasted_shape[i] ? stride2 : 0;
        stride1 *= (dim1 == broadcasted_shape[i]) ? dim1 : 1;
        stride2 *= (dim2 == broadcasted_shape[i]) ? dim2 : 1;
    }

    // Perform element-wise addition with broadcasting
    for (int i = 0; i < broadcasted_size; i++) {
        int index1 = 0, index2 = 0;
        int linear_index = i;
        for (int j = max_ndim - 1; j >= 0; j--) {
            int pos = linear_index % broadcasted_shape[j];
            linear_index /= broadcasted_shape[j];
            if (strides1[j] != 0) index1 += pos * strides1[j];
            if (strides2[j] != 0) index2 += pos * strides2[j];
        }
        result_data[i] = tensor1->data[index1] - tensor2->data[index2];
    }

    // Free strides
    free(strides1);
    free(strides2);
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

void tensor_div_tensor_cpu(Tensor* tensor1, Tensor* tensor2, float* result_data) {
    
    for (int i = 0; i < tensor1->size; i++) {
        result_data[i] = tensor1->data[i] / tensor2->data[i];
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

void sum_tensor_cpu(Tensor* tensor, float* result_data, int size, int* result_shape, int axis) {
    if (axis == -1) {
        // Sum over all elements
        float sum = 0.0;
        for (int i = 0; i < tensor->size; i++) {
            sum += tensor->data[i];
        }
        *result_data = sum;
    } else {
        if (axis < 0 || axis >= tensor->ndim) {
            printf("Invalid axis");
            return;
        }
        
        int axis_stride = tensor->strides[axis];

        for (int i = 0; i < tensor->shape[axis]; i++) {
            for (int j = 0; j < size; j++) {
                int index = 0;
                int remainder = j;
                for (int k = tensor->ndim - 2; k >= 0; k--) {
                    index += (remainder % result_shape[k]) * tensor->strides[k < axis ? k : k + 1];     
                    remainder /= result_shape[k];
                }
                result_data[j] += tensor->data[index + i * axis_stride];
            }
        }
    }
}

void max_tensor_cpu(Tensor* tensor, float* result_data, int size, int* result_shape, int axis) {
    if (axis == -1) {
        float max_value = -INFINITY;
        for (int i = 0; i < tensor->size; i++) {
            max_value = fmax(max_value, tensor->data[i]);
        }
        *result_data = max_value;
    } else {
        for (int i = 0; i < size; i++) {
            result_data[i] = -INFINITY;
        }
        if (axis < 0 || axis >= tensor->ndim) {
            printf("Invalid axis");
            return;
        }
        
        int axis_stride = tensor->strides[axis];

        for (int i = 0; i < tensor->shape[axis]; i++) {
            for (int j = 0; j < size; j++) {
                int index = 0;
                int remainder = j;
                for (int k = tensor->ndim - 2; k >= 0; k--) {
                    index += (remainder % result_shape[k]) * tensor->strides[k < axis ? k : k + 1];     
                    remainder /= result_shape[k];
                }
                result_data[j] = fmax(result_data[j], tensor->data[index + i * axis_stride]);
            }
        }
    }
}

void min_tensor_cpu(Tensor* tensor, float* result_data, int size, int* result_shape, int axis) {
    if (axis == -1) {
        float min_value = INFINITY;
        for (int i = 0; i < tensor->size; i++) {
            min_value = fmin(min_value, tensor->data[i]);
        }
        *result_data = min_value;
    } else {
        for (int i = 0; i < size; i++) {
            result_data[i] = INFINITY;
        }
        if (axis < 0 || axis >= tensor->ndim) {
            printf("Invalid axis");
            return;
        }
        
        int axis_stride = tensor->strides[axis];

        for (int i = 0; i < tensor->shape[axis]; i++) {
            for (int j = 0; j < size; j++) {
                int index = 0;
                int remainder = j;
                for (int k = tensor->ndim - 2; k >= 0; k--) {
                    index += (remainder % result_shape[k]) * tensor->strides[k < axis ? k : k + 1];     
                    remainder /= result_shape[k];
                }
                result_data[j] = fmin(result_data[j], tensor->data[index + i * axis_stride]);
            }
        }
    }
}

void equal_tensor_cpu(Tensor* tensor1, Tensor* tensor2, float* result_data) {
    
    for (int i = 0; i < tensor1->size; i++) {
        result_data[i] = (tensor1->data[i] == tensor2->data[i]) ? 1.0f : 0.0f;
    }
}

void equal_broadcasted_tensor_cpu(Tensor* tensor1, Tensor* tensor2, float* result_data, int* broadcasted_shape, int broadcasted_size) {
    int max_ndim = tensor1->ndim > tensor2->ndim ? tensor1->ndim : tensor2->ndim;

    // Calculate strides for broadcasting
    int* strides1 = (int*)malloc(max_ndim * sizeof(int));
    int* strides2 = (int*)malloc(max_ndim * sizeof(int));
    if (strides1 == NULL || strides2 == NULL) {
        fprintf(stderr, "Memory allocation failed\n");
        exit(1);
    }

    int stride1 = 1, stride2 = 1;
    for (int i = max_ndim - 1; i >= 0; i--) {
        int dim1 = i < tensor1->ndim ? tensor1->shape[tensor1->ndim - max_ndim + i] : 1;
        int dim2 = i < tensor2->ndim ? tensor2->shape[tensor2->ndim - max_ndim + i] : 1;
        strides1[i] = dim1 == broadcasted_shape[i] ? stride1 : 0;
        strides2[i] = dim2 == broadcasted_shape[i] ? stride2 : 0;
        stride1 *= (dim1 == broadcasted_shape[i]) ? dim1 : 1;
        stride2 *= (dim2 == broadcasted_shape[i]) ? dim2 : 1;
    }

    // Perform element-wise equal with broadcasting
    for (int i = 0; i < broadcasted_size; i++) {
        int index1 = 0, index2 = 0;
        int linear_index = i;
        for (int j = max_ndim - 1; j >= 0; j--) {
            int pos = linear_index % broadcasted_shape[j];
            linear_index /= broadcasted_shape[j];
            if (strides1[j] != 0) index1 += pos * strides1[j];
            if (strides2[j] != 0) index2 += pos * strides2[j];
        }
        result_data[i] = (tensor1->data[index1] == tensor2->data[index2]) ? 1.0f : 0.0f;
    }

    // Free strides
    free(strides1);
    free(strides2);
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

void sin_tensor_cpu(Tensor* tensor, float* result_data) {
    for (int i = 0; i < tensor->size; i++) {
        result_data[i] = sinf(tensor->data[i]);
    }
}

void sigmoid_tensor_cpu(Tensor* tensor, float* result_data) {
    for (int i = 0; i < tensor->size; i++) {
        // avoid overflow
        if (tensor->data[i] >= 0) {

            float z = expf(-tensor->data[i]);
            result_data[i] = 1 / (1 + z);

        } else {

            float z = expf(tensor->data[i]);
            result_data[i] = z / (1 + z);
        }
    }
}

void cos_tensor_cpu(Tensor* tensor, float* result_data) {
    for (int i = 0; i < tensor->size; i++) {
        result_data[i] = cosf(tensor->data[i]);
    }
}
