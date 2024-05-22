#ifndef CPU_H
#define CPU_H

#include "tensor.h"

void add_tensor_cpu(Tensor* tensor1, Tensor* tensor2, float* result_data);
void add_broadcasted_tensor_cpu(Tensor* tensor1, Tensor* tensor2, float* result_data, int* broadcasted_shape, int broadcasted_size);
void sum_tensor_cpu(Tensor* tensor, float* result_data, int size, int* shape, int axis);
void max_tensor_cpu(Tensor* tensor, float* result_data, int size, int* result_shape, int axis);
void min_tensor_cpu(Tensor* tensor, float* result_data, int size, int* result_shape, int axis);
void sub_tensor_cpu(Tensor* tensor1, Tensor* tensor2, float* result_data);
void sub_broadcasted_tensor_cpu(Tensor* tensor1, Tensor* tensor2, float* result_data, int* broadcasted_shape, int broadcasted_size);
void elementwise_mul_tensor_cpu(Tensor* tensor1, Tensor* tensor2, float* result_data);
void scalar_div_tensor_cpu(float scalar, Tensor* tensor, float* result_data);
void tensor_div_scalar_cpu(Tensor* tensor, float scalar, float* result_data);
void tensor_div_tensor_cpu(Tensor* tensor1, Tensor* tensor2, float* result_data);
void matmul_tensor_cpu(Tensor* tensor1, Tensor* tensor2, float* result_data);
void broadcasted_batched_matmul_tensor_cpu(Tensor* tensor1, Tensor* tensor2, float* result_data);
void batched_matmul_tensor_cpu(Tensor* tensor1, Tensor* tensor2, float* result_data);
void scalar_pow_tensor_cpu(float base, Tensor* tensor, float* result_data);
void tensor_pow_scalar_cpu(Tensor* tensor, float exponent, float* result_data);
void log_tensor_cpu(Tensor* tensor, float* result_data);
void scalar_mul_tensor_cpu(Tensor* tensor, float scalar, float* result_data);
void equal_tensor_cpu(Tensor* tensor1, Tensor* tensor2, float* result_data);
void equal_broadcasted_tensor_cpu(Tensor* tensor1, Tensor* tensor2, float* result_data, int* broadcasted_shape, int broadcasted_size);
void ones_like_tensor_cpu(Tensor* tensor, float* result_data);
void zeros_like_tensor_cpu(Tensor* tensor, float* result_data);
void transpose_1D_tensor_cpu(Tensor* tensor, float* result_data);
void transpose_2D_tensor_cpu(Tensor* tensor, float* result_data);
void transpose_3D_tensor_cpu(Tensor* tensor, float* result_data);
void transpose_axes_cpu(Tensor* tensor, float* result_data, int axis1, int axis2, int* new_shape);
void assign_tensor_cpu(Tensor* tensor, float* result_data);
void sin_tensor_cpu(Tensor* tensor, float* result_data);
void cos_tensor_cpu(Tensor* tensor, float* result_data);
void sigmoid_tensor_cpu(Tensor* tensor, float* result_data);

#endif /* CPU_H */
