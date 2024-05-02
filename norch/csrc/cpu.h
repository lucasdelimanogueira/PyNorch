#ifndef CPU_H
#define CPU_H

#include "tensor.h"

void add_tensor_cpu(Tensor* tensor1, Tensor* tensor2, float* result_data);
void sum_tensor_cpu(Tensor* tensor1, float* result_data);
void sub_tensor_cpu(Tensor* tensor1, Tensor* tensor2, float* result_data);
void elementwise_mul_tensor_cpu(Tensor* tensor1, Tensor* tensor2, float* result_data);
void matmul_tensor_cpu(Tensor* tensor1, Tensor* tensor2, float* result_data);
void batched_matmul_tensor_cpu(Tensor* tensor1, Tensor* tensor2, float* result_data);
void pow_tensor_cpu(Tensor* tensor, float power, float* result_data);
void scalar_mul_tensor_cpu(Tensor* tensor, float scalar, float* result_data);
void ones_like_tensor_cpu(Tensor* tensor, float* result_data);
void zeros_like_tensor_cpu(Tensor* tensor, float* result_data);
void transpose_tensor_cpu(Tensor* tensor, float* result_data);
void assign_tensor_cpu(Tensor* tensor, float* result_data);

#endif /* CPU_H */
