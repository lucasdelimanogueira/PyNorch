#ifndef TENSOR_H
#define TENSOR_H

typedef struct {
    float* data;
    int* strides;
    int* shape;
    int* strides_cuda;
    int* shape_cuda;
    int ndim;
    int size;
    char* device;
} Tensor;

extern "C" {
    Tensor* create_tensor(float* data, int* shape, int ndim, char* device);
    float get_item(Tensor* tensor, int* indices);
    Tensor* add_tensor(Tensor* tensor1, Tensor* tensor2);
    Tensor* sum_tensor(Tensor* tensor, int axis, bool keepdims);
    Tensor* max_tensor(Tensor* tensor, int axis, bool keepdim);
    Tensor* min_tensor(Tensor* tensor, int axis, bool keepdim);
    Tensor* sub_tensor(Tensor* tensor1, Tensor* tensor2);
    Tensor* elementwise_mul_tensor(Tensor* tensor1, Tensor* tensor2);
    Tensor* scalar_mul_tensor(Tensor* tensor, float scalar);
    Tensor* scalar_div_tensor(float scalar, Tensor* tensor);
    Tensor* tensor_div_scalar(Tensor* tensor, float scalar);
    Tensor* tensor_div_tensor(Tensor* tensor1, Tensor* tensor2);
    Tensor* reshape_tensor(Tensor* tensor, int* new_shape, int new_ndim);
    Tensor* matmul_tensor(Tensor* tensor1, Tensor* tensor2);
    Tensor* tensor_pow_scalar(Tensor* tensor, float exponent);
    Tensor* scalar_pow_tensor(float base, Tensor* tensor);
    Tensor* log_tensor(Tensor* tensor);
    Tensor* equal_tensor(Tensor* tensor1, Tensor* tensor2);
    Tensor* equal_broadcasted_tensor(Tensor* tensor1, Tensor* tensor2);
    void to_device(Tensor* tensor, char* device);
    Tensor* ones_like_tensor(Tensor* tensor);
    Tensor* zeros_like_tensor(Tensor* tensor);
    Tensor* sin_tensor(Tensor* tensor);
    Tensor* cos_tensor(Tensor* tensor);
    Tensor* transpose_tensor(Tensor* tensor);
    Tensor* transpose_axes_tensor(Tensor* tensor, int axis1, int axis2);
    void make_contiguous(Tensor* tensor);
}

#endif /* TENSOR_H */
