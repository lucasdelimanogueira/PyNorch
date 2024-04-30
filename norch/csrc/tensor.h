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
    Tensor* sum_tensor(Tensor* tensor);
    Tensor* sub_tensor(Tensor* tensor1, Tensor* tensor2);
    Tensor* elementwise_mul_tensor(Tensor* tensor1, Tensor* tensor2);
    Tensor* scalar_mul_tensor(Tensor* tensor, float scalar);
    void reshape_tensor(Tensor* tensor, int* new_shape, int new_ndim);
    Tensor* matmul_tensor(Tensor* tensor1, Tensor* tensor2);
    Tensor* pow_tensor(Tensor* tensor, float power);
    void to_device(Tensor* tensor, char* device);
}

#endif /* TENSOR_H */
