#ifndef TENSOR_H
#define TENSOR_H

typedef struct {
    float *data;
    int *strides;
    int *shape;
    int ndim;
} Tensor;

extern "C" {
    Tensor *create_tensor(float *data, int *shape, int ndim);
    float get_item(Tensor *tensor, int *indices);
    void delete_tensor(Tensor* tensor);
}

#endif /* TENSOR_H */
