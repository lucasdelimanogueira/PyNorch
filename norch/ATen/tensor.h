#ifndef TENSOR_H
#define TENSOR_H

struct Tensor {
    float* data;
    int* shape;
    int* strides;
    int ndim;
};

extern "C" {
    void create_tensor(std::vector<float> data, std::vector<int> shape, int ndim);
    float get_item(Tensor* tensor, int* indices);
    //Tensor* add(Tensor* tensor1, Tensor* tensor2);
}

#endif // TENSOR_H
