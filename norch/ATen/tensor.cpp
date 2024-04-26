#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <iostream>
#include <memory>
#include "tensor.h"

namespace py = pybind11;

void create_tensor(std::vector<float> data, std::vector<int> shape, int ndim) {
    
    Tensor* tensor = new Tensor;
    tensor->data = data.data();
    tensor->shape = shape.data();
    tensor->ndim = ndim;

    tensor->strides = new int[ndim];
    int stride = 1;
    for (int i = ndim - 1; i >= 0; i--) {
        tensor->strides[i] = stride;
        stride *= shape[i];
    }
}

float get_item(Tensor* tensor, int* indices) {
    int index = 0;
    for (int i = 0; i < tensor->ndim; i++) {
        index += indices[i] * tensor->strides[i];
    }
    return tensor->data[index];
}

/*Tensor* add(Tensor* tensor1, Tensor* tensor2) {
    if (tensor1->ndim != tensor2->ndim) {
        throw std::runtime_error("Error: Tensors must have the same number of dimensions");
    }
    for (int i = 0; i < tensor1->ndim; i++) {
        if (tensor1->shape[i] != tensor2->shape[i]) {
            throw std::runtime_error("Error: Tensors must have the same shape");
        }
    }

    int num_elements = 1;
    for (int i = 0; i < tensor1->ndim; i++) {
        num_elements *= tensor1->shape[i];
    }

    std::vector<float> result_data(num_elements);
    for (int i = 0; i < num_elements; i++) {
        result_data[i] = tensor1->data[i] + tensor2->data[i];
    }

    Tensor* result = create_tensor(result_data, tensor1->shape, tensor1->ndim);
    return result;
}*/

PYBIND11_MODULE(norch_C, m) {
    m.doc() = "norch_C plugin";

    m.def("create_tensor", &create_tensor, "Create tensor from data, shape, and ndim");
    m.def("get_item", &get_item, "Get item from tensor");
    //m.def("add", &add, "Add two tensors");
}
