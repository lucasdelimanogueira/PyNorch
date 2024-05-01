#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <cuda_runtime_api.h>
#include "tensor.h"
#include "cuda.h"
#include "cpu.h"

extern "C" {

    Tensor* create_tensor(float* data, int* shape, int ndim, char* device) {

        printf("Creating tensor\n");
        Tensor* tensor = (Tensor*)malloc(sizeof(Tensor));
        if (tensor == NULL) {
            fprintf(stderr, "Memory allocation failed\n");
            exit(1);
        }
        tensor->data = data;
        tensor->shape = shape;
        tensor->ndim = ndim;

        tensor->device = (char*)malloc(strlen(device) + 1);
        if (device != NULL) {
            strcpy(tensor->device, device);
        } else {
            fprintf(stderr, "Memory allocation failed\n");
            exit(-1);
        }

        tensor->size = 1;
        for (int i = 0; i < ndim; i++) {
            tensor->size *= shape[i];
        }

        tensor->strides = (int*)malloc(ndim * sizeof(int));
        if (tensor->strides == NULL) {
            fprintf(stderr, "Memory allocation failed\n");
            exit(1);
        }
        int stride = 1;
        for (int i = ndim - 1; i >= 0; i--) {
            tensor->strides[i] = stride;
            stride *= shape[i];
        }

        printf("Tensor created successfully\n");
        printf("Tensor information:\n");
        printf("Number of dimensions: %d\n", tensor->ndim);
        printf("Number size: %d\n", tensor->size);
        printf("Device: %s\n", tensor->device);

        printf("Shape: [");
        for (int i = 0; i < ndim; i++) {
            printf("%d", tensor->shape[i]);
            if (i < ndim - 1) {
                printf(", ");
            }
        }
        printf("]\n");

        /*printf("Data:\n[");
        for (int i = 0; i < stride; i++) {
            printf("%.2f", tensor->data[i]);
            if (i < stride - 1) {
                printf(", ");
            }
        }
        printf("]\n\n\n");*/
        
        return tensor;
    }

    float get_item(Tensor* tensor, int* indices) {
        int index = 0;
        for (int i = 0; i < tensor->ndim; i++) {
            index += indices[i] * tensor->strides[i];
        }

        float result;
        if (strcmp(tensor->device, "cuda") == 0) {
            cudaMemcpy(&result, tensor->data + index, sizeof(float), cudaMemcpyDeviceToHost);
        } else {
            result = tensor->data[index];
        }

        return result;
    }

    void to_device(Tensor* tensor, char* target_device) {
        printf("Sending tensor to device: %s\n", target_device);

        if ((strcmp(target_device, "cuda") == 0) && (strcmp(tensor->device, "cpu") == 0)) {
            cpu_to_cuda(tensor);
        }

        else if ((strcmp(target_device, "cpu") == 0) && (strcmp(tensor->device, "cuda") == 0)) {
            cuda_to_cpu(tensor);
        }
    }

    Tensor* add_tensor(Tensor* tensor1, Tensor* tensor2) {
        if (tensor1->ndim != tensor2->ndim) {
            fprintf(stderr, "Tensors must have the same number of dimensions %d and %d for addition\n", tensor1->ndim, tensor2->ndim);
            exit(1);
        }

        if (strcmp(tensor1->device, tensor2->device) != 0) {
            fprintf(stderr, "Tensors must be on the same device: %s and %s\n", tensor1->device, tensor2->device);
            exit(1);
        }

        char* device = (char*)malloc(strlen(tensor1->device) + 1);
        if (device != NULL) {
            strcpy(device, tensor1->device);
        } else {
            fprintf(stderr, "Memory allocation failed\n");
            exit(-1);
        }
        int ndim = tensor1->ndim;
        int* shape = (int*)malloc(ndim * sizeof(int));
        if (shape == NULL) {
            fprintf(stderr, "Memory allocation failed\n");
            exit(1);
        }

        for (int i = 0; i < ndim; i++) {
            if (tensor1->shape[i] != tensor2->shape[i]) {
                fprintf(stderr, "Tensors must have the same shape %d and %d at index %d for addition\n", tensor1->shape[i], tensor2->shape[i], i);
                exit(1);
            }
            shape[i] = tensor1->shape[i];
        }        
        if (strcmp(tensor1->device, "cuda") == 0) {

            float* result_data;
            cudaMalloc((void **)&result_data, tensor1->size * sizeof(float));
            add_tensor_cuda(tensor1, tensor2, result_data);
            return create_tensor(result_data, shape, ndim, device);
        } 
        else {
            float* result_data = (float*)malloc(tensor1->size * sizeof(float));
            if (result_data == NULL) {
                fprintf(stderr, "Memory allocation failed\n");
                exit(1);
            }
            add_tensor_cpu(tensor1, tensor2, result_data);
            return create_tensor(result_data, shape, ndim, device);
        }     
    }

    Tensor* sum_tensor(Tensor* tensor) {

        char* device = (char*)malloc(strlen(tensor->device) + 1);
        if (device != NULL) {
            strcpy(device, tensor->device);
        } else {
            fprintf(stderr, "Memory allocation failed\n");
            exit(-1);
        }
        int ndim = 1;
        int* shape = (int*)malloc(ndim * sizeof(int));
        if (shape == NULL) {
            fprintf(stderr, "Memory allocation failed\n");
            exit(1);
        }

        shape[0] = 1;
  
        if (strcmp(tensor->device, "cuda") == 0) {

            float* result_data;
            int number_of_blocks = (tensor->size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
            cudaMalloc((void**)&result_data, number_of_blocks * sizeof(float));
            sum_tensor_cuda(tensor, result_data, number_of_blocks);
            return create_tensor(result_data, shape, ndim, device);
        } 
        else {
            float* result_data = (float*)malloc(1 * sizeof(float));
            if (result_data == NULL) {
                fprintf(stderr, "Memory allocation failed\n");
                exit(1);
            }
            sum_tensor_cpu(tensor, result_data);
            return create_tensor(result_data, shape, ndim, device);
        }     
    }

    Tensor* sub_tensor(Tensor* tensor1, Tensor* tensor2) {
        if (tensor1->ndim != tensor2->ndim) {
            fprintf(stderr, "Tensors must have the same number of dimensions %d and %d for subtraction\n", tensor1->ndim, tensor2->ndim);
            exit(1);
        }

        if (strcmp(tensor1->device, tensor2->device) != 0) {
            fprintf(stderr, "Tensors must be on the same device: %s and %s\n", tensor1->device, tensor2->device);
            exit(1);
        }

        char* device = (char*)malloc(strlen(tensor1->device) + 1);
        if (device != NULL) {
            strcpy(device, tensor1->device);
        } else {
            fprintf(stderr, "Memory allocation failed\n");
            exit(-1);
        }
        int ndim = tensor1->ndim;
        int* shape = (int*)malloc(ndim * sizeof(int));
        if (shape == NULL) {
            fprintf(stderr, "Memory allocation failed\n");
            exit(1);
        }

        for (int i = 0; i < ndim; i++) {
            if (tensor1->shape[i] != tensor2->shape[i]) {
                fprintf(stderr, "Tensors must have the same shape %d and %d at index %d for subtraction\n", tensor1->shape[i], tensor2->shape[i], i);
                exit(1);
            }
            shape[i] = tensor1->shape[i];
        }

        if (strcmp(tensor1->device, "cuda") == 0) {

            float* result_data;
            cudaMalloc((void **)&result_data, tensor1->size * sizeof(float));
            sub_tensor_cuda(tensor1, tensor2, result_data);
            return create_tensor(result_data, shape, ndim, device);
        } 
        else {
            float* result_data = (float*)malloc(tensor1->size * sizeof(float));
            if (result_data == NULL) {
                fprintf(stderr, "Memory allocation failed\n");
                exit(1);
            }
            sub_tensor_cpu(tensor1, tensor2, result_data);
            return create_tensor(result_data, shape, ndim, device);
        }
    }

    Tensor* elementwise_mul_tensor(Tensor* tensor1, Tensor* tensor2) {
        if (tensor1->ndim != tensor2->ndim) {
            fprintf(stderr, "Tensors must have the same number of dimensions %d and %d for element-wise multiplication\n", tensor1->ndim, tensor2->ndim);
            exit(1);
        }

        if (strcmp(tensor1->device, tensor2->device) != 0) {
            fprintf(stderr, "Tensors must be on the same device: %s and %s\n", tensor1->device, tensor2->device);
            exit(1);
        }

        char* device = (char*)malloc(strlen(tensor1->device) + 1);
        if (device != NULL) {
            strcpy(device, tensor1->device);
        } else {
            fprintf(stderr, "Memory allocation failed\n");
            exit(-1);
        }
        int ndim = tensor1->ndim;
        int* shape = (int*)malloc(ndim * sizeof(int));
        if (shape == NULL) {
            fprintf(stderr, "Memory allocation failed\n");
            exit(1);
        }

        for (int i = 0; i < ndim; i++) {
            if (tensor1->shape[i] != tensor2->shape[i]) {
                fprintf(stderr, "Tensors must have the same shape %d and %d at index %d for element-wise multiplication\n", tensor1->shape[i], tensor2->shape[i], i);
                exit(1);
            }
            shape[i] = tensor1->shape[i];
        }

        if (strcmp(tensor1->device, "cuda") == 0) {

            float* result_data;
            cudaMalloc((void **)&result_data, tensor1->size * sizeof(float));
            elementwise_mul_tensor_cuda(tensor1, tensor2, result_data);
            return create_tensor(result_data, shape, ndim, device);
        } 
        else {
            float* result_data = (float*)malloc(tensor1->size * sizeof(float));
            if (result_data == NULL) {
                fprintf(stderr, "Memory allocation failed\n");
                exit(1);
            }
            elementwise_mul_tensor_cpu(tensor1, tensor2, result_data);
            return create_tensor(result_data, shape, ndim, device);
        }
    }

    Tensor* scalar_mul_tensor(Tensor* tensor, float scalar) {

        char* device = (char*)malloc(strlen(tensor->device) + 1);
        if (device != NULL) {
            strcpy(device, tensor->device);
        } else {
            fprintf(stderr, "Memory allocation failed\n");
            exit(-1);
        }
        int ndim = tensor->ndim;
        int* shape = (int*)malloc(ndim * sizeof(int));
        if (shape == NULL) {
            fprintf(stderr, "Memory allocation failed\n");
            exit(1);
        }

        for (int i = 0; i < ndim; i++) {
            shape[i] = tensor->shape[i];
        }

        if (strcmp(tensor->device, "cuda") == 0) {

            float* result_data;
            cudaMalloc((void **)&result_data, tensor->size * sizeof(float));
            scalar_mul_tensor_cuda(tensor, scalar, result_data);
            return create_tensor(result_data, shape, ndim, device);
        } 
        else {
            float* result_data = (float*)malloc(tensor->size * sizeof(float));
            if (result_data == NULL) {
                fprintf(stderr, "Memory allocation failed\n");
                exit(1);
            }
            scalar_mul_tensor_cpu(tensor, scalar, result_data);
            return create_tensor(result_data, shape, ndim, device);
        }
    }

    Tensor* matmul_tensor(Tensor* tensor1, Tensor* tensor2) {
        // Check if tensors have compatible shapes for matrix multiplication
        if (tensor1->shape[1] != tensor2->shape[0]) {
            fprintf(stderr, "Incompatible shapes for matrix multiplication\n");
            exit(1);
        }

        if (strcmp(tensor1->device, tensor2->device) != 0) {
            fprintf(stderr, "Tensors must be on the same device: %s and %s\n", tensor1->device, tensor2->device);
            exit(1);
        }

        char* device = (char*)malloc(strlen(tensor1->device) + 1);
        if (device != NULL) {
            strcpy(device, tensor1->device);
        } else {
            fprintf(stderr, "Memory allocation failed\n");
            exit(-1);
        }
        int ndim = tensor1->ndim + tensor2->ndim - 2;
        int* shape = (int*)malloc(ndim * sizeof(int));
        if (shape == NULL) {
            fprintf(stderr, "Memory allocation failed\n");
            exit(1);
        }
        for (int i = 0; i < tensor1->ndim - 1; i++) {
            shape[i] = tensor1->shape[i];
        }
        for (int i = tensor1->ndim - 1; i < ndim; i++) {
            shape[i] = tensor2->shape[i - tensor1->ndim + 2];
        }

        int size = 1;
        for (int i = 0; i < ndim; i++) {
            size *= shape[i];
        }

        float* result_data = (float*)malloc(size * sizeof(float));
        if (result_data == NULL) {
            fprintf(stderr, "Memory allocation failed\n");
            exit(1);
        }

        if (strcmp(tensor1->device, "cuda") == 0) {

            float* result_data;
            cudaMalloc((void **)&result_data, size * sizeof(float));
            matmul_tensor_cuda(tensor1, tensor2, result_data);
            return create_tensor(result_data, shape, ndim, device);
        } 
        else {
            float* result_data = (float*)malloc(size * sizeof(float));
            if (result_data == NULL) {
                fprintf(stderr, "Memory allocation failed\n");
                exit(1);
            }
            matmul_tensor_cpu(tensor1, tensor2, result_data);
            return create_tensor(result_data, shape, ndim, device);
        }
    }

    Tensor* pow_tensor(Tensor* tensor, float power) {
        char* device = (char*)malloc(strlen(tensor->device) + 1);
        if (device != NULL) {
            strcpy(device, tensor->device);
        } else {
            fprintf(stderr, "Memory allocation failed\n");
            exit(-1);
        }
        int ndim = tensor->ndim;
        int* shape = (int*)malloc(ndim * sizeof(int));
        if (shape == NULL) {
            fprintf(stderr, "Memory allocation failed\n");
            exit(1);
        }

        for (int i = 0; i < ndim; i++) {
            shape[i] = tensor->shape[i];
        }

        if (strcmp(tensor->device, "cuda") == 0) {

            float* result_data;
            cudaMalloc((void **)&result_data, tensor->size * sizeof(float));
            pow_tensor_cuda(tensor, power, result_data);
            return create_tensor(result_data, shape, ndim, device);
        } 
        else {
            float* result_data = (float*)malloc(tensor->size * sizeof(float));
            if (result_data == NULL) {
                fprintf(stderr, "Memory allocation failed\n");
                exit(1);
            }
            pow_tensor_cpu(tensor, power, result_data);
            return create_tensor(result_data, shape, ndim, device);
        }
    }

    Tensor* reshape_tensor(Tensor* tensor, int* new_shape, int new_ndim) {
        char* device = (char*)malloc(strlen(tensor->device) + 1);
        if (device != NULL) {
            strcpy(device, tensor->device);
        } else {
            fprintf(stderr, "Memory allocation failed\n");
            exit(-1);
        }

        // Calculate the total number of elements in the new shape
        int new_size = 1;
        for (int i = 0; i < new_ndim; i++) {
            new_size *= new_shape[i];
        }

        // Check if the total number of elements matches the current tensor's size
        if (new_size != tensor->size) {
            fprintf(stderr, "Cannot reshape tensor. Total number of elements in new shape does not match the current size of the tensor.\n");
            exit(1);
        }

        if (strcmp(tensor->device, "cuda") == 0) {

            float* result_data;
            cudaMalloc((void **)&result_data, tensor->size * sizeof(float));
            assign_tensor_cuda(tensor, result_data);
            return create_tensor(result_data, new_shape, new_ndim, device);
        } 
        else {
            float* result_data = (float*)malloc(tensor->size * sizeof(float));
            if (result_data == NULL) {
                fprintf(stderr, "Memory allocation failed\n");
                exit(1);
            }
            assign_tensor_cpu(tensor, result_data);
            return create_tensor(result_data, new_shape, new_ndim, device);
        }
    }

    Tensor* ones_like_tensor(Tensor* tensor) {
        char* device = (char*)malloc(strlen(tensor->device) + 1);
        if (device != NULL) {
            strcpy(device, tensor->device);
        } else {
            fprintf(stderr, "Memory allocation failed\n");
            exit(-1);
        }
        int ndim = tensor->ndim;
        int* shape = (int*)malloc(ndim * sizeof(int));
        if (shape == NULL) {
            fprintf(stderr, "Memory allocation failed\n");
            exit(1);
        }

        for (int i = 0; i < ndim; i++) {
            shape[i] = tensor->shape[i];
        }

        if (strcmp(tensor->device, "cuda") == 0) {

            float* result_data;
            cudaMalloc((void **)&result_data, tensor->size * sizeof(float));
            ones_like_tensor_cuda(tensor, result_data);
            return create_tensor(result_data, shape, ndim, device);
        } 
        else {
            float* result_data = (float*)malloc(tensor->size * sizeof(float));
            if (result_data == NULL) {
                fprintf(stderr, "Memory allocation failed\n");
                exit(1);
            }
            ones_like_tensor_cpu(tensor, result_data);
            return create_tensor(result_data, shape, ndim, device);
        }
    }

    Tensor* zeros_like_tensor(Tensor* tensor) {
        char* device = (char*)malloc(strlen(tensor->device) + 1);
        if (device != NULL) {
            strcpy(device, tensor->device);
        } else {
            fprintf(stderr, "Memory allocation failed\n");
            exit(-1);
        }
        int ndim = tensor->ndim;
        int* shape = (int*)malloc(ndim * sizeof(int));
        if (shape == NULL) {
            fprintf(stderr, "Memory allocation failed\n");
            exit(1);
        }

        for (int i = 0; i < ndim; i++) {
            shape[i] = tensor->shape[i];
        }

        if (strcmp(tensor->device, "cuda") == 0) {

            float* result_data;
            cudaMalloc((void **)&result_data, tensor->size * sizeof(float));
            zeros_like_tensor_cuda(tensor, result_data);
            return create_tensor(result_data, shape, ndim, device);
        } 
        else {
            float* result_data = (float*)malloc(tensor->size * sizeof(float));
            if (result_data == NULL) {
                fprintf(stderr, "Memory allocation failed\n");
                exit(1);
            }
            zeros_like_tensor_cpu(tensor, result_data);
            return create_tensor(result_data, shape, ndim, device);
        }
    }

    Tensor* transpose_tensor(Tensor* tensor) {
        char* device = (char*)malloc(strlen(tensor->device) + 1);
        if (device != NULL) {
            strcpy(device, tensor->device);
        } else {
            fprintf(stderr, "Memory allocation failed\n");
            exit(-1);
        }

        int ndim = tensor->ndim;
        int* shape = (int*)malloc(ndim * sizeof(int));
        if (shape == NULL) {
            fprintf(stderr, "Memory allocation failed\n");
            exit(-1);
        }

        for (int i = 0; i < ndim; i++) {
            shape[i] = tensor->shape[ndim - 1 - i];
        }

        int size = tensor->size;

        if (strcmp(tensor->device, "cuda") == 0) {

            float* result_data;
            cudaMalloc((void **)&result_data, size * sizeof(float));
            transpose_tensor_cuda(tensor, result_data);
            return create_tensor(result_data, shape, ndim, device);
        } 
        else {
            float* result_data = (float*)malloc(size * sizeof(float));
            if (result_data == NULL) {
                fprintf(stderr, "Memory allocation failed\n");
                exit(1);
            }
            transpose_tensor_cpu(tensor, result_data);
            return create_tensor(result_data, shape, ndim, device);
        }
    }
}
