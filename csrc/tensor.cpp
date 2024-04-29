#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "tensor.h"
#include "cuda.h"
#define THREADS_PER_BLOCK 128

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
        tensor->device = device;

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
        return tensor->data[index];
    }

    void to_device(Tensor* tensor, char* device) {
        #ifdef __NVCC__
            if ((strcmp(device, "cuda") == 0) && (strcmp(tensor->device, "cpu") == 0)) {
                
                float* data_tmp;

                cudaMalloc((void **)&data_tmp, tensor->size * sizeof(float));
                cudaMemcpy(data_tmp, tensor->data, tensor->size * sizeof(float), cudaMemcpyHostToDevice);

                free(tensor_data);
                tensor->data = data_tmp;
            }

            else if ((strcmp(device, "cpu") == 0) && (strcmp(tensor->device, "cuda") == 0)) {
                float* data_tmp = (float*)malloc(tensor->size * sizeof(float));

                cudaMemcpy(data_tmp, tensor->data, tensor->size * sizeof(float), cudaMemcpyDeviceToHost);
                cudaFree(tensor->data);

                tensor->data = data_tmp;
            }
        #endif
    }



    Tensor* add_tensor(Tensor* tensor1, Tensor* tensor2) {
        printf("Adding tensor\n");
        if (tensor1->ndim != tensor2->ndim) {
            fprintf(stderr, "Tensors must have the same number of dimensions %d and %d for addition\n", tensor1->ndim, tensor2->ndim);
            exit(1);
        }

        if (strcmp(tensor1->device, tensor2->device) != 0) {
            fprintf(stderr, "Tensors must be on the same device\n");
            exit(1);
        }

        char* device = tensor1->device;
        int ndim = tensor1->ndim;
        int* shape = (int*)malloc(ndim * sizeof(int));
        if (shape == NULL) {
            fprintf(stderr, "Memory allocation failed\n");
            exit(1);
        }

        printf("Size: %d\n", tensor1->size);
        /*printf("Data: [");
        for (int i = 0; i < tensor1->size; i++) {
            printf("%.2f", tensor1->data[i]);
            if (i < tensor1->size - 1) {
                printf(", ");
            }
        }
        printf("]\n");*/

        printf("Size: %d\n", tensor2->size);
        /*printf("Data: [");
        for (int i = 0; i < tensor2->size; i++) {
            printf("%.2f", tensor2->data[i]);
            if (i < tensor2->size - 1) {
                printf(", ");
            }
        }
        printf("]\n");*/

        printf("Shapes : [");
        for (int i = 0; i < tensor1->ndim; i++) {
            printf("%d", tensor1->shape[i]);
            if (i < tensor1->ndim - 1) {
                printf(", ");
            }
        }
        printf("]\n");

        printf("Shapes : [");
        for (int i = 0; i < tensor2->ndim; i++) {
            printf("%d", tensor2->shape[i]);
            if (i < tensor2->ndim - 1) {
                printf(", ");
            }
        }
        printf("]\n");

        for (int i = 0; i < ndim; i++) {
            if (tensor1->shape[i] != tensor2->shape[i]) {
                fprintf(stderr, "Tensors must have the same shape %d and %d at index %d for addition\n", tensor1->shape[i], tensor2->shape[i], i);
                exit(1);
            }
            shape[i] = tensor1->shape[i];
        }
        
        #ifdef __NVCC__
            if (strcmp(tensor1->device, "cuda") != 0) {

                float* result_data;

                cudaMalloc((void **)&result_data, tensor1->size * sizeof(float));

                int number_of_blocks = (tensor1->size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
                add_tensor_cuda<<<number_of_blocks, THREADS_PER_BLOCK>>>(tensor1->data, tensor2->data, result_data, tensor1->size);

                cudaError_t error = cudaGetLastError();
                if (error != cudaSuccess) {
                    printf("CUDA error: %s\n", cudaGetErrorString(error));
                    exit(-1);
                }

                cudaDeviceSynchronize();

                return create_tensor(result_data, shape, ndim, device);
            } 
        #endif
        
        float* result_data = (float*)malloc(tensor1->size * sizeof(float));
        if (result_data == NULL) {
            fprintf(stderr, "Memory allocation failed\n");
            exit(1);
        }

        for (int i = 0; i < tensor1->size; i++) {
            result_data[i] = tensor1->data[i] + tensor2->data[i];
        }
        

        return create_tensor(result_data, shape, ndim, device);
    }

    Tensor* sub_tensor(Tensor* tensor1, Tensor* tensor2) {
        printf("Adding tensor\n");
        if (tensor1->ndim != tensor2->ndim) {
            fprintf(stderr, "Tensors must have the same number of dimensions %d and %d for subtraction\n", tensor1->ndim, tensor2->ndim);
            exit(1);
        }

        if (strcmp(tensor1->device, tensor2->device) != 0) {
            fprintf(stderr, "Tensors must be on the same device\n");
            exit(1);
        }

        char* device = tensor1->device;

        int ndim = tensor1->ndim;
        int* shape = (int*)malloc(ndim * sizeof(int));
        if (shape == NULL) {
            fprintf(stderr, "Memory allocation failed\n");
            exit(1);
        }

        printf("Size: %d\n", tensor1->size);
        printf("Data: [");
        for (int i = 0; i < tensor1->size; i++) {
            printf("%.2f", tensor1->data[i]);
            if (i < tensor1->size - 1) {
                printf(", ");
            }
        }
        printf("]\n");

        printf("Size: %d\n", tensor2->size);
        printf("Data: [");
        for (int i = 0; i < tensor2->size; i++) {
            printf("%.2f", tensor2->data[i]);
            if (i < tensor2->size - 1) {
                printf(", ");
            }
        }
        printf("]\n");

        printf("Shapes : [");
        for (int i = 0; i < tensor1->ndim; i++) {
            printf("%d", tensor1->shape[i]);
            if (i < tensor1->ndim - 1) {
                printf(", ");
            }
        }
        printf("]\n");

        printf("Shapes : [");
        for (int i = 0; i < tensor2->ndim; i++) {
            printf("%d", tensor2->shape[i]);
            if (i < tensor2->ndim - 1) {
                printf(", ");
            }
        }
        printf("]\n");

        for (int i = 0; i < ndim; i++) {
            if (tensor1->shape[i] != tensor2->shape[i]) {
                fprintf(stderr, "Tensors must have the same shape %d and %d at index %d for subtraction\n", tensor1->shape[i], tensor2->shape[i], i);
                exit(1);
            }
            shape[i] = tensor1->shape[i];
        }

        float* result_data = (float*)malloc(tensor1->size * sizeof(float));
        if (result_data == NULL) {
            fprintf(stderr, "Memory allocation failed\n");
            exit(1);
        }

        for (int i = 0; i < tensor1->size; i++) {
            result_data[i] = tensor1->data[i] - tensor2->data[i];
        }

        return create_tensor(result_data, shape, ndim, device);
    }

    Tensor* elementwise_mul_tensor(Tensor* tensor1, Tensor* tensor2) {
        printf("Adding tensor\n");
        if (tensor1->ndim != tensor2->ndim) {
            fprintf(stderr, "Tensors must have the same number of dimensions %d and %d for element-wise multiplication\n", tensor1->ndim, tensor2->ndim);
            exit(1);
        }

        if (strcmp(tensor1->device, tensor2->device) != 0) {
            fprintf(stderr, "Tensors must be on the same device\n");
            exit(1);
        }

        char* device = tensor1->device;

        int ndim = tensor1->ndim;
        int* shape = (int*)malloc(ndim * sizeof(int));
        if (shape == NULL) {
            fprintf(stderr, "Memory allocation failed\n");
            exit(1);
        }

        printf("Size: %d\n", tensor1->size);
        printf("Data: [");
        for (int i = 0; i < tensor1->size; i++) {
            printf("%.2f", tensor1->data[i]);
            if (i < tensor1->size - 1) {
                printf(", ");
            }
        }
        printf("]\n");

        printf("Size: %d\n", tensor2->size);
        printf("Data: [");
        for (int i = 0; i < tensor2->size; i++) {
            printf("%.2f", tensor2->data[i]);
            if (i < tensor2->size - 1) {
                printf(", ");
            }
        }
        printf("]\n");

        printf("Shapes : [");
        for (int i = 0; i < tensor1->ndim; i++) {
            printf("%d", tensor1->shape[i]);
            if (i < tensor1->ndim - 1) {
                printf(", ");
            }
        }
        printf("]\n");

        printf("Shapes : [");
        for (int i = 0; i < tensor2->ndim; i++) {
            printf("%d", tensor2->shape[i]);
            if (i < tensor2->ndim - 1) {
                printf(", ");
            }
        }
        printf("]\n");

        for (int i = 0; i < ndim; i++) {
            if (tensor1->shape[i] != tensor2->shape[i]) {
                fprintf(stderr, "Tensors must have the same shape %d and %d at index %d for element-wise multiplication\n", tensor1->shape[i], tensor2->shape[i], i);
                exit(1);
            }
            shape[i] = tensor1->shape[i];
        }

        float* result_data = (float*)malloc(tensor1->size * sizeof(float));
        if (result_data == NULL) {
            fprintf(stderr, "Memory allocation failed\n");
            exit(1);
        }

        for (int i = 0; i < tensor1->size; i++) {
            result_data[i] = tensor1->data[i] * tensor2->data[i];
        }

        return create_tensor(result_data, shape, ndim, device);
    }

    Tensor* matmul_tensor(Tensor* tensor1, Tensor* tensor2) {
        // Check if tensors have compatible shapes for matrix multiplication
        if (tensor1->shape[1] != tensor2->shape[0]) {
            fprintf(stderr, "Incompatible shapes for matrix multiplication\n");
            exit(1);
        }

        if (strcmp(tensor1->device, tensor2->device) != 0) {
            fprintf(stderr, "Tensors must be on the same device\n");
            exit(1);
        }

        char* device = tensor1->device;

        // Calculate the shape of the result tensor
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

        for (int i = 0; i < tensor1->shape[0]; i++) {
            for (int j = 0; j < tensor2->shape[1]; j++) {
                float sum = 0.0;
                for (int k = 0; k < tensor1->shape[1]; k++) {
                    sum += tensor1->data[i * tensor1->shape[1] + k] * tensor2->data[k * tensor2->shape[1] + j];
                }
                result_data[i * tensor2->shape[1] + j] = sum;
            }
        }

        return create_tensor(result_data, shape, ndim, device);
    }

    Tensor* pow_tensor(Tensor* tensor, float power) {
        printf("Powering tensor\n");
        char* device = tensor->device;
        int ndim = tensor->ndim;
        int* shape = (int*)malloc(ndim * sizeof(int));
        if (shape == NULL) {
            fprintf(stderr, "Memory allocation failed\n");
            exit(1);
        }

        printf("Size: %d\n", tensor->size);
        printf("Data: [");
        for (int i = 0; i < tensor->size; i++) {
            printf("%.2f", tensor->data[i]);
            if (i < tensor->size - 1) {
                printf(", ");
            }
        }
        printf("]\n");

        printf("Shapes : [");
        for (int i = 0; i < tensor->ndim; i++) {
            printf("%d", tensor->shape[i]);
            if (i < tensor->ndim - 1) {
                printf(", ");
            }
        }
        printf("]\n");

        for (int i = 0; i < ndim; i++) {
            shape[i] = tensor->shape[i];
        }

        float* result_data = (float*)malloc(tensor->size * sizeof(float));
        if (result_data == NULL) {
            fprintf(stderr, "Memory allocation failed\n");
            exit(1);
        }

        for (int i = 0; i < tensor->size; i++) {
            result_data[i] = powf(tensor->data[i], power);
        }

        return create_tensor(result_data, shape, ndim, device);
    }

    void reshape_tensor(Tensor* tensor, int* new_shape, int new_ndim) {
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

        // Update the shape
        tensor->shape = (int*)malloc(new_ndim * sizeof(int));
        if (tensor->shape == NULL) {
            fprintf(stderr, "Memory allocation failed\n");
            exit(1);
        }
        for (int i = 0; i < new_ndim; i++) {
            tensor->shape[i] = new_shape[i];
        }
        tensor->ndim = new_ndim;

        // Update the strides
        tensor->strides = (int*)malloc(new_ndim * sizeof(int));
        if (tensor->strides == NULL) {
            fprintf(stderr, "Memory allocation failed\n");
            exit(1);
        }

        int stride = 1;
        for (int i = new_ndim - 1; i >= 0; i--) {
            tensor->strides[i] = stride;
            stride *= new_shape[i];
        }
    }
}
