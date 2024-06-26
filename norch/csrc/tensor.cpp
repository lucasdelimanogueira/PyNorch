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

        return tensor;
    }

    void delete_tensor(Tensor* tensor) {
        if (tensor != NULL) {
            free(tensor);
            tensor = NULL;
        }
    }

    void delete_shape(Tensor* tensor) {
        if (tensor->shape != NULL) {
            free(tensor->shape);
            tensor->shape = NULL;
        }
    }

    void delete_data(Tensor* tensor) {
        if (tensor->data != NULL) {
            if (strcmp(tensor->device, "cpu") == 0) {
                free(tensor->data);
            } else {
                free_cuda(tensor->data);
            }
            tensor->data = NULL;
        }
    }

    void delete_strides(Tensor* tensor) {
        if (tensor->strides != NULL) {
            free(tensor->strides);
            tensor->strides = NULL;
        }
    }

    void delete_device(Tensor* tensor) {
        if (tensor->device != NULL) {
            free(tensor->device);
            tensor->device = NULL;
        }
    }

    float get_item(Tensor* tensor, int* indices) {
        int index = 0;
        for (int i = 0; i < tensor->ndim; i++) {
            index += indices[i] * tensor->strides[i];
        }

        float result;
        if (strcmp(tensor->device, "cpu") == 0) {
            result = tensor->data[index];
        } else {
            cudaMemcpy(&result, tensor->data + index, sizeof(float), cudaMemcpyDeviceToHost);            
        }

        return result;
    }

    void to_device(Tensor* tensor, char* target_device) {
        int device_id = 0;
        char* endptr;

        char* target_device_type;

        long num = strtol(target_device, &endptr, 10);
        if (*endptr == '\0') {
            device_id = (int)num;
            target_device_type = new char[strlen("cuda") + 1];
            strcpy(target_device_type, "cuda");
        }
        else {
            target_device_type = new char[strlen("cuda") + 1];
            strcpy(target_device_type, "cpu");
        }

        if ((strcmp(target_device_type, "cuda") == 0) && (strcmp(tensor->device, "cpu") == 0)) {
            cpu_to_cuda(tensor, device_id);
        }

        else if ((strcmp(target_device_type, "cpu") == 0) && (strcmp(tensor->device, "cuda") == 0)) {
            cuda_to_cpu(tensor);
        }

        free(target_device_type);
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
        
        if (strcmp(tensor1->device, "cpu") == 0) {
            float* result_data = (float*)malloc(tensor1->size * sizeof(float));
            if (result_data == NULL) {
                fprintf(stderr, "Memory allocation failed\n");
                exit(1);
            }
            add_tensor_cpu(tensor1, tensor2, result_data);
            return create_tensor(result_data, shape, ndim, tensor1->device);
        } 
        else {
            float* result_data;
            cudaMalloc((void **)&result_data, tensor1->size * sizeof(float));
            add_tensor_cuda(tensor1, tensor2, result_data);
            return create_tensor(result_data, shape, ndim, tensor1->device);
        }     
    }

    Tensor* add_broadcasted_tensor(Tensor* tensor1, Tensor* tensor2) {

        if (strcmp(tensor1->device, tensor2->device) != 0) {
            fprintf(stderr, "Tensors must be on the same device: %s and %s\n", tensor1->device, tensor2->device);
            exit(1);
        }

        int max_ndim = tensor1->ndim > tensor2->ndim ? tensor1->ndim : tensor2->ndim;

        // Determine the broadcasted shape
        int* broadcasted_shape = (int*)malloc(max_ndim * sizeof(int));
        if (broadcasted_shape == NULL) {
            fprintf(stderr, "Memory allocation failed\n");
            exit(1);
        }
        for (int i = 0; i < max_ndim; i++) {
            int dim1 = i < tensor1->ndim ? tensor1->shape[tensor1->ndim - 1 - i] : 1;
            int dim2 = i < tensor2->ndim ? tensor2->shape[tensor2->ndim - 1 - i] : 1;
            if (dim1 != dim2 && dim1 != 1 && dim2 != 1) {
                fprintf(stderr, "Shapes are not compatible for broadcasting\n");
                exit(1);
            }
            broadcasted_shape[max_ndim - 1 - i] = dim1 > dim2 ? dim1 : dim2;
        }

        int broadcasted_size = 1;
        for (int i = 0; i < max_ndim; i++) {
            broadcasted_size *= broadcasted_shape[i];
        }

        if (strcmp(tensor1->device, "cpu") == 0) {
            float* result_data = (float*)malloc(broadcasted_size * sizeof(float));
            if (result_data == NULL) {
                fprintf(stderr, "Memory allocation failed\n");
                exit(1);
            }

            add_broadcasted_tensor_cpu(tensor1, tensor2, result_data, broadcasted_shape, broadcasted_size);
            return create_tensor(result_data, broadcasted_shape, max_ndim, tensor1->device);
        } 
        else {
            float* result_data;
            cudaMalloc((void **)&result_data, broadcasted_size * sizeof(float));
            add_broadcasted_tensor_cuda(tensor1, tensor2, result_data, broadcasted_shape, broadcasted_size);
            return create_tensor(result_data, broadcasted_shape, max_ndim, tensor1->device);
        }
    }

    Tensor* sum_tensor(Tensor* tensor, int axis, bool keepdim) {
        int ndim;
        int* shape;
    
        if (axis > tensor->ndim - 1) {
            fprintf(stderr, "Error: axis argument %d must be smaller than tensor dimension %d", axis, tensor->ndim);
        }

        if (axis == -1) {
            
            shape = (int*) malloc(sizeof(int));
            shape[0] = 1;
            ndim = 1;
        } else {
            shape = (int*) malloc((tensor->ndim - 1) * sizeof(int));
            for (int i = 0, j = 0; i < tensor->ndim; ++i) {
                if (i != axis) {
                    shape[j++] = tensor->shape[i];
                }
            }
            ndim = tensor->ndim - 1;
        }

        int axis_size = 1;
        for (int i = 0; i < ndim; i++) {
            axis_size *= shape[i];
        }
  
        if (strcmp(tensor->device, "cpu") == 0) {
            float* result_data = (float*)calloc(axis_size, sizeof(float));
            if (result_data == NULL) {
                fprintf(stderr, "Memory allocation failed\n");
                exit(1);
            }

            sum_tensor_cpu(tensor, result_data, axis_size, shape, axis);

            if (keepdim) {
                if (axis == -1){
                    ndim = tensor->ndim;
                    shape = (int*) malloc((tensor->ndim) * sizeof(int));
                    for (int i = 0; i < tensor->ndim; i++) {
                        shape[i] = 1;
                    }
                } else {
                    shape = (int*) malloc((tensor->ndim) * sizeof(int));
                    for (int i = 0; i < tensor->ndim; i++) {
                        shape[i] = tensor->shape[i];
                    }
                    shape[axis] = 1;
                    ndim = tensor->ndim;
                }
                
            }
            return create_tensor(result_data, shape, ndim, tensor->device);
        } 
        else {
            float* result_data;
            if (axis == -1) {
                cudaMalloc((void**)&result_data, tensor->size * sizeof(float));
            } else {
                cudaMalloc((void**)&result_data, axis_size * sizeof(float));
            }
            sum_tensor_cuda(tensor, result_data, axis);
            
            if (keepdim) {
                if (axis == -1){
                    ndim = tensor->ndim;
                    shape = (int*) malloc((tensor->ndim) * sizeof(int));
                    for (int i = 0; i < tensor->ndim; i++) {
                        shape[i] = 1;
                    }
                } else {
                    shape = (int*) malloc((tensor->ndim) * sizeof(int));
                    for (int i = 0; i < tensor->ndim; i++) {
                        shape[i] = tensor->shape[i];
                    }
                    shape[axis] = 1;
                    ndim = tensor->ndim;
                }
                
            }
            return create_tensor(result_data, shape, ndim, tensor->device);
        }     
    }

    Tensor* max_tensor(Tensor* tensor, int axis, bool keepdim) {
        int ndim;
        int* shape;
        if (axis == -1) {     
            shape = (int*) malloc(sizeof(int));
            shape[0] = 1;
            ndim = 1;
        } else {
            shape = (int*) malloc((tensor->ndim - 1) * sizeof(int));
            for (int i = 0, j = 0; i < tensor->ndim; ++i) {
                if (i != axis) {
                    shape[j++] = tensor->shape[i];
                }
            }
            ndim = tensor->ndim - 1;
        }

        int axis_size = 1;
        for (int i = 0; i < ndim; i++) {
            axis_size *= shape[i];
        }
  
        if (strcmp(tensor->device, "cpu") == 0) {
            float* result_data = (float*)malloc(axis_size * sizeof(float));
            if (result_data == NULL) {
                fprintf(stderr, "Memory allocation failed\n");
                exit(1);
            }
            max_tensor_cpu(tensor, result_data, axis_size, shape, axis);

            if (keepdim) {
                if (axis == -1){
                    ndim = tensor->ndim;
                    shape = (int*) malloc((tensor->ndim) * sizeof(int));
                    for (int i = 0; i < tensor->ndim; i++) {
                        shape[i] = 1;
                    }
                } else {
                    shape = (int*) malloc((tensor->ndim) * sizeof(int));
                    for (int i = 0; i < tensor->ndim; i++) {
                        shape[i] = tensor->shape[i];
                    }
                    shape[axis] = 1;
                    ndim = tensor->ndim;
                }
                
            }
            return create_tensor(result_data, shape, ndim, tensor->device);

        } 
        else {
            float* result_data;
            if (axis == -1) {
                cudaMalloc((void**)&result_data, tensor->size * sizeof(float));
            } else {
                cudaMalloc((void**)&result_data, axis_size * sizeof(float));
            }
            max_tensor_cuda(tensor, result_data, axis);

            if (keepdim) {
                if (axis == -1){
                    ndim = tensor->ndim;
                    shape = (int*) malloc((tensor->ndim) * sizeof(int));
                    for (int i = 0; i < tensor->ndim; i++) {
                        shape[i] = 1;
                    }
                } else {
                    shape = (int*) malloc((tensor->ndim) * sizeof(int));
                    for (int i = 0; i < tensor->ndim; i++) {
                        shape[i] = tensor->shape[i];
                    }
                    shape[axis] = 1;
                    ndim = tensor->ndim;
                }
                
            }
            return create_tensor(result_data, shape, ndim, tensor->device);
        }     
    }

    Tensor* min_tensor(Tensor* tensor, int axis, bool keepdim) {
        int ndim;
        int* shape;
        if (axis == -1) {
            shape = (int*) malloc(sizeof(int));
            shape[0] = 1;
            ndim = 1;
        } else {
            shape = (int*) malloc((tensor->ndim - 1) * sizeof(int));
            for (int i = 0, j = 0; i < tensor->ndim; ++i) {
                if (i != axis) {
                    shape[j++] = tensor->shape[i];
                }
            }
            ndim = tensor->ndim - 1;
        }
        
        int axis_size = 1;
        for (int i = 0; i < ndim; i++) {
            axis_size *= shape[i];
        }
  
        if (strcmp(tensor->device, "cpu") == 0) {
            float* result_data = (float*)malloc(axis_size * sizeof(float));
            if (result_data == NULL) {
                fprintf(stderr, "Memory allocation failed\n");
                exit(1);
            }
            min_tensor_cpu(tensor, result_data, axis_size, shape, axis);

            if (keepdim) {
                if (axis == -1){
                    ndim = tensor->ndim;
                    shape = (int*) malloc((tensor->ndim) * sizeof(int));
                    for (int i = 0; i < tensor->ndim; i++) {
                        shape[i] = 1;
                    }
                } else {
                    shape = (int*) malloc((tensor->ndim) * sizeof(int));
                    for (int i = 0; i < tensor->ndim; i++) {
                        shape[i] = tensor->shape[i];
                    }
                    shape[axis] = 1;
                    ndim = tensor->ndim;
                }
                
            }
            return create_tensor(result_data, shape, ndim, tensor->device);

        } 
        else {
            float* result_data;
            if (axis == -1) {
                cudaMalloc((void**)&result_data, tensor->size * sizeof(float));
            } else {
                cudaMalloc((void**)&result_data, axis_size * sizeof(float));
            }
            min_tensor_cuda(tensor, result_data, axis);

            if (keepdim) {
                if (axis == -1){
                    ndim = tensor->ndim;
                    shape = (int*) malloc((tensor->ndim) * sizeof(int));
                    for (int i = 0; i < tensor->ndim; i++) {
                        shape[i] = 1;
                    }
                } else {
                    shape = (int*) malloc((tensor->ndim) * sizeof(int));
                    for (int i = 0; i < tensor->ndim; i++) {
                        shape[i] = tensor->shape[i];
                    }
                    shape[axis] = 1;
                    ndim = tensor->ndim;
                }
                
            }
            return create_tensor(result_data, shape, ndim, tensor->device);
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

        if (strcmp(tensor1->device, "cpu") == 0) {
            float* result_data = (float*)malloc(tensor1->size * sizeof(float));
            if (result_data == NULL) {
                fprintf(stderr, "Memory allocation failed\n");
                exit(1);
            }
            sub_tensor_cpu(tensor1, tensor2, result_data);
            return create_tensor(result_data, shape, ndim, tensor1->device);
        } 
        else {
            float* result_data;
            cudaMalloc((void **)&result_data, tensor1->size * sizeof(float));
            sub_tensor_cuda(tensor1, tensor2, result_data);
            return create_tensor(result_data, shape, ndim, tensor1->device);
        }
    }

    Tensor* sub_broadcasted_tensor(Tensor* tensor1, Tensor* tensor2) {

        if (strcmp(tensor1->device, tensor2->device) != 0) {
            fprintf(stderr, "Tensors must be on the same device: %s and %s\n", tensor1->device, tensor2->device);
            exit(1);
        }

        int max_ndim = tensor1->ndim > tensor2->ndim ? tensor1->ndim : tensor2->ndim;

        // Determine the broadcasted shape
        int* broadcasted_shape = (int*)malloc(max_ndim * sizeof(int));
        if (broadcasted_shape == NULL) {
            fprintf(stderr, "Memory allocation failed\n");
            exit(1);
        }
        for (int i = 0; i < max_ndim; i++) {
            int dim1 = i < tensor1->ndim ? tensor1->shape[tensor1->ndim - 1 - i] : 1;
            int dim2 = i < tensor2->ndim ? tensor2->shape[tensor2->ndim - 1 - i] : 1;
            if (dim1 != dim2 && dim1 != 1 && dim2 != 1) {
                fprintf(stderr, "Shapes are not compatible for broadcasting\n");
                exit(1);
            }
            broadcasted_shape[max_ndim - 1 - i] = dim1 > dim2 ? dim1 : dim2;
        }

        int broadcasted_size = 1;
        for (int i = 0; i < max_ndim; i++) {
            broadcasted_size *= broadcasted_shape[i];
        }

        if (strcmp(tensor1->device, "cpu") == 0) {
            float* result_data = (float*)malloc(broadcasted_size * sizeof(float));
            if (result_data == NULL) {
                fprintf(stderr, "Memory allocation failed\n");
                exit(1);
            }

            sub_broadcasted_tensor_cpu(tensor1, tensor2, result_data, broadcasted_shape, broadcasted_size);
            return create_tensor(result_data, broadcasted_shape, max_ndim, tensor1->device);
        } 
        else {

            float* result_data;
            cudaMalloc((void **)&result_data, broadcasted_size * sizeof(float));
            sub_broadcasted_tensor_cuda(tensor1, tensor2, result_data, broadcasted_shape, broadcasted_size);
            return create_tensor(result_data, broadcasted_shape, max_ndim, tensor1->device);
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

        if (strcmp(tensor1->device, "cpu") == 0) {
            float* result_data = (float*)malloc(tensor1->size * sizeof(float));
            if (result_data == NULL) {
                fprintf(stderr, "Memory allocation failed\n");
                exit(1);
            }
            elementwise_mul_tensor_cpu(tensor1, tensor2, result_data);
            return create_tensor(result_data, shape, ndim, tensor1->device);
        } 
        else {
            float* result_data;
            cudaMalloc((void **)&result_data, tensor1->size * sizeof(float));
            elementwise_mul_tensor_cuda(tensor1, tensor2, result_data);
            return create_tensor(result_data, shape, ndim, tensor1->device);
        }
    }

    Tensor* scalar_mul_tensor(Tensor* tensor, float scalar) {

        int ndim = tensor->ndim;
        int* shape = (int*)malloc(ndim * sizeof(int));
        if (shape == NULL) {
            fprintf(stderr, "Memory allocation failed\n");
            exit(1);
        }

        for (int i = 0; i < ndim; i++) {
            shape[i] = tensor->shape[i];
        }

        if (strcmp(tensor->device, "cpu") == 0) {
            float* result_data = (float*)malloc(tensor->size * sizeof(float));
            if (result_data == NULL) {
                fprintf(stderr, "Memory allocation failed\n");
                exit(1);
            }
            scalar_mul_tensor_cpu(tensor, scalar, result_data);
            return create_tensor(result_data, shape, ndim, tensor->device);
        } 
        else {
            float* result_data;
            cudaMalloc((void **)&result_data, tensor->size * sizeof(float));
            scalar_mul_tensor_cuda(tensor, scalar, result_data);
            return create_tensor(result_data, shape, ndim, tensor->device);
        }
    }

    Tensor* scalar_div_tensor(float scalar, Tensor* tensor) {

        int ndim = tensor->ndim;
        int* shape = (int*)malloc(ndim * sizeof(int));
        if (shape == NULL) {
            fprintf(stderr, "Memory allocation failed\n");
            exit(1);
        }

        for (int i = 0; i < ndim; i++) {
            shape[i] = tensor->shape[i];
        }

        if (strcmp(tensor->device, "cpu") == 0) {
            float* result_data = (float*)malloc(tensor->size * sizeof(float));
            if (result_data == NULL) {
                fprintf(stderr, "Memory allocation failed\n");
                exit(1);
            }
            scalar_div_tensor_cpu(scalar, tensor, result_data);
            return create_tensor(result_data, shape, ndim, tensor->device);
        } 
        else {
            float* result_data;
            cudaMalloc((void **)&result_data, tensor->size * sizeof(float));
            scalar_div_tensor_cuda(scalar, tensor, result_data);
            return create_tensor(result_data, shape, ndim, tensor->device);
        }
    }

    Tensor* tensor_div_scalar(Tensor* tensor, float scalar) {

        int ndim = tensor->ndim;
        int* shape = (int*)malloc(ndim * sizeof(int));
        if (shape == NULL) {
            fprintf(stderr, "Memory allocation failed\n");
            exit(1);
        }

        for (int i = 0; i < ndim; i++) {
            shape[i] = tensor->shape[i];
        }

        if (strcmp(tensor->device, "cpu") == 0) {
            float* result_data = (float*)malloc(tensor->size * sizeof(float));
            if (result_data == NULL) {
                fprintf(stderr, "Memory allocation failed\n");
                exit(1);
            }
            tensor_div_scalar_cpu(tensor, scalar, result_data);
            return create_tensor(result_data, shape, ndim, tensor->device);
        } 
        else {
            float* result_data;
            cudaMalloc((void **)&result_data, tensor->size * sizeof(float));
            tensor_div_scalar_cuda(tensor, scalar, result_data);
            return create_tensor(result_data, shape, ndim, tensor->device); 
        }
    }

    Tensor* tensor_div_tensor(Tensor* tensor1, Tensor* tensor2) {
        if (tensor1->ndim != tensor2->ndim) {
            fprintf(stderr, "Tensors must have the same number of dimensions %d and %d for element-wise division\n", tensor1->ndim, tensor2->ndim);
            exit(1);
        }

        if (strcmp(tensor1->device, tensor2->device) != 0) {
            fprintf(stderr, "Tensors must be on the same device: %s and %s\n", tensor1->device, tensor2->device);
            exit(1);
        }

        int ndim = tensor1->ndim;
        int* shape = (int*)malloc(ndim * sizeof(int));
        if (shape == NULL) {
            fprintf(stderr, "Memory allocation failed\n");
            exit(1);
        }

        for (int i = 0; i < ndim; i++) {
            if (tensor1->shape[i] != tensor2->shape[i]) {
                fprintf(stderr, "Tensors must have the same shape %d and %d at index %d for division\n", tensor1->shape[i], tensor2->shape[i], i);
                exit(1);
            }
            shape[i] = tensor1->shape[i];
        }

        if (strcmp(tensor1->device, "cpu") == 0) {
            float* result_data = (float*)malloc(tensor1->size * sizeof(float));
            if (result_data == NULL) {
                fprintf(stderr, "Memory allocation failed\n");
                exit(1);
            }
            tensor_div_tensor_cpu(tensor1, tensor2, result_data);
            return create_tensor(result_data, shape, ndim, tensor1->device);
        } 
        else {
            float* result_data;
            cudaMalloc((void **)&result_data, tensor1->size * sizeof(float));
            tensor_div_tensor_cuda(tensor1, tensor2, result_data);
            return create_tensor(result_data, shape, ndim, tensor1->device);
        }
    }

    Tensor* matmul_tensor(Tensor* tensor1, Tensor* tensor2) {
        //MxN @ NxP = MxP
        // Check if tensors have compatible shapes for matrix multiplication
        if (tensor1->shape[1] != tensor2->shape[0]) {
            fprintf(stderr, "Incompatible shapes for matrix multiplication %dx%d and %dx%d\n", tensor1->shape[0], tensor1->shape[1], tensor2->shape[0], tensor2->shape[1]);
            exit(1);
        }

        if (strcmp(tensor1->device, tensor2->device) != 0) {
            fprintf(stderr, "Tensors must be on the same device: %s and %s\n", tensor1->device, tensor2->device);
            exit(1);
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

        if (strcmp(tensor1->device, "cpu") == 0) {
            float* result_data = (float*)malloc(size * sizeof(float));
            if (result_data == NULL) {
                fprintf(stderr, "Memory allocation failed\n");
                exit(1);
            }
            matmul_tensor_cpu(tensor1, tensor2, result_data);
            return create_tensor(result_data, shape, ndim, tensor1->device);
        } 
        else {
            float* result_data;
            cudaMalloc((void **)&result_data, size * sizeof(float));
            matmul_tensor_cuda(tensor1, tensor2, result_data);
            return create_tensor(result_data, shape, ndim, tensor1->device);
        }
    }

    Tensor* broadcasted_batched_matmul_tensor(Tensor* tensor1, Tensor* tensor2) {
        //BATCHxMxP = MxN @ BATCHxNxP
        // Check if tensors have compatible shapes for matrix multiplication
        if (tensor1->shape[1] != tensor2->shape[1]) {
            fprintf(stderr, "Incompatible shapes for broadcasted batched matrix multiplication %dx%d and %dx%dx%d\n", tensor1->shape[0], tensor1->shape[1], tensor2->shape[0], tensor2->shape[1], tensor2->shape[2]);
            exit(1);
        }

        if (strcmp(tensor1->device, tensor2->device) != 0) {
            fprintf(stderr, "Tensors must be on the same device: %s and %s\n", tensor1->device, tensor2->device);
            exit(1);
        }

        int ndim = 3;
        int* shape = (int*)malloc(ndim * sizeof(int));
        if (shape == NULL) {
            fprintf(stderr, "Memory allocation failed\n");
            exit(1);
        }
        
        shape[0] = tensor2->shape[0];;
        shape[1] = tensor1->shape[0];
        shape[2] = tensor2->shape[2];  

        int size = 1;
        for (int i = 0; i < ndim; i++) {
            size *= shape[i];
        }

        float* result_data = (float*)malloc(size * sizeof(float));
        if (result_data == NULL) {
            fprintf(stderr, "Memory allocation failed\n");
            exit(1);
        }

        if (strcmp(tensor1->device, "cpu") == 0) {
            float* result_data = (float*)malloc(size * sizeof(float));
            if (result_data == NULL) {
                fprintf(stderr, "Memory allocation failed\n");
                exit(1);
            }
            broadcasted_batched_matmul_tensor_cpu(tensor1, tensor2, result_data);
            return create_tensor(result_data, shape, ndim, tensor1->device);
        } 
        else {
            float* result_data;
            cudaMalloc((void **)&result_data, size * sizeof(float));
            broadcasted_batched_matmul_tensor_cuda(tensor1, tensor2, result_data);
            return create_tensor(result_data, shape, ndim, tensor1->device);
        }

    }

    Tensor* batched_matmul_tensor(Tensor* tensor1, Tensor* tensor2) {
        //BATCHxMxP = BATCHxMxN @ BATCHxNxP
        // Check if tensors have compatible shapes for matrix multiplication

        if (tensor1->shape[0] != tensor2->shape[0]) {
            fprintf(stderr, "Tensors must have same batch dimension for batch matmul %d and %d\n", tensor1->shape[0], tensor2->shape[0]);
            exit(1);
        }

        if (tensor1->shape[2] != tensor2->shape[1]) {
            fprintf(stderr, "Incompatible shapes for matrix multiplication %dx%d and %dx%d\n", tensor1->shape[0], tensor1->shape[1], tensor2->shape[0], tensor2->shape[1]);
            exit(1);
        }
        
        if (strcmp(tensor1->device, tensor2->device) != 0) {
            fprintf(stderr, "Tensors must be on the same device: %s and %s\n", tensor1->device, tensor2->device);
            exit(1);
        }

        int ndim = 3;
        int* shape = (int*)malloc(ndim * sizeof(int));
        if (shape == NULL) {
            fprintf(stderr, "Memory allocation failed\n");
            exit(1);
        }
        
        shape[0] = tensor2->shape[0];;
        shape[1] = tensor1->shape[1];
        shape[2] = tensor2->shape[2];  

        int size = 1;
        for (int i = 0; i < ndim; i++) {
            size *= shape[i];
        }

        float* result_data = (float*)malloc(size * sizeof(float));
        if (result_data == NULL) {
            fprintf(stderr, "Memory allocation failed\n");
            exit(1);
        }

        if (strcmp(tensor1->device, "cpu") == 0) {
            float* result_data = (float*)malloc(size * sizeof(float));
            if (result_data == NULL) {
                fprintf(stderr, "Memory allocation failed\n");
                exit(1);
            }
            batched_matmul_tensor_cpu(tensor1, tensor2, result_data);
            return create_tensor(result_data, shape, ndim, tensor1->device);
        } 
        else {
            float* result_data;
            cudaMalloc((void **)&result_data, size * sizeof(float));
            batched_matmul_tensor_cuda(tensor1, tensor2, result_data);
            return create_tensor(result_data, shape, ndim, tensor1->device);
        }

    }


    Tensor* tensor_pow_scalar(Tensor* tensor, float exponent) {
        int ndim = tensor->ndim;
        int* shape = (int*)malloc(ndim * sizeof(int));
        if (shape == NULL) {
            fprintf(stderr, "Memory allocation failed\n");
            exit(1);
        }

        for (int i = 0; i < ndim; i++) {
            shape[i] = tensor->shape[i];
        }

        if (strcmp(tensor->device, "cpu") == 0) {
            float* result_data = (float*)malloc(tensor->size * sizeof(float));
            if (result_data == NULL) {
                fprintf(stderr, "Memory allocation failed\n");
                exit(1);
            }
            tensor_pow_scalar_cpu(tensor, exponent, result_data);
            return create_tensor(result_data, shape, ndim, tensor->device);
        } 
        else {
            float* result_data;
            cudaMalloc((void **)&result_data, tensor->size * sizeof(float));
            tensor_pow_scalar_cuda(tensor, exponent, result_data);
            return create_tensor(result_data, shape, ndim, tensor->device);
        }
    }

    Tensor* scalar_pow_tensor(float base, Tensor* tensor) {
        int ndim = tensor->ndim;
        int* shape = (int*)malloc(ndim * sizeof(int));
        if (shape == NULL) {
            fprintf(stderr, "Memory allocation failed\n");
            exit(1);
        }

        for (int i = 0; i < ndim; i++) {
            shape[i] = tensor->shape[i];
        }

        if (strcmp(tensor->device, "cpu") == 0) {
            float* result_data = (float*)malloc(tensor->size * sizeof(float));
            if (result_data == NULL) {
                fprintf(stderr, "Memory allocation failed\n");
                exit(1);
            }
            scalar_pow_tensor_cpu(base, tensor, result_data);
            return create_tensor(result_data, shape, ndim, tensor->device);
        } 
        else {
            float* result_data;
            cudaMalloc((void **)&result_data, tensor->size * sizeof(float));
            scalar_pow_tensor_cuda(base, tensor, result_data);
            return create_tensor(result_data, shape, ndim, tensor->device);       
        }
    }

    Tensor* log_tensor(Tensor* tensor) {
        int ndim = tensor->ndim;
        int* shape = (int*)malloc(ndim * sizeof(int));
        if (shape == NULL) {
            fprintf(stderr, "Memory allocation failed\n");
            exit(1);
        }

        for (int i = 0; i < ndim; i++) {
            shape[i] = tensor->shape[i];
        }

        if (strcmp(tensor->device, "cpu") == 0) {
            float* result_data = (float*)malloc(tensor->size * sizeof(float));
            if (result_data == NULL) {
                fprintf(stderr, "Memory allocation failed\n");
                exit(1);
            }
            log_tensor_cpu(tensor, result_data);
            return create_tensor(result_data, shape, ndim, tensor->device);
        } 
        else {
            float* result_data;
            cudaMalloc((void **)&result_data, tensor->size * sizeof(float));
            log_tensor_cuda(tensor, result_data);
            return create_tensor(result_data, shape, ndim, tensor->device);
        }
    }

    Tensor* reshape_tensor(Tensor* tensor, int* new_shape, int new_ndim) {

        int ndim = new_ndim;
        int* shape = (int*)malloc(ndim * sizeof(int));
        if (shape == NULL) {
            fprintf(stderr, "Memory allocation failed\n");
            exit(1);
        }

        for (int i = 0; i < ndim; i++) {
            shape[i] = new_shape[i];
        }

        // Calculate the total number of elements in the new shape
        int size = 1;
        for (int i = 0; i < new_ndim; i++) {
            size *= shape[i];
        }

        // Check if the total number of elements matches the current tensor's size
        if (size != tensor->size) {
            fprintf(stderr, "Cannot reshape tensor. Total number of elements in new shape does not match the current size of the tensor.\n");
            exit(1);
        }

        if (strcmp(tensor->device, "cpu") == 0) {
            float* result_data = (float*)malloc(tensor->size * sizeof(float));
            if (result_data == NULL) {
                fprintf(stderr, "Memory allocation failed\n");
                exit(1);
            }
            assign_tensor_cpu(tensor, result_data);
            return create_tensor(result_data, shape, ndim, tensor->device);
        } 
        else {
            float* result_data;
            cudaMalloc((void **)&result_data, tensor->size * sizeof(float));
            assign_tensor_cuda(tensor, result_data);
            return create_tensor(result_data, shape, ndim, tensor->device);
        }
    }

    Tensor* equal_tensor(Tensor* tensor1, Tensor* tensor2) {
        if (tensor1->ndim != tensor2->ndim) {
            fprintf(stderr, "Tensors must have the same number of dimensions %d and %d for equal\n", tensor1->ndim, tensor2->ndim);
            exit(1);
        }

        if (strcmp(tensor1->device, tensor2->device) != 0) {
            fprintf(stderr, "Tensors must be on the same device: %s and %s\n", tensor1->device, tensor2->device);
            exit(1);
        }

        int ndim = tensor1->ndim;
        int* shape = (int*)malloc(ndim * sizeof(int));
        if (shape == NULL) {
            fprintf(stderr, "Memory allocation failed\n");
            exit(1);
        }

        for (int i = 0; i < ndim; i++) {
            if (tensor1->shape[i] != tensor2->shape[i]) {
                fprintf(stderr, "Tensors must have the same shape %d and %d at index %d for equal\n", tensor1->shape[i], tensor2->shape[i], i);
                exit(1);
            }
            shape[i] = tensor1->shape[i];
        }        
        
        if (strcmp(tensor1->device, "cpu") == 0) {
            float* result_data = (float*)malloc(tensor1->size * sizeof(float));
            if (result_data == NULL) {
                fprintf(stderr, "Memory allocation failed\n");
                exit(1);
            }
            equal_tensor_cpu(tensor1, tensor2, result_data);
            return create_tensor(result_data, shape, ndim, tensor1->device);
        } 
        else {
            float* result_data;
            cudaMalloc((void **)&result_data, tensor1->size * sizeof(float));
            equal_tensor_cuda(tensor1, tensor2, result_data);
            return create_tensor(result_data, shape, ndim, tensor1->device);         
        }     
    }

    Tensor* equal_broadcasted_tensor(Tensor* tensor1, Tensor* tensor2) {

        if (strcmp(tensor1->device, tensor2->device) != 0) {
            fprintf(stderr, "Tensors must be on the same device: %s and %s\n", tensor1->device, tensor2->device);
            exit(1);
        }

        int max_ndim = tensor1->ndim > tensor2->ndim ? tensor1->ndim : tensor2->ndim;

        // Determine the broadcasted shape
        int* broadcasted_shape = (int*)malloc(max_ndim * sizeof(int));
        if (broadcasted_shape == NULL) {
            fprintf(stderr, "Memory allocation failed\n");
            exit(1);
        }
        for (int i = 0; i < max_ndim; i++) {
            int dim1 = i < tensor1->ndim ? tensor1->shape[tensor1->ndim - 1 - i] : 1;
            int dim2 = i < tensor2->ndim ? tensor2->shape[tensor2->ndim - 1 - i] : 1;
            if (dim1 != dim2 && dim1 != 1 && dim2 != 1) {
                fprintf(stderr, "Shapes are not compatible for broadcasting\n");
                exit(1);
            }
            broadcasted_shape[max_ndim - 1 - i] = dim1 > dim2 ? dim1 : dim2;
        }

        int broadcasted_size = 1;
        for (int i = 0; i < max_ndim; i++) {
            broadcasted_size *= broadcasted_shape[i];
        }

        if (strcmp(tensor1->device, "cpu") == 0) {
            float* result_data = (float*)malloc(broadcasted_size * sizeof(float));
            if (result_data == NULL) {
                fprintf(stderr, "Memory allocation failed\n");
                exit(1);
            }

            equal_broadcasted_tensor_cpu(tensor1, tensor2, result_data, broadcasted_shape, broadcasted_size);
            return create_tensor(result_data, broadcasted_shape, max_ndim, tensor1->device);
        } 
        else {
            float* result_data;
            cudaMalloc((void **)&result_data, broadcasted_size * sizeof(float));
            equal_broadcasted_tensor_cuda(tensor1, tensor2, result_data, broadcasted_shape, broadcasted_size);
            return create_tensor(result_data, broadcasted_shape, max_ndim, tensor1->device);
        }
    }


    Tensor* ones_like_tensor(Tensor* tensor) {
        int ndim = tensor->ndim;
        int* shape = (int*)malloc(ndim * sizeof(int));
        if (shape == NULL) {
            fprintf(stderr, "Memory allocation failed\n");
            exit(1);
        }

        for (int i = 0; i < ndim; i++) {
            shape[i] = tensor->shape[i];
        }

        if (strcmp(tensor->device, "cpu") == 0) {
            float* result_data = (float*)malloc(tensor->size * sizeof(float));
            if (result_data == NULL) {
                fprintf(stderr, "Memory allocation failed\n");
                exit(1);
            }
            ones_like_tensor_cpu(tensor, result_data);
            return create_tensor(result_data, shape, ndim, tensor->device);
        } 
        else {
            float* result_data;
            cudaMalloc((void **)&result_data, tensor->size * sizeof(float));
            ones_like_tensor_cuda(tensor, result_data);
            return create_tensor(result_data, shape, ndim, tensor->device);
        }
    }

    Tensor* zeros_like_tensor(Tensor* tensor) {
        int ndim = tensor->ndim;
        int* shape = (int*)malloc(ndim * sizeof(int));
        if (shape == NULL) {
            fprintf(stderr, "Memory allocation failed\n");
            exit(1);
        }

        for (int i = 0; i < ndim; i++) {
            shape[i] = tensor->shape[i];
        }

        if (strcmp(tensor->device, "cpu") == 0) {
            float* result_data = (float*)malloc(tensor->size * sizeof(float));
            if (result_data == NULL) {
                fprintf(stderr, "Memory allocation failed\n");
                exit(1);
            }
            zeros_like_tensor_cpu(tensor, result_data);
            return create_tensor(result_data, shape, ndim, tensor->device);
        } 
        else {
            float* result_data;
            cudaMalloc((void **)&result_data, tensor->size * sizeof(float));
            zeros_like_tensor_cuda(tensor, result_data);
            return create_tensor(result_data, shape, ndim, tensor->device);
        }
    }

    Tensor* sin_tensor(Tensor* tensor) {
        int ndim = tensor->ndim;
        int* shape = (int*)malloc(ndim * sizeof(int));
        if (shape == NULL) {
            fprintf(stderr, "Memory allocation failed\n");
            exit(1);
        }

        for (int i = 0; i < ndim; i++) {
            shape[i] = tensor->shape[i];
        }

        if (strcmp(tensor->device, "cpu") == 0) {
            float* result_data = (float*)malloc(tensor->size * sizeof(float));
            if (result_data == NULL) {
                fprintf(stderr, "Memory allocation failed\n");
                exit(1);
            }
            sin_tensor_cpu(tensor, result_data);
            return create_tensor(result_data, shape, ndim, tensor->device);
        } 
        else {
            float* result_data;
            cudaMalloc((void **)&result_data, tensor->size * sizeof(float));
            sin_tensor_cuda(tensor, result_data);
            return create_tensor(result_data, shape, ndim, tensor->device);
        }
    }

    Tensor* cos_tensor(Tensor* tensor) {

        int ndim = tensor->ndim;
        int* shape = (int*)malloc(ndim * sizeof(int));
        if (shape == NULL) {
            fprintf(stderr, "Memory allocation failed\n");
            exit(1);
        }

        for (int i = 0; i < ndim; i++) {
            shape[i] = tensor->shape[i];
        }

        if (strcmp(tensor->device, "cpu") == 0) {
            float* result_data = (float*)malloc(tensor->size * sizeof(float));
            if (result_data == NULL) {
                fprintf(stderr, "Memory allocation failed\n");
                exit(1);
            }
            cos_tensor_cpu(tensor, result_data);
            return create_tensor(result_data, shape, ndim, tensor->device);
        } 
        else {
            float* result_data;
            cudaMalloc((void **)&result_data, tensor->size * sizeof(float));
            cos_tensor_cuda(tensor, result_data);
            return create_tensor(result_data, shape, ndim, tensor->device);
        }
    }

    Tensor* sigmoid_tensor(Tensor* tensor) {
        int ndim = tensor->ndim;
        int* shape = (int*)malloc(ndim * sizeof(int));
        if (shape == NULL) {
            fprintf(stderr, "Memory allocation failed\n");
            exit(1);
        }

        for (int i = 0; i < ndim; i++) {
            shape[i] = tensor->shape[i];
        }

        if (strcmp(tensor->device, "cpu") == 0) {
            float* result_data = (float*)malloc(tensor->size * sizeof(float));
            if (result_data == NULL) {
                fprintf(stderr, "Memory allocation failed\n");
                exit(1);
            }
            sigmoid_tensor_cpu(tensor, result_data);
            return create_tensor(result_data, shape, ndim, tensor->device);
        } 
        else {
            float* result_data;
            cudaMalloc((void **)&result_data, tensor->size * sizeof(float));
            sigmoid_tensor_cuda(tensor, result_data);
            return create_tensor(result_data, shape, ndim, tensor->device); 
        }
    }

    Tensor* transpose_tensor(Tensor* tensor) {
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

        if (strcmp(tensor->device, "cpu") == 0) {
            float* result_data = (float*)malloc(size * sizeof(float));
            if (result_data == NULL) {
                fprintf(stderr, "Memory allocation failed\n");
                exit(1);
            }
            switch (ndim) {
                case 1:
                    transpose_1D_tensor_cpu(tensor, result_data);
                    break;
                case 2:
                    transpose_2D_tensor_cpu(tensor, result_data);
                    break;
                case 3:
                    transpose_3D_tensor_cpu(tensor, result_data);
                    break;
                default:
                    fprintf(stderr, "Transpose only supports tensors up to 3 dimensions.\n");
                    exit(-1);
            }
            return create_tensor(result_data, shape, ndim, tensor->device);
        } 
        else {
            float* result_data;
            cudaMalloc((void **)&result_data, size * sizeof(float));
            switch (ndim) {
                case 1:
                    transpose_1D_tensor_cuda(tensor, result_data);
                    break;
                case 2:
                    transpose_2D_tensor_cuda(tensor, result_data);
                    break;
                case 3:
                    transpose_3D_tensor_cuda(tensor, result_data);
                    break;
                default:
                    fprintf(stderr, "Transpose only supports tensors up to 3 dimensions.\n");
                    exit(-1);
            }
            return create_tensor(result_data, shape, ndim, tensor->device);
        }
    }

    Tensor* transpose_axes_tensor(Tensor* tensor, int axis1, int axis2) {
        int ndim = tensor->ndim;
        int* shape = (int*)malloc(ndim * sizeof(int));
        if (shape == NULL) {
            fprintf(stderr, "Memory allocation failed\n");
            exit(-1);
        }

        for (int i = 0; i < ndim; i++) {
            shape[i] = tensor->shape[i];
        }

        shape[axis1] = tensor->shape[axis2];
        shape[axis2] = tensor->shape[axis1];

        int size = tensor->size;

        if (strcmp(tensor->device, "cpu") == 0) {
            float* result_data = (float*)malloc(size * sizeof(float));
            if (result_data == NULL) {
                fprintf(stderr, "Memory allocation failed\n");
                exit(1);
            }
            assign_tensor_cpu(tensor, result_data);
            
            Tensor* new_tensor = create_tensor(result_data, shape, ndim, tensor->device);
            for (int i = 0; i < ndim; i++) {
                new_tensor->strides[i] = tensor->strides[i];
            }
            new_tensor->strides[axis1] = tensor->strides[axis2];
            new_tensor->strides[axis2] = tensor->strides[axis1];
            make_contiguous(new_tensor);
            return new_tensor;
        } 
        else {
            float* result_data;
            cudaMalloc((void **)&result_data, size * sizeof(float));
            assign_tensor_cuda(tensor, result_data);

            Tensor* new_tensor = create_tensor(result_data, shape, ndim, tensor->device);
            for (int i = 0; i < ndim; i++) {
                new_tensor->strides[i] = tensor->strides[i];
            }
            new_tensor->strides[axis1] = tensor->strides[axis2];
            new_tensor->strides[axis2] = tensor->strides[axis1];
            make_contiguous(new_tensor);
            return new_tensor;
        }
    }

    void make_contiguous(Tensor* tensor) {
        
        int* new_strides = (int*)malloc(tensor->ndim * sizeof(int));
        if (new_strides == NULL) {
            fprintf(stderr, "Memory allocation failed\n");
        }

        // Calculate new strides assuming C-contiguous order
        int stride = 1;
        for (int i = tensor->ndim - 1; i >= 0; i--) {
            new_strides[i] = stride;
            stride *= tensor->shape[i];
        }

        if (strcmp(tensor->device, "cpu") == 0) {
            float* result_data = (float*)malloc(tensor->size * sizeof(float));
            if (result_data == NULL) {
                fprintf(stderr, "Memory allocation failed\n");
            }
            make_contiguous_tensor_cpu(tensor, result_data, new_strides);
        } else {
            float* result_data;
            cudaMalloc((void **)&result_data, tensor->size * sizeof(float));
            make_contiguous_tensor_cuda(tensor, result_data, new_strides);
        }
    }
}
