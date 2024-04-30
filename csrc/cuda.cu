#include "tensor.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define THREADS_PER_BLOCK 128

__host__ void cpu_to_cuda(Tensor* tensor) {
    
    float* data_tmp;
    cudaMalloc((void **)&data_tmp, tensor->size * sizeof(float));
    cudaMemcpy(data_tmp, tensor->data, tensor->size * sizeof(float), cudaMemcpyHostToDevice);

    tensor->data = data_tmp;

    const char* device_str = "cuda";
    tensor->device = (char*)malloc(strlen(device_str) + 1);
    strcpy(tensor->device, device_str); 

    printf("Successfully sent tensor to: %s\n", tensor->device);
}

__host__ void cuda_to_cpu(Tensor* tensor) {
    float* data_tmp = (float*)malloc(tensor->size * sizeof(float));

    cudaMemcpy(data_tmp, tensor->data, tensor->size * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(tensor->data);

    tensor->data = data_tmp;

    const char* device_str = "cpu";
    tensor->device = (char*)malloc(strlen(device_str) + 1);
    strcpy(tensor->device, device_str); 

    printf("Successfully sent tensor to: %s\n", tensor->device);
}

__global__ void add_tensor_cuda_kernel(float* data1, float* data2, float* result_data, int size) {
    
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        result_data[i] = data1[i] + data2[i];
    }
}

__host__ void add_tensor_cuda(Tensor* tensor1, Tensor* tensor2, float* result_data) {
    
    int number_of_blocks = (tensor1->size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    add_tensor_cuda_kernel<<<number_of_blocks, THREADS_PER_BLOCK>>>(tensor1->data, tensor2->data, result_data, tensor1->size);

    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(error));
        exit(-1);
    }

    cudaDeviceSynchronize();
}

__global__ void sub_tensor_cuda_kernel(float* data1, float* data2, float* result_data, int size) {
   
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        result_data[i] = data1[i] - data2[i];
    }
}

__host__ void sub_tensor_cuda(Tensor* tensor1, Tensor* tensor2, float* result_data) {
    
    int number_of_blocks = (tensor1->size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    sub_tensor_cuda_kernel<<<number_of_blocks, THREADS_PER_BLOCK>>>(tensor1->data, tensor2->data, result_data, tensor1->size);

    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(error));
        exit(-1);
    }

    cudaDeviceSynchronize();
}

__global__ void elementwise_mul_tensor_cuda_kernel(float* data1, float* data2, float* result_data, int size) {
    
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        result_data[i] = data1[i] * data2[i];
    }
}

__host__ void elementwise_mul_tensor_cuda(Tensor* tensor1, Tensor* tensor2, float* result_data) {
    
    int number_of_blocks = (tensor1->size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    elementwise_mul_tensor_cuda_kernel<<<number_of_blocks, THREADS_PER_BLOCK>>>(tensor1->data, tensor2->data, result_data, tensor1->size);

    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(error));
        exit(-1);
    }

    cudaDeviceSynchronize();
}

__global__ void matmul_tensor_cuda_kernel(float* data1, float* data2, float* result_data, int rows1, int cols1, int cols2) {    

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < rows1 && col < cols2) {
        float sum = 0.0;
        for (int k = 0; k < cols1; k++) {
            sum += data1[row * cols1 + k] * data2[k * cols2 + col];
        }
        result_data[row * cols2 + col] = sum;
    }

}

__host__ void matmul_tensor_cuda(Tensor* tensor1, Tensor* tensor2, float* result_data) {
    
    int rows1 = tensor1->shape[0];
    int cols1 = tensor1->shape[1];
    int cols2 = tensor2->shape[1];

    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((cols2 + threadsPerBlock.x - 1) / threadsPerBlock.x, (rows1 + threadsPerBlock.y - 1) / threadsPerBlock.y);
    matmul_tensor_cuda_kernel<<<numBlocks, threadsPerBlock>>>(tensor1->data, tensor2->data, result_data, rows1, cols1, cols2);


    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(error));
        exit(-1);
    }

    cudaDeviceSynchronize();
}

__global__ void pow_tensor_cuda_kernel(float* data, float power, float* result_data, int size) {
    
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        result_data[i] = powf(data[i], power);
    }
}

__host__ void pow_tensor_cuda(Tensor* tensor, float power, float* result_data) {
    
    int number_of_blocks = (tensor->size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    pow_tensor_cuda_kernel<<<number_of_blocks, THREADS_PER_BLOCK>>>(tensor->data, power, result_data, tensor->size);

    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(error));
        exit(-1);
    }

    cudaDeviceSynchronize();
}


