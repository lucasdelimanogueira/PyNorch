#include "tensor.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define THREADS_PER_BLOCK 128
#define TILE_SIZE 32

__host__ void cpu_to_cuda(Tensor* tensor) {
    
    float* data_tmp;
    cudaMalloc((void **)&data_tmp, tensor->size * sizeof(float));
    cudaMemcpy(data_tmp, tensor->data, tensor->size * sizeof(float), cudaMemcpyHostToDevice);

    tensor->data = data_tmp;

    const char* device_str = "cuda";
    tensor->device = (char*)malloc(strlen(device_str) + 1);
    strcpy(tensor->device, device_str); 
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

__global__ void add_broadcasted_tensor_cuda_kernel(float* data1, float* data2, float* result_data, int* broadcasted_shape, int* strides1, int*strides2, int max_ndim, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= size) return;
    
    int index1 = 0, index2 = 0;
    int linear_index = i;
    for (int j = max_ndim - 1; j >= 0; j--) {
        int pos = linear_index % broadcasted_shape[j];
        linear_index /= broadcasted_shape[j];
        if (strides1[j] != 0) index1 += pos * strides1[j];
        if (strides2[j] != 0) index2 += pos * strides2[j];
    }
    result_data[i] = data1[index1] + data2[index2];
}

__host__ void add_broadcasted_tensor_cuda(Tensor* tensor1, Tensor* tensor2, float* result_data, int* broadcasted_shape, int broadcasted_size) {
    int max_ndim = tensor1->ndim > tensor2->ndim ? tensor1->ndim : tensor2->ndim;

    int* strides1 = (int*)malloc(max_ndim * sizeof(int));
    int* strides2 = (int*)malloc(max_ndim * sizeof(int));
    if (strides1 == NULL || strides2 == NULL) {
        fprintf(stderr, "Memory allocation failed\n");
        exit(1);
    }

    int stride1 = 1, stride2 = 1;
    for (int i = max_ndim - 1; i >= 0; i--) {
        int dim1 = i < tensor1->ndim ? tensor1->shape[tensor1->ndim - max_ndim + i] : 1;
        int dim2 = i < tensor2->ndim ? tensor2->shape[tensor2->ndim - max_ndim + i] : 1;
        strides1[i] = dim1 == broadcasted_shape[i] ? stride1 : 0;
        strides2[i] = dim2 == broadcasted_shape[i] ? stride2 : 0;
        stride1 *= (dim1 == broadcasted_shape[i]) ? dim1 : 1;
        stride2 *= (dim2 == broadcasted_shape[i]) ? dim2 : 1;
    }
    
    int* d_broadcasted_shape;
    int* d_strides1;
    int* d_strides2;

    cudaMalloc((void**)&d_broadcasted_shape, max_ndim * sizeof(int));
    cudaMemcpy(d_broadcasted_shape, broadcasted_shape, max_ndim * sizeof(int), cudaMemcpyHostToDevice);

    cudaMalloc((void**)&d_strides1, max_ndim * sizeof(int));
    cudaMemcpy(d_strides1, strides1, max_ndim * sizeof(int), cudaMemcpyHostToDevice);

    cudaMalloc((void**)&d_strides2, max_ndim * sizeof(int));
    cudaMemcpy(d_strides2, strides2, max_ndim * sizeof(int), cudaMemcpyHostToDevice);

    int number_of_blocks = (broadcasted_size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    add_broadcasted_tensor_cuda_kernel<<<number_of_blocks, THREADS_PER_BLOCK>>>(tensor1->data, tensor2->data, result_data, d_broadcasted_shape, d_strides1, d_strides2, max_ndim, broadcasted_size);

    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(error));
        exit(-1);
    }

    cudaDeviceSynchronize();
    cudaFree(d_broadcasted_shape);
}

__global__ void sum_tensor_cuda_kernel(float* data, float* result_data, int size) {
    __shared__ float partial_sum[THREADS_PER_BLOCK * sizeof(float)];

    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    partial_sum[tid] = (i < size) ? data[i] : 0;

    __syncthreads();

    // Perform block-wise reduction
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            partial_sum[tid] += partial_sum[tid + s];
        }
        __syncthreads();
    }

    // Write block sum to global memory
    if (tid == 0) {
        result_data[blockIdx.x] = partial_sum[0];
    }
}

__host__ void sum_tensor_cuda(Tensor* tensor, float* result_data) {
    cudaMemcpy(result_data, tensor->data, tensor->size * sizeof(float), cudaMemcpyHostToDevice);

    int num_blocks = (tensor->size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

    // First-level reduction
    sum_tensor_cuda_kernel<<<num_blocks, THREADS_PER_BLOCK>>>(tensor->data, result_data, tensor->size);

    // If necessary, perform multiple levels of reduction
    while (num_blocks > 1) {
        int num_blocks_next = (num_blocks + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
        sum_tensor_cuda_kernel<<<num_blocks_next, THREADS_PER_BLOCK>>>(result_data, result_data, num_blocks);
        num_blocks = num_blocks_next;
    }

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

__global__ void sub_broadcasted_tensor_cuda_kernel(float* data1, float* data2, float* result_data, int* broadcasted_shape, int* strides1, int*strides2, int max_ndim, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= size) return;
    
    int index1 = 0, index2 = 0;
    int linear_index = i;
    for (int j = max_ndim - 1; j >= 0; j--) {
        int pos = linear_index % broadcasted_shape[j];
        linear_index /= broadcasted_shape[j];
        if (strides1[j] != 0) index1 += pos * strides1[j];
        if (strides2[j] != 0) index2 += pos * strides2[j];
    }
    result_data[i] = data1[index1] - data2[index2];
}

__host__ void sub_broadcasted_tensor_cuda(Tensor* tensor1, Tensor* tensor2, float* result_data, int* broadcasted_shape, int broadcasted_size) {
    int max_ndim = tensor1->ndim > tensor2->ndim ? tensor1->ndim : tensor2->ndim;

    int* strides1 = (int*)malloc(max_ndim * sizeof(int));
    int* strides2 = (int*)malloc(max_ndim * sizeof(int));
    if (strides1 == NULL || strides2 == NULL) {
        fprintf(stderr, "Memory allocation failed\n");
        exit(1);
    }

    int stride1 = 1, stride2 = 1;
    for (int i = max_ndim - 1; i >= 0; i--) {
        int dim1 = i < tensor1->ndim ? tensor1->shape[tensor1->ndim - max_ndim + i] : 1;
        int dim2 = i < tensor2->ndim ? tensor2->shape[tensor2->ndim - max_ndim + i] : 1;
        strides1[i] = dim1 == broadcasted_shape[i] ? stride1 : 0;
        strides2[i] = dim2 == broadcasted_shape[i] ? stride2 : 0;
        stride1 *= (dim1 == broadcasted_shape[i]) ? dim1 : 1;
        stride2 *= (dim2 == broadcasted_shape[i]) ? dim2 : 1;
    }
    
    int* d_broadcasted_shape;
    int* d_strides1;
    int* d_strides2;

    cudaMalloc((void**)&d_broadcasted_shape, max_ndim * sizeof(int));
    cudaMemcpy(d_broadcasted_shape, broadcasted_shape, max_ndim * sizeof(int), cudaMemcpyHostToDevice);

    cudaMalloc((void**)&d_strides1, max_ndim * sizeof(int));
    cudaMemcpy(d_strides1, strides1, max_ndim * sizeof(int), cudaMemcpyHostToDevice);

    cudaMalloc((void**)&d_strides2, max_ndim * sizeof(int));
    cudaMemcpy(d_strides2, strides2, max_ndim * sizeof(int), cudaMemcpyHostToDevice);

    int number_of_blocks = (broadcasted_size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    sub_broadcasted_tensor_cuda_kernel<<<number_of_blocks, THREADS_PER_BLOCK>>>(tensor1->data, tensor2->data, result_data, d_broadcasted_shape, d_strides1, d_strides2, max_ndim, broadcasted_size);

    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(error));
        exit(-1);
    }

    cudaDeviceSynchronize();
    cudaFree(d_broadcasted_shape);
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

__global__ void scalar_mul_tensor_cuda_kernel(float* data, float scalar, float* result_data, int size) {
    
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        result_data[i] = scalar * data[i];
    }
}

__host__ void scalar_mul_tensor_cuda(Tensor* tensor, float scalar, float* result_data) {
    
    int number_of_blocks = (tensor->size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    scalar_mul_tensor_cuda_kernel<<<number_of_blocks, THREADS_PER_BLOCK>>>(tensor->data, scalar, result_data, tensor->size);

    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(error));
        exit(-1);
    }

    cudaDeviceSynchronize();
}

__global__ void scalar_div_tensor_cuda_kernel(float scalar, float* data, float* result_data, int size) {
    
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        result_data[i] = scalar / data[i];
    }
}

__host__ void scalar_div_tensor_cuda(float scalar, Tensor* tensor, float* result_data) {
    
    int number_of_blocks = (tensor->size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    scalar_div_tensor_cuda_kernel<<<number_of_blocks, THREADS_PER_BLOCK>>>(scalar, tensor->data, result_data, tensor->size);

    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(error));
        exit(-1);
    }

    cudaDeviceSynchronize();
}

__global__ void tensor_div_scalar_cuda_kernel(float* data, float scalar, float* result_data, int size) {
    
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        result_data[i] = data[i] / scalar;
    }
}

__host__ void tensor_div_scalar_cuda(Tensor* tensor, float scalar, float* result_data) {
    
    int number_of_blocks = (tensor->size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    tensor_div_scalar_cuda_kernel<<<number_of_blocks, THREADS_PER_BLOCK>>>(tensor->data, scalar, result_data, tensor->size);

    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(error));
        exit(-1);
    }

    cudaDeviceSynchronize();
}

__global__ void tensor_div_tensor_cuda_kernel(float* data1, float* data2, float* result_data, int size) {
    
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        result_data[i] = data1[i] / data2[i];
    }
}

__host__ void tensor_div_tensor_cuda(Tensor* tensor1, Tensor* tensor2, float* result_data) {
    
    int number_of_blocks = (tensor1->size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    tensor_div_tensor_cuda_kernel<<<number_of_blocks, THREADS_PER_BLOCK>>>(tensor1->data, tensor2->data, result_data, tensor1->size);

    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(error));
        exit(-1);
    }

    cudaDeviceSynchronize();
}

/*__global__ void matmul_tensor_cuda_kernel(float* data1, float* data2, float* result_data, int rows1, int cols1, int cols2) {    

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < rows1 && col < cols2) {
        float sum = 0.0;
        for (int k = 0; k < cols1; k++) {
            sum += data1[row * cols1 + k] * data2[k * cols2 + col];
        }
        result_data[row * cols2 + col] = sum;
    }

}*/

__global__ void matmul_tensor_cuda_kernel(float* data1, float* data2, float* result_data, int rows1, int cols1, int cols2) {    

    // Shared memory for tiles
    __shared__ float tile1[TILE_SIZE][TILE_SIZE];
    __shared__ float tile2[TILE_SIZE][TILE_SIZE];

    // Thread indices
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // Output position
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    float sum = 0.0;

    // Iterate over tiles
    for (int i = 0; i < (cols1 + TILE_SIZE - 1) / TILE_SIZE; ++i) {

        // Load tiles into shared memory
        if (row < rows1 && i * TILE_SIZE + tx < cols1)
            tile1[ty][tx] = data1[row * cols1 + i * TILE_SIZE + tx];
        else
            tile1[ty][tx] = 0.0;

        if (col < cols2 && i * TILE_SIZE + ty < cols1)
            tile2[ty][tx] = data2[(i * TILE_SIZE + ty) * cols2 + col];
        else
            tile2[ty][tx] = 0.0;

        // Synchronize threads
        __syncthreads();

        // Accumulate sum
        for (int k = 0; k < TILE_SIZE; ++k)
            sum += tile1[ty][k] * tile2[k][tx];

        // Synchronize threads
        __syncthreads();
    }

    // Write result to global memory
    if (row < rows1 && col < cols2)
        result_data[row * cols2 + col] = sum;
}


__host__ void matmul_tensor_cuda(Tensor* tensor1, Tensor* tensor2, float* result_data) {
    
    int rows1 = tensor1->shape[0];
    int cols1 = tensor1->shape[1];
    int cols2 = tensor2->shape[1];

    dim3 threadsPerBlock(16, 16);
    dim3 number_of_blocks((cols2 + threadsPerBlock.x - 1) / threadsPerBlock.x, (rows1 + threadsPerBlock.y - 1) / threadsPerBlock.y);
    matmul_tensor_cuda_kernel<<<number_of_blocks, threadsPerBlock>>>(tensor1->data, tensor2->data, result_data, rows1, cols1, cols2);


    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(error));
        exit(-1);
    }

    cudaDeviceSynchronize();
}

__global__ void batched_matmul_tensor_cuda_kernel(float* data1, float* data2, float* result_data, int batch_size, int rows1, int cols1, int cols2) {
    int batch = blockIdx.z;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < rows1 && col < cols2) {
        float sum = 0.0f;
        for (int k = 0; k < cols1; ++k) {
            sum += data1[batch * rows1 * cols1 + row * cols1 + k] * 
                   data2[batch * cols1 * cols2 + k * cols2 + col];
        }
        result_data[batch * rows1 * cols2 + row * cols2 + col] = sum;
    }    
}

__host__ void batched_matmul_tensor_cuda(Tensor* tensor1, Tensor* tensor2, float* result_data) {
    
    int batch_size = tensor2->shape[0];
    int rows1 = tensor1->shape[1];
    int cols1 = tensor1->shape[2];
    int cols2 = tensor2->shape[2];

    dim3 threadsPerBlock(16, 16);
    dim3 number_of_blocks((cols2 + threadsPerBlock.x - 1) / threadsPerBlock.x, (rows1 + threadsPerBlock.y - 1) / threadsPerBlock.y, batch_size);
    batched_matmul_tensor_cuda_kernel<<<number_of_blocks, threadsPerBlock>>>(tensor1->data, tensor2->data, result_data, batch_size, rows1, cols1, cols2);


    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(error));
        exit(-1);
    }

    cudaDeviceSynchronize();
}

__global__ void tensor_pow_scalar_cuda_kernel(float* data, float exponent, float* result_data, int size) {
    
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        result_data[i] = powf(data[i], exponent);
    }
}

__host__ void tensor_pow_scalar_cuda(Tensor* tensor, float exponent, float* result_data) {
    
    int number_of_blocks = (tensor->size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    tensor_pow_scalar_cuda_kernel<<<number_of_blocks, THREADS_PER_BLOCK>>>(tensor->data, exponent, result_data, tensor->size);

    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(error));
        exit(-1);
    }

    cudaDeviceSynchronize();
}

__global__ void scalar_pow_tensor_cuda_kernel(float base, float* data, float* result_data, int size) {
    
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        result_data[i] = powf(base, data[i]);
    }
}

__host__ void scalar_pow_tensor_cuda(float base, Tensor* tensor, float* result_data) {
    
    int number_of_blocks = (tensor->size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    scalar_pow_tensor_cuda_kernel<<<number_of_blocks, THREADS_PER_BLOCK>>>(base, tensor->data, result_data, tensor->size);

    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(error));
        exit(-1);
    }

    cudaDeviceSynchronize();
}

__global__ void log_tensor_cuda_kernel(float* data, float* result_data, int size) {
    
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        result_data[i] = logf(data[i]);
    }
}

__host__ void log_tensor_cuda(Tensor* tensor, float* result_data) {
    
    int number_of_blocks = (tensor->size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    log_tensor_cuda_kernel<<<number_of_blocks, THREADS_PER_BLOCK>>>(tensor->data, result_data, tensor->size);

    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(error));
        exit(-1);
    }

    cudaDeviceSynchronize();
}

__global__ void equal_tensor_cuda_kernel(float* data1, float* data2, float* result_data, int size) {
    
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        result_data[i] = (data1[i] == data2[i]) ? 1.0f : 0.0f;
    }
}

__host__ void equal_tensor_cuda(Tensor* tensor1, Tensor* tensor2, float* result_data) {
    
    int number_of_blocks = (tensor1->size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    equal_tensor_cuda_kernel<<<number_of_blocks, THREADS_PER_BLOCK>>>(tensor1->data, tensor2->data, result_data, tensor1->size);

    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(error));
        exit(-1);
    }

    cudaDeviceSynchronize();
}

__global__ void equal_broadcasted_tensor_cuda_kernel(float* data1, float* data2, float* result_data, int* broadcasted_shape, int* strides1, int*strides2, int max_ndim, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= size) return;
    
    int index1 = 0, index2 = 0;
    int linear_index = i;
    for (int j = max_ndim - 1; j >= 0; j--) {
        int pos = linear_index % broadcasted_shape[j];
        linear_index /= broadcasted_shape[j];
        if (strides1[j] != 0) index1 += pos * strides1[j];
        if (strides2[j] != 0) index2 += pos * strides2[j];
    }
    result_data[i] = (data1[index1] == data2[index2]) ? 1.0f : 0.0f;
}

__host__ void equal_broadcasted_tensor_cuda(Tensor* tensor1, Tensor* tensor2, float* result_data, int* broadcasted_shape, int broadcasted_size) {
    int max_ndim = tensor1->ndim > tensor2->ndim ? tensor1->ndim : tensor2->ndim;

    int* strides1 = (int*)malloc(max_ndim * sizeof(int));
    int* strides2 = (int*)malloc(max_ndim * sizeof(int));
    if (strides1 == NULL || strides2 == NULL) {
        fprintf(stderr, "Memory allocation failed\n");
        exit(1);
    }

    int stride1 = 1, stride2 = 1;
    for (int i = max_ndim - 1; i >= 0; i--) {
        int dim1 = i < tensor1->ndim ? tensor1->shape[tensor1->ndim - max_ndim + i] : 1;
        int dim2 = i < tensor2->ndim ? tensor2->shape[tensor2->ndim - max_ndim + i] : 1;
        strides1[i] = dim1 == broadcasted_shape[i] ? stride1 : 0;
        strides2[i] = dim2 == broadcasted_shape[i] ? stride2 : 0;
        stride1 *= (dim1 == broadcasted_shape[i]) ? dim1 : 1;
        stride2 *= (dim2 == broadcasted_shape[i]) ? dim2 : 1;
    }
    
    int* d_broadcasted_shape;
    int* d_strides1;
    int* d_strides2;

    cudaMalloc((void**)&d_broadcasted_shape, max_ndim * sizeof(int));
    cudaMemcpy(d_broadcasted_shape, broadcasted_shape, max_ndim * sizeof(int), cudaMemcpyHostToDevice);

    cudaMalloc((void**)&d_strides1, max_ndim * sizeof(int));
    cudaMemcpy(d_strides1, strides1, max_ndim * sizeof(int), cudaMemcpyHostToDevice);

    cudaMalloc((void**)&d_strides2, max_ndim * sizeof(int));
    cudaMemcpy(d_strides2, strides2, max_ndim * sizeof(int), cudaMemcpyHostToDevice);

    int number_of_blocks = (broadcasted_size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    equal_broadcasted_tensor_cuda_kernel<<<number_of_blocks, THREADS_PER_BLOCK>>>(tensor1->data, tensor2->data, result_data, d_broadcasted_shape, d_strides1, d_strides2, max_ndim, broadcasted_size);

    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(error));
        exit(-1);
    }

    cudaDeviceSynchronize();
    cudaFree(d_broadcasted_shape);
}


__global__ void ones_like_tensor_cuda_kernel(float* data, float* result_data, int size) {
    
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        result_data[i] = 1.0;
    }
}

__host__ void ones_like_tensor_cuda(Tensor* tensor, float* result_data) {
    
    int number_of_blocks = (tensor->size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    ones_like_tensor_cuda_kernel<<<number_of_blocks, THREADS_PER_BLOCK>>>(tensor->data, result_data, tensor->size);

    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(error));
        exit(-1);
    }

    cudaDeviceSynchronize();
}

__global__ void zeros_like_tensor_cuda_kernel(float* data, float* result_data, int size) {
    
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        result_data[i] = 0.0;
    }
}

__host__ void zeros_like_tensor_cuda(Tensor* tensor, float* result_data) {
    
    int number_of_blocks = (tensor->size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    zeros_like_tensor_cuda_kernel<<<number_of_blocks, THREADS_PER_BLOCK>>>(tensor->data, result_data, tensor->size);

    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(error));
        exit(-1);
    }

    cudaDeviceSynchronize();
}

__global__ void transpose_tensor_cuda_kernel(float* data, float* result_data, int rows, int cols) {
    int tid_x = blockIdx.x * blockDim.x + threadIdx.x;
    int tid_y = blockIdx.y * blockDim.y + threadIdx.y;

    if (tid_x < cols && tid_y < rows) {
        result_data[tid_x * rows + tid_y] = data[tid_y * cols + tid_x];
    }
}

__host__ void transpose_tensor_cuda(Tensor* tensor, float* result_data) {
    
    int rows = tensor->shape[0];
    int cols = tensor->shape[1];

    dim3 threadsPerBlock(16, 16);
    dim3 number_of_blocks((cols + threadsPerBlock.x - 1) / threadsPerBlock.x, (rows + threadsPerBlock.y - 1) / threadsPerBlock.y);
    transpose_tensor_cuda_kernel<<<number_of_blocks, threadsPerBlock>>>(tensor->data, result_data, rows, cols);


    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(error));
        exit(-1);
    }

    cudaDeviceSynchronize();
}

__global__ void assign_tensor_cuda_kernel(float* data, float* result_data, int size) {
    
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        result_data[i] = data[i];
    }
}

__host__ void assign_tensor_cuda(Tensor* tensor, float* result_data) {
    
    int number_of_blocks = (tensor->size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    assign_tensor_cuda_kernel<<<number_of_blocks, THREADS_PER_BLOCK>>>(tensor->data, result_data, tensor->size);

    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(error));
        exit(-1);
    }

    cudaDeviceSynchronize();
}

__global__ void sin_tensor_cuda_kernel(float* data, float* result_data, int size) {
    
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        result_data[i] = sinf(data[i]);
    }
}

__host__ void sin_tensor_cuda(Tensor* tensor, float* result_data) {
    
    int number_of_blocks = (tensor->size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    sin_tensor_cuda_kernel<<<number_of_blocks, THREADS_PER_BLOCK>>>(tensor->data, result_data, tensor->size);

    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(error));
        exit(-1);
    }

    cudaDeviceSynchronize();
}

__global__ void cos_tensor_cuda_kernel(float* data, float* result_data, int size) {
    
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        result_data[i] = cosf(data[i]);
    }
}

__host__ void cos_tensor_cuda(Tensor* tensor, float* result_data) {
    
    int number_of_blocks = (tensor->size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    cos_tensor_cuda_kernel<<<number_of_blocks, THREADS_PER_BLOCK>>>(tensor->data, result_data, tensor->size);

    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(error));
        exit(-1);
    }

    cudaDeviceSynchronize();
}

__global__ void sigmoid_tensor_cuda_kernel(float* data, float* result_data, int size) {
    
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < size) {
        // avoid overflow
        if (data[i] >= 0) {

            float z = expf(-data[i]);
            result_data[i] = 1 / (1 + z);

        } else {

            float z = expf(data[i]);
            result_data[i] = z / (1 + z);
        }
    }
}

__host__ void sigmoid_tensor_cuda(Tensor* tensor, float* result_data) {
    
    int number_of_blocks = (tensor->size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    sigmoid_tensor_cuda_kernel<<<number_of_blocks, THREADS_PER_BLOCK>>>(tensor->data, result_data, tensor->size);

    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(error));
        exit(-1);
    }

    cudaDeviceSynchronize();
}




