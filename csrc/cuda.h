#ifndef CUDA_KERNEL_H_
#define CUDA_KERNEL_H_

    __host__ void cpu_to_cuda(Tensor* tensor);
    __host__ void cuda_to_cpu(Tensor* tensor);
    
    __global__ void add_tensor_cuda_kernel(float* data1, float* data2, float* result_data, int size);
    __host__ void add_tensor_cuda(Tensor* tensor1, Tensor* tensor2, float* result_data);
    
    __global__ void sub_tensor_cuda_kernel(float* data1, float* data2, float* result_data, int size);
    __host__ void sub_tensor_cuda(Tensor* tensor1, Tensor* tensor2, float* result_data);
    
    __global__ void elementwise_mul_tensor_cuda_kernel(float* data1, float* data2, float* result_data, int size);
    __host__ void elementwise_mul_tensor_cuda(Tensor* tensor1, Tensor* tensor2, float* result_data);

    __global__ void pow_tensor_cuda_kernel(float* data, float power, float* result_data, int size);
    __host__ void pow_tensor_cuda(Tensor* tensor, float power, float* result_data);

#endif /* CUDA_KERNEL_H_ */
