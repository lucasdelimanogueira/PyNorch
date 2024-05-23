#ifndef CUDA_KERNEL_H_
#define CUDA_KERNEL_H_

    __host__ void cpu_to_cuda(Tensor* tensor);
    __host__ void cuda_to_cpu(Tensor* tensor);
    
    __global__ void add_tensor_cuda_kernel(float* data1, float* data2, float* result_data, int size);
    __host__ void add_tensor_cuda(Tensor* tensor1, Tensor* tensor2, float* result_data);

    __global__ void add_broadcasted_tensor_cuda_kernel(float* data1, float* data2, float* result_data, int* broadcasted_shape, int* strides1, int*strides2, int max_ndim, int size);
    __host__ void add_broadcasted_tensor_cuda(Tensor* tensor1, Tensor* tensor2, float* result_data, int* broadcasted_shape, int broadcasted_size);

    __global__ void sub_broadcasted_tensor_cuda_kernel(float* data1, float* data2, float* result_data, int* broadcasted_shape, int* strides1, int*strides2, int max_ndim, int size);
    __host__ void sub_broadcasted_tensor_cuda(Tensor* tensor1, Tensor* tensor2, float* result_data, int* broadcasted_shape, int broadcasted_size);

    __global__ void sum_tensor_cuda_kernel(float* data, float* result_data);
    __global__ void sum_tensor_cuda_kernel_axis(float* data, float* result_data, int* strides, int* shape, int axis, int ndim, int axis_stride, int size, int result_size);
    __host__ void sum_tensor_cuda(Tensor* tensor, float* result_data, int axis);

    __global__ void max_tensor_cuda_kernel(float* data, float* result_data);
    __global__ void max_tensor_cuda_kernel_axis(float* data, float* result_data, int* strides, int* shape, int axis, int ndim, int axis_stride, int size, int result_size);
    __host__ void max_tensor_cuda(Tensor* tensor, float* result_data, int axis);

    __global__ void min_tensor_cuda_kernel(float* data, float* result_data);
    __global__ void min_tensor_cuda_kernel_axis(float* data, float* result_data, int* strides, int* shape, int axis, int ndim, int axis_stride, int size, int result_size);
    __host__ void min_tensor_cuda(Tensor* tensor, float* result_data, int axis);

    __global__ void sub_tensor_cuda_kernel(float* data1, float* data2, float* result_data, int size);
    __host__ void sub_tensor_cuda(Tensor* tensor1, Tensor* tensor2, float* result_data);
    
    __global__ void elementwise_mul_tensor_cuda_kernel(float* data1, float* data2, float* result_data, int size);
    __host__ void elementwise_mul_tensor_cuda(Tensor* tensor1, Tensor* tensor2, float* result_data);

    __global__ void scalar_mul_tensor_cuda_kernel(float* data, float scalar, float* result_data, int size);
    __host__ void scalar_mul_tensor_cuda(Tensor* tensor, float scalar, float* result_data);

    __global__ void scalar_div_tensor_cuda_kernel(float scalar, float* data, float* result_data, int size);
    __host__ void scalar_div_tensor_cuda(float scalar, Tensor* tensor, float* result_data);

    __global__ void tensor_div_scalar_cuda_kernel(float* data, float scalar, float* result_data, int size);
    __host__ void tensor_div_scalar_cuda(Tensor* tensor, float scalar, float* result_data);

    __global__ void tensor_div_tensor_cuda_kernel(float* data1, float* data2, float* result_data, int size);
    __host__ void tensor_div_tensor_cuda(Tensor* tensor1, Tensor* tensor2, float* result_data);
    
    __global__ void matmul_tensor_cuda_kernel(float* data1, float* data2, float* result_data, int rows1, int cols1, int cols2);
    __host__ void matmul_tensor_cuda(Tensor* tensor1, Tensor* tensor2, float* result_data);

    __global__ void batched_matmul_tensor_cuda_kernel(float* data1, float* data2, float* result_data, int batch_size, int rows1, int cols1, int cols2);
    __host__ void batched_matmul_tensor_cuda(Tensor* tensor1, Tensor* tensor2, float* result_data);

    __global__ void broadcasted_batched_matmul_tensor_cuda_kernel(float* data1, float* data2, float* result_data, int batch_size, int rows1, int cols1, int cols2);
    __host__ void broadcasted_batched_matmul_tensor_cuda(Tensor* tensor1, Tensor* tensor2, float* result_data);

    __global__ void tensor_pow_scalar_cuda_kernel(float* data, float exponent, float* result_data, int size);
    __host__ void tensor_pow_scalar_cuda(Tensor* tensor, float exponent, float* result_data);

    __global__ void scalar_pow_tensor_cuda_kernel(float base, float* data, float* result_data, int size);
    __host__ void scalar_pow_tensor_cuda(float base, Tensor* tensor, float* result_data);

    __global__ void log_tensor_cuda_kernel(float* data, float* result_data, int size);
    __host__ void log_tensor_cuda(Tensor* tensor, float* result_data);

    __global__ void equal_tensor_cuda_kernel(float* data1, float* data2, float* result_data, int size);
    __host__ void equal_tensor_cuda(Tensor* tensor1, Tensor* tensor2, float* result_data);

    __global__ void equal_broadcasted_tensor_cuda_kernel(float* data1, float* data2, float* result_data, int* broadcasted_shape, int* strides1, int*strides2, int max_ndim, int size);
    __host__ void equal_broadcasted_tensor_cuda(Tensor* tensor1, Tensor* tensor2, float* result_data, int* broadcasted_shape, int broadcasted_size);

    __global__ void ones_like_tensor_cuda_kernel(float* data, float* result_data, int size);
    __host__ void ones_like_tensor_cuda(Tensor* tensor, float* result_data);

    __global__ void zeros_like_tensor_cuda_kernel(float* data, float* result_data, int size);
    __host__ void zeros_like_tensor_cuda(Tensor* tensor, float* result_data);

    __global__ void transpose_1D_tensor_cuda_kernel(float* data, float* result_data, int rows, int cols);
    __host__ void transpose_1D_tensor_cuda(Tensor* tensor, float* result_data);

    __global__ void transpose_2D_tensor_cuda_kernel(float* data, float* result_data, int rows, int cols);
    __host__ void transpose_2D_tensor_cuda(Tensor* tensor, float* result_data);

    __global__ void transpose_3D_tensor_cuda_kernel(float* data, float* result_data, int batch, int rows, int cols);
    __host__ void transpose_3D_tensor_cuda(Tensor* tensor, float* result_data);

    __global__ void assign_tensor_cuda_kernel(float* data, float* result_data, int size);
    __host__ void assign_tensor_cuda(Tensor* tensor, float* result_data);

    __global__ void make_contiguous_tensor_cuda_kernel(float* data, float* result_data, int ndim, int size, int* strides, int* new_strides);
    __host__ void make_contiguous_tensor_cuda(Tensor* tensor, float* result_data, int* new_strides);

    __global__ void sin_tensor_cuda_kernel(float* data, float* result_data, int size);
    __host__ void sin_tensor_cuda(Tensor* tensor, float* result_data);

     __global__ void cos_tensor_cuda_kernel(float* data, float* result_data, int size);
    __host__ void cos_tensor_cuda(Tensor* tensor, float* result_data);

    __global__ void sigmoid_tensor_cuda_kernel(float* data, float* result_data, int size);
    __host__ void sigmoid_tensor_cuda(Tensor* tensor, float* result_data);




#endif /* CUDA_KERNEL_H_ */
