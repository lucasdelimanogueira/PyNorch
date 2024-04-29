#ifndef CUDA_KERNEL_H_
#define CUDA_KERNEL_H_

#ifdef __NVCC__
    __global__ void add_tensor_cuda(float* data1, float* data2, float* result_data, int size);
#endif

#endif /* CUDA_KERNEL_H_ */
