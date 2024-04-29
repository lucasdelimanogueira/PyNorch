#ifdef __CUDACC__
    __global__ void add_tensor_cuda(float* data1, float* data2, float* result_data, int size) {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < size) 
            result_data[i] = data1[i] + data2[i];
    }
#endif