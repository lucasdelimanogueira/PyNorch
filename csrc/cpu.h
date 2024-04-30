void add_tensor_cpu(Tensor* tensor1, Tensor* tensor2, float* result_data);
void sub_tensor_cpu(Tensor* tensor1, Tensor* tensor2, float* result_data);
void elementwise_mul_tensor_cpu(Tensor* tensor1, Tensor* tensor2, float* result_data);

void pow_tensor_cpu(Tensor* tensor, float power, float* result_data);