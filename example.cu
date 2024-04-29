#include <stdio.h>

// Kernel definition
__global__ void AddTwoVectors(float A[], float B[], float C[]) {
    int i = threadIdx.x;
    C[i] = A[i] + B[i];
}

int main() {
    printf("kkkkkkkkkkkkkkkkkkkk");
    int N = 1000; // Size of the vectors
    float A[N], B[N], C[N]; // Arrays for vectors A, B, and C

    // Initialize vectors A and B
    for (int i = 0; i < N; ++i) {
        A[i] = 1;
        B[i] = 3;
    }

    float *d_A, *d_B, *d_C; // Device pointers for vectors A, B, and C

    // Allocate memory on the device for vectors A, B, and C
    cudaMalloc((void **)&d_A, N * sizeof(float));
    cudaMalloc((void **)&d_B, N * sizeof(float));
    cudaMalloc((void **)&d_C, N * sizeof(float));
    printf("\nkk2kkkkkkkkkkkkkkkkkk");

    // Copy vectors A and B from host to device
    cudaMemcpy(d_A, A, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, N * sizeof(float), cudaMemcpyHostToDevice);
    printf("\nkk3kkkkkkkkkkkkkkkkkk");

    // Kernel invocation with N threads
    AddTwoVectors<<<1, N>>>(d_A, d_B, d_C);
    printf("\nkk4kkkkkkkkkkkkkkkkkk");

    // Check for error
    cudaError_t error = cudaGetLastError();
    if(error != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(error));
        exit(-1);
    }
    
    // Waits untill all CUDA threads are executed
    cudaDeviceSynchronize();
    
    // Copy vector C from device to host
    cudaMemcpy(C, d_C, N * sizeof(float), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}