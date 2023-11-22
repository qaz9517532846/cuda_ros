#include <iostream>

__global__ void cudaKernel() {
    // Your CUDA kernel code here
    printf("Hello from CUDA kernel!\n");
}

int main() {
    // Launch CUDA kernel
    cudaKernel<<<1, 1>>>();
    cudaDeviceSynchronize();

    std::cout << "CUDA program executed successfully in ROS environment." << std::endl;

    return 0;
}