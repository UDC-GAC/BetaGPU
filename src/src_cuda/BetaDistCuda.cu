#include "BetaDistCuda.hpp"


__global__ void betapdf_kernel(double *x, double *y, double alpha, double beta, size_t size){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size){
        y[idx] = pow(x[idx], alpha - 1) * pow(1 - x[idx], beta - 1) * exp(lgamma(alpha + beta) - lgamma(alpha) - lgamma(beta));
    }
}

// CUDA kernel to compute the beta distribution
std::vector<double> betapdf_cuda(std::vector<double> x, double alpha, double beta){
    // Allocate memory on the device
    double *d_x, *d_y;
    cudaMalloc(&d_x, x.size() * sizeof(double));
    cudaMalloc(&d_y, x.size() * sizeof(double));

    // Copy the data to the device
    cudaMemcpy(d_x, x.data(), x.size() * sizeof(double), cudaMemcpyHostToDevice);

    // Launch the kernel
    int block_size = 256;
    int n_blocks = x.size() / block_size + (x.size() % block_size == 0 ? 0 : 1);
    betapdf_kernel<<<n_blocks, block_size>>>(d_x, d_y, alpha, beta, x.size());

    // Copy the result back to the host
    std::vector<double> y(x.size());
    cudaMemcpy(y.data(), d_y, x.size() * sizeof(double), cudaMemcpyDeviceToHost);

    // Free the memory on the device
    cudaFree(d_x);
    cudaFree(d_y);

    return y;
}


std::vector<double> betacdf_cuda(std::vector<double> x, double alpha, double beta){
    return std::vector<double>();
}
