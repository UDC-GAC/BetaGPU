#include "BetaDistCuda.hpp"

#include <cuda_fp16.h>


__global__ void betapdf_kernel(double *x, double *y, double alpha, double beta, size_t size){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size){
        y[idx] = pow(x[idx], alpha - 1) * pow(1 - x[idx], beta - 1) * exp(lgamma(alpha + beta) - lgamma(alpha) - lgamma(beta));
    }
}

__global__ void betapdf_kernel_f(float *x, float *y, float alpha, float beta, size_t size){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size){
        y[idx] = powf(x[idx], alpha - 1) * powf(1 - x[idx], beta - 1) * expf(lgammaf(alpha + beta) - lgammaf(alpha) - lgammaf(beta));
    }
}

__global__ void betapdf_kernel_h(float *x, float *y, float alpha, float beta, size_t size){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size){
        y[idx] = powf(x[idx], alpha - 1) * powf(1 - x[idx], beta - 1) * expf(lgammaf(alpha + beta) - lgammaf(alpha) - lgammaf(beta));
    }
}

// CUDA kernel launch to compute the beta distribution
std::vector<double> betapdf_cuda(std::vector<double> x, double alpha, double beta, GPU_Type precision){
    // Allocate memory on the device
    double *d_x, *d_y;
    float *d_x_f, *d_y_f, alpha_f, beta_f;
    if (precision == GPU_Type::DOUBLE){
        cudaMalloc(&d_x, x.size() * sizeof(double));
        cudaMalloc(&d_y, x.size() * sizeof(double));
    }
    if (precision == GPU_Type::FLOAT || precision == GPU_Type::HALF){
        alpha_f = (float)alpha;
        beta_f = (float)beta;
        cudaMalloc(&d_x_f, x.size() * sizeof(float));
        cudaMalloc(&d_y_f, x.size() * sizeof(float));
    }

    // Copy the data to the device
    if (precision == GPU_Type::DOUBLE){
        cudaMemcpy(d_x, x.data(), x.size() * sizeof(double), cudaMemcpyHostToDevice);
    }
    if (precision == GPU_Type::FLOAT || precision == GPU_Type::HALF){
        std::vector<float> x_f(x.begin(), x.end());
        cudaMemcpy(d_x_f, x_f.data(), x_f.size() * sizeof(float), cudaMemcpyHostToDevice);
    }

    // Launch the kernel
    int block_size = 256;
    int n_blocks = x.size() / block_size + (x.size() % block_size == 0 ? 0 : 1);
    if (precision == GPU_Type::DOUBLE)
        betapdf_kernel<<<n_blocks, block_size>>>(d_x, d_y, alpha, beta, x.size());
    if (precision == GPU_Type::FLOAT)
        betapdf_kernel_f<<<n_blocks, block_size>>>(d_x_f, d_y_f, alpha, beta, x.size());
    if (precision == GPU_Type::HALF)
        betapdf_kernel_h<<<n_blocks, block_size>>>(d_x_f, d_y_f, alpha_f, beta_f, x.size());

    // Copy the result back to the host
    std::vector<double> y(x.size());
    if (precision == GPU_Type::DOUBLE)
        cudaMemcpy(y.data(), d_y, x.size() * sizeof(double), cudaMemcpyDeviceToHost);
    if (precision == GPU_Type::FLOAT || precision == GPU_Type::HALF){
        std::vector<float> y_f(x.size());
        cudaMemcpy(y_f.data(), d_y_f, x.size() * sizeof(float), cudaMemcpyDeviceToHost);
        y = std::vector<double>(y_f.begin(), y_f.end());
    }

    // Free the memory on the device
    if (precision == GPU_Type::DOUBLE){
        cudaFree(d_x);
        cudaFree(d_y);
    }
    if (precision == GPU_Type::FLOAT || precision == GPU_Type::HALF){
        cudaFree(d_x_f);
        cudaFree(d_y_f);
    }

    return y;
}

std::vector<double> betapdf_cuda_f(std::vector<double> x, double alpha, double beta){
    float alpha_f = (float)alpha;
    float beta_f = (float)beta;
    std::vector<float> x_f(x.begin(), x.end());

    // Allocate memory on the device
    float *d_x, *d_y;
    cudaMalloc(&d_x, x_f.size() * sizeof(float));
    cudaMalloc(&d_y, x_f.size() * sizeof(float));

    // Copy the data to the device
    cudaMemcpy(d_x, x_f.data(), x_f.size() * sizeof(float), cudaMemcpyHostToDevice);

    // Launch the kernel
    int block_size = 256;
    int n_blocks = x_f.size() / block_size + (x_f.size() % block_size == 0 ? 0 : 1);
    betapdf_kernel_f<<<n_blocks, block_size>>>(d_x, d_y, alpha_f, beta_f, x_f.size());

    // Copy the result back to the host
    std::vector<float> y_f(x_f.size());
    cudaMemcpy(y_f.data(), d_y, x_f.size() * sizeof(float), cudaMemcpyDeviceToHost);

    // Free the memory on the device
    cudaFree(d_x);
    cudaFree(d_y);

    return std::vector<double>(y_f.begin(), y_f.end());
}


std::vector<double> betacdf_cuda(std::vector<double> x, double alpha, double beta){
    return std::vector<double>();
}
