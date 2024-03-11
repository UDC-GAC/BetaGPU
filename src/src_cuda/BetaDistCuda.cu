#include "BetaDistCuda.hpp"

#include <cuda_fp16.h>

#ifdef DEBUG

#include <chrono>
#include <iostream>

using std::cerr;
using std::endl;

using profile_clock_t = std::chrono::high_resolution_clock;
using profile_duration_t = std::chrono::duration<double>;

#endif

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

#ifdef DEBUG
std::vector<double> betapdf_cuda_times(std::vector<double> x, double alpha, double beta, GPU_Type precision){
    cudaEvent_t t1, t2, t3, t4;
    float elapsedMemcpyCG, elapsedKernel, elapsedMemcpyGC, elapsedTotal;
    cudaEventCreate(&t1);
    cudaEventCreate(&t2);
    cudaEventCreate(&t3);
    cudaEventCreate(&t4);

    auto start = profile_clock_t::now();

    cudaEventRecord(t1, 0);
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

    cudaEventRecord(t2, 0);
    cudaEventSynchronize(t2);
    cudaEventElapsedTime(&elapsedMemcpyCG, t1, t2);

    // Launch the kernel
    int block_size = 256;
    int n_blocks = x.size() / block_size + (x.size() % block_size == 0 ? 0 : 1);
    if (precision == GPU_Type::DOUBLE)
        betapdf_kernel<<<n_blocks, block_size>>>(d_x, d_y, alpha, beta, x.size());
    if (precision == GPU_Type::FLOAT)
        betapdf_kernel_f<<<n_blocks, block_size>>>(d_x_f, d_y_f, alpha, beta, x.size());
    if (precision == GPU_Type::HALF)
        betapdf_kernel_h<<<n_blocks, block_size>>>(d_x_f, d_y_f, alpha_f, beta_f, x.size());

    cudaEventRecord(t3, 0);
    cudaEventSynchronize(t3);
    cudaEventElapsedTime(&elapsedKernel, t2, t3);

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

    cudaEventRecord(t4, 0);
    cudaEventSynchronize(t4);
    cudaEventElapsedTime(&elapsedMemcpyGC, t3, t4);
    cudaEventElapsedTime(&elapsedTotal, t1, t4);

    auto end = profile_clock_t::now();

    cerr << "Full function time(chrono) = " << profile_duration_t(end - start).count() << endl;
    cerr << "Full function time(events) = " << elapsedTotal / 1000 << endl;
    cerr << " Kernel execution time = " << elapsedKernel / 1000 << endl;
    cerr << " Memory transfer time = " << elapsedMemcpyCG / 1000 << " + " << elapsedMemcpyGC / 1000 << endl;

    return y;
}
#endif

std::vector<double> betacdf_cuda(std::vector<double> x, double alpha, double beta){
    return std::vector<double>();
}
