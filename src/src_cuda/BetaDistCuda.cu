#include "beta_cuda/BetaDistCuda.hpp"

#include <cuda_fp16.h>
#include <gsl/gsl_sf_gamma.h>
#include <stdexcept>

#include <omp.h>
  
#ifdef DEBUG

#include <chrono>
#include <iostream>

using std::cerr;
using std::endl;

using profile_clock_t = std::chrono::high_resolution_clock;
using profile_duration_t = std::chrono::duration<double>;

#endif

// Define a type for the function that launches the kernel
typedef void (*KernelLauncher)(const double*, double*, double, double, int, int, cudaStream_t stream);
typedef void (*KernelLauncherFloat)(const float*, float*, float, float, int, int, cudaStream_t stream);

const cudaStream_t CUDA_DEFAULT_STREAM = (cudaStream_t) 0;
constexpr int DEFAULT_BLOCK_SIZE = 256;


/* --------------- Beta PDF Kernels --------------- */


__global__ void betapdf_kernel(const double *x, double *y, double alpha, double beta, size_t size){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size){
        y[idx] = pow(x[idx], alpha - 1) * pow(1 - x[idx], beta - 1) * exp(lgamma(alpha + beta) - lgamma(alpha) - lgamma(beta));
    }
}

__global__ void betapdf_kernel_f(const float *x, float *y, float alpha, float beta, size_t size){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size){
        y[idx] = powf(x[idx], alpha - 1) * powf(1 - x[idx], beta - 1) * expf(lgammaf(alpha + beta) - lgammaf(alpha) - lgammaf(beta));
    }
}

__global__ void betapdf_kernel_h(const float *x, float *y, float alpha, float beta, size_t size){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size){
        y[idx] = powf(x[idx], alpha - 1) * powf(1 - x[idx], beta - 1) * expf(lgammaf(alpha + beta) - lgammaf(alpha) - lgammaf(beta));
    }
}


/* --------------- Beta CDF Kernels --------------- */


// Look https://github.com/ampl/gsl/blob/master/cdf/beta_inc.c#L26
__device__ __host__ double cuda_beta_cont_frac (const double a, const double b, const double x,
                const double epsabs) {
  const unsigned int max_iter = 512;    /* control iterations      */
  const double cutoff = 2.0 * CUDA_DBL_MIN;      /* control the zero cutoff */
  unsigned int iter_count = 0;
  double cf;

  /* standard initialization for continued fraction */
  double num_term = 1.0;
  double den_term = 1.0 - (a + b) * x / (a + 1.0);

  if (fabs (den_term) < cutoff)
    den_term = nan("");

  den_term = 1.0 / den_term;
  cf = den_term;

  while (iter_count < max_iter)
    {
      const int k = iter_count + 1;
      double coeff = k * (b - k) * x / (((a - 1.0) + 2 * k) * (a + 2 * k));
      double delta_frac;

      /* first step */
      den_term = 1.0 + coeff * den_term;
      num_term = 1.0 + coeff / num_term;

      if (fabs (den_term) < cutoff)
        den_term = nan("");

      if (fabs (num_term) < cutoff)
        num_term = nan("");

      den_term = 1.0 / den_term;

      delta_frac = den_term * num_term;
      cf *= delta_frac;

      coeff = -(a + k) * (a + b + k) * x / ((a + 2 * k) * (a + 2 * k + 1.0));

      /* second step */
      den_term = 1.0 + coeff * den_term;
      num_term = 1.0 + coeff / num_term;

      if (fabs (den_term) < cutoff)
        den_term = nan("");

      if (fabs (num_term) < cutoff)
        num_term = nan("");

      den_term = 1.0 / den_term;

      delta_frac = den_term * num_term;
      cf *= delta_frac;

      if (fabs (delta_frac - 1.0) < 2.0 * CUDA_DBL_EPSILON)
        break;

      if (cf * fabs (delta_frac - 1.0) < epsabs)
        break;

      ++iter_count;
    }

  if (iter_count >= max_iter)
    return nan("");

  return cf;
}

__global__ void betacdf_dirCF_kernel(double *x, double *y, double alpha, double beta, double ln_beta, size_t size){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    double a = alpha;
    double b = beta;
    if (idx < size){
        double my_x = x[idx];

        double ln_pre = -ln_beta + a * log(my_x) + b * log1p(-my_x);
        double prefactor = exp(ln_pre);

        double epsabs = 0.;
        double cf = cuda_beta_cont_frac(a, b, my_x, epsabs);

        y[idx] = prefactor * cf / a;
        
    }
}

__global__ void betacdf_hypergeoCF_kernel(double *x, double *y, double alpha, double beta, double ln_beta, size_t size){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    double a = alpha;
    double b = beta;
    if (idx < size){
        double my_x = x[idx];

        double ln_pre = -ln_beta + a * log(my_x) + b * log1p(-my_x);
        double prefactor = exp(ln_pre);

        double epsabs = 1. / (prefactor / b) * CUDA_DBL_EPSILON;
        double cf = cuda_beta_cont_frac(b, a, 1. - my_x, epsabs);

        double term = prefactor * cf / b;

        y[idx] = 1. - term;
    }
}

__global__ void betacdf_CF_kernel(const double *x, double *y, double alpha, double beta, double ln_beta, size_t size){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    double a = alpha;
    double b = beta;
    double limit = (alpha + 1.0) / (alpha + beta + 2.0);
    if (idx < size){
        double my_x = x[idx];

        double ln_pre = -ln_beta + a * log(my_x) + b * log1p(-my_x);
        double prefactor = exp(ln_pre);

        double epsabs = my_x < limit ? 0. : 1. / (prefactor / b) * CUDA_DBL_EPSILON; // Now every value can be one of two cases
        double cf_a = my_x < limit ? a : b;
        double cf_b = my_x < limit ? b : a;
        double cf_x = my_x < limit ? my_x : 1. - my_x;
        double cf = cuda_beta_cont_frac(cf_a, cf_b, cf_x, epsabs);

        double term = prefactor * cf / cf_a;

        double my_y = my_x < limit ? term : 1. - term;
        y[idx] =  my_y; // Now every value can be one of two cases
    }
}

// https://github.com/ampl/gsl/blob/master/specfunc/gamma_inc.c#L500
__global__ void betacdf_la_sb_kernel(double *x, double *y, double alpha, double beta, size_t size){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    double a = alpha;
    double b = beta;
    if (idx < size){
        double my_x = x[idx];
        double N = a + (b - 1.) / 2.;
        my_x = -N * log1p(-my_x);

        y[idx] = nan("");
        return;
    }
}

// https://github.com/ampl/gsl/blob/master/specfunc/gamma_inc.c#L581
__global__ void betacdf_sa_lb_kernel_f(float *x, float *y, float alpha, float beta, size_t size){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    float a = alpha;
    float b = beta;
    if (idx < size){
        float my_x = x[idx];
        float N = b + (a - 1.) / 2.;
        my_x = -N * log1pf(-my_x);

        y[idx] = nanf("");
        return;
    }
}

__global__ void betacdf_prefix_only_kernel(const double *x, double *y, double alpha, double beta, double ln_beta, size_t size){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    double a = alpha;
    double b = beta;
    if (idx < size){
        double my_x = x[idx];

        double ln_pre = -ln_beta + a * log(my_x) + b * log1p(-my_x);
        double prefactor = exp(ln_pre);

        y[idx] = prefactor;
    }
}


/* --------------- Kernel launchers --------------- */


inline void launch_betapdf_kernel(const double *d_x, double *d_y, double alpha, double beta, int size, int block_size, cudaStream_t stream=CUDA_DEFAULT_STREAM) {
    int n_blocks = size / block_size + (size % block_size == 0 ? 0 : 1);
    betapdf_kernel<<<n_blocks, block_size,0,stream>>>(d_x, d_y, alpha, beta, size);

}

inline void launch_betapdf_kernel_f(const float *d_x, float *d_y, float alpha, float beta, int size, int block_size, cudaStream_t stream=CUDA_DEFAULT_STREAM) {
    int n_blocks = size / block_size + (size % block_size == 0 ? 0 : 1);
    betapdf_kernel_f<<<n_blocks, block_size,0,stream>>>(d_x, d_y, alpha, beta, size);
}

inline void launch_betacdf_prefactor_only_kernel(const double *d_x, double *d_y, double alpha, double beta, int size, int block_size, cudaStream_t stream=CUDA_DEFAULT_STREAM) {
    int n_blocks = size / block_size + (size % block_size == 0 ? 0 : 1);
    double ln_beta = gsl_sf_lnbeta(alpha, beta);
    betacdf_prefix_only_kernel<<<n_blocks, block_size,0,stream>>>(d_x, d_y, alpha, beta, ln_beta, size);
}

inline void launch_betacdf_withCF_kernel(const double *d_x, double *d_y, double alpha, double beta, int size, int block_size, cudaStream_t stream=CUDA_DEFAULT_STREAM) {
    int n_blocks = size / block_size + (size % block_size == 0 ? 0 : 1);
    double ln_beta = gsl_sf_lnbeta(alpha, beta);
    betacdf_CF_kernel<<<n_blocks, block_size,0,stream>>>(d_x, d_y, alpha, beta, ln_beta, size);
}


/* --------------- Auxiliar encapsulation functions --------------- */

size_t get_free_GPU_memory(){
  size_t free_bytes, total_bytes;

  cudaMemGetInfo( &free_bytes, &total_bytes);

  #ifdef DEBUG
    size_t used_bytes = total_bytes - free_bytes;
    std::cerr << "GPU memory usage: " << (used_bytes>>20) << " bytes used, " << (free_bytes>>20) << " MegaBytes free, " << (total_bytes>>20) << " MegaBytes total." << std::endl;
  #endif

  return free_bytes;
}

template <typename T, typename K>
void beta_array_cuda(const T *x, T *y, const T alpha, const T beta, unsigned long size, K kernel_launcher){

  #ifdef DEBUG
    cudaEvent_t t1, t2, t3, t4;
    float elapsedMemcpyCG, elapsedKernel, elapsedMemcpyGC, elapsedTotal;
    cudaEventCreate(&t1);
    cudaEventCreate(&t2);
    cudaEventCreate(&t3);
    cudaEventCreate(&t4);

    cudaEventRecord(t1, 0);
  #endif

  // Allocate memory on the device
  T *d_x, *d_y;

  cudaMalloc(&d_x, size * sizeof(T));
  cudaMalloc(&d_y, size * sizeof(T));

  // Copy the data to the device
  cudaMemcpy(d_x, x, size * sizeof(T), cudaMemcpyHostToDevice);

  #ifdef DEBUG
    cudaEventRecord(t2, 0);
    cudaEventSynchronize(t2);
    cudaEventElapsedTime(&elapsedMemcpyCG, t1, t2);
  #endif

  // Launch the kernel
  int block_size = DEFAULT_BLOCK_SIZE;
  kernel_launcher(d_x, d_y, alpha, beta, size, block_size, CUDA_DEFAULT_STREAM);

  #ifdef DEBUG
    cudaEventRecord(t3, 0);
    cudaEventSynchronize(t3);
    cudaEventElapsedTime(&elapsedKernel, t2, t3);
  #endif

  // Copy the result back to the host
  cudaMemcpy(y, d_y, size * sizeof(T), cudaMemcpyDeviceToHost);

  // Free the memory on the device
  cudaFree(d_x);
  cudaFree(d_y);

  #ifdef DEBUG
    cudaEventRecord(t4, 0);
    cudaEventSynchronize(t4);
    cudaEventElapsedTime(&elapsedMemcpyGC, t3, t4);
    cudaEventElapsedTime(&elapsedTotal, t1, t4);

    cerr << "Full function time(events) = " << elapsedTotal / 1000 << endl;
    cerr << "\tKernel execution time = " << elapsedKernel / 1000 << endl;
    cerr << "\tMemory transfer time = " << elapsedMemcpyCG / 1000 << " + " << elapsedMemcpyGC / 1000 << endl;
  #endif

  return;
}

template <typename T, typename K>
void beta_array_cuda_streams(const T *x, T *y, const T alpha, const T beta, unsigned long size, K kernel_launcher, const unsigned int num_streams=2, const unsigned int chunks_per_stream=2){

  // Allocate memory on the device
  T *d_x[num_streams], *d_y[num_streams];

  unsigned long chunk_size = size / (chunks_per_stream * num_streams);
  unsigned long remainer_elements = size % (chunks_per_stream * num_streams);

  // Create streams
  cudaStream_t streams[num_streams];
  for (unsigned int i = 0; i < num_streams; i++){
    cudaStreamCreate(&streams[i]);

    unsigned long max_chunk_size = remainer_elements ? chunk_size + 1 : chunk_size;

    // Allocate memory on the device
    cudaMalloc(&d_x[i], (max_chunk_size) * sizeof(T));
    cudaMalloc(&d_y[i], (max_chunk_size) * sizeof(T));
  }

  // Work 
  unsigned long start_offset = 0;
  for (int block_idx = 0; block_idx < chunks_per_stream * num_streams; block_idx++){
    int stream_idx = block_idx % num_streams;
    unsigned long current_chunk_size = block_idx < remainer_elements ? chunk_size + 1 : chunk_size;

    // Copy the data to the device
    cudaMemcpyAsync(d_x[stream_idx], x + start_offset, current_chunk_size * sizeof(T), cudaMemcpyHostToDevice, streams[stream_idx]);

    kernel_launcher(d_x[stream_idx], d_y[stream_idx], alpha, beta, current_chunk_size, 256, streams[stream_idx]);

    cudaMemcpyAsync(y + start_offset, d_y[stream_idx], current_chunk_size * sizeof(T), cudaMemcpyDeviceToHost, streams[stream_idx]);

    start_offset += current_chunk_size;
  }

  // Destroy streams
  for (unsigned int i = 0; i < num_streams; i++){
    cudaStreamDestroy(streams[i]);

    // Free the memory on the device
    cudaFree(d_x[i]);
    cudaFree(d_y[i]);
  }

  return;

}

template <typename T, typename K>
void beta_array_cuda_wrapper (const T *x, T *y, const T alpha, const T beta, unsigned long size, K kernel_launcher){
  size_t free_bytes = get_free_GPU_memory();
  size_t needed_bytes = ((2 * size ) + 2)* sizeof(double); // 2 arrays of size +
                                                           // 1 extra element per stream 
                                                           // (for odd distributions to fill the remainer element, 2 streams fixed)

  #ifdef DEBUG
    std::cerr << "! Executing GPU function." << std::endl;
    std::cerr << "+-- Needed memory: " << (needed_bytes>>20) << " MegaBytes." << std::endl;
    std::cerr << "+-- Free memory: " << (free_bytes>>20) << " MegaBytes." << std::endl;
  #endif
  if (needed_bytes < free_bytes){
    // If i have enough memory, use the normal function
    #ifdef DEBUG
      std::cerr << "+- Using normal function." << std::endl;
    #endif
    beta_array_cuda<T, K>(x, y, alpha, beta, size, kernel_launcher);
  } else {
    // If i don't have enough memory, use streams
    unsigned int chunks_per_stream = (needed_bytes / free_bytes) + 1;
    #ifdef DEBUG
      std::cerr << "+- Using streams with " << chunks_per_stream << " chunks per stream." << std::endl;
    #endif
    beta_array_cuda_streams<T, K>(x, y, alpha, beta, size, kernel_launcher, 2, chunks_per_stream);
  }

  return;
}

/* --------------- Export fuctions --------------- */


// CUDA kernel launch to compute the beta distribution
void betapdf_cuda(const double *x, double *y, const double alpha, const double beta, unsigned long size, Memory_Type memory_type){

  if (memory_type == Memory_Type::HOST){
    beta_array_cuda_wrapper<double, KernelLauncher>(x, y, alpha, beta, size, launch_betapdf_kernel);
  } 

  // If the memory type is DEVICE, we can use KernelLauncher function directly
  if (memory_type == Memory_Type::DEVICE){
    launch_betapdf_kernel(x, y, alpha, beta, size, DEFAULT_BLOCK_SIZE);
  }
  

  return;
}

// CUDA kernel launch to compute the beta distribution
void betapdf_cuda(const float *x, float *y, const float alpha, const float beta, unsigned long size, Memory_Type memory_type){

  if (memory_type == Memory_Type::HOST){
    beta_array_cuda_wrapper<float, KernelLauncherFloat>(x, y, alpha, beta, size, launch_betapdf_kernel_f);
  }

  // If the memory type is DEVICE, we can use KernelLauncher function directly
  if (memory_type == Memory_Type::DEVICE){
    launch_betapdf_kernel_f(x, y, alpha, beta, size, DEFAULT_BLOCK_SIZE);
  }

  return;
}

// CUDA kernel launch to compute the beta distribution
void betacdf_cuda(const double *x, double *y, const double alpha, const double beta, unsigned long size, Memory_Type memory_type){

  if (memory_type == Memory_Type::HOST){
    beta_array_cuda_wrapper<double, KernelLauncher>(x, y, alpha, beta, size, launch_betacdf_withCF_kernel);
  }

  // If the memory type is DEVICE, we can use KernelLauncher function directly
  if (memory_type == Memory_Type::DEVICE){
    launch_betacdf_withCF_kernel(x, y, alpha, beta, size, DEFAULT_BLOCK_SIZE);
  }

  return;
}

void betacdf_cuda_GPU_CPU(const double *x, double *y, const double alpha, const double beta, unsigned long size, Memory_Type memory_type){

  if (memory_type == Memory_Type::HOST) {
    beta_array_cuda_wrapper<double, KernelLauncher>(x, y, alpha, beta, size, launch_betacdf_prefactor_only_kernel);
  }

  // If the memory type is DEVICE, we can use KernelLauncher function directly
  if (memory_type == Memory_Type::DEVICE) {
    throw std::runtime_error("This function is not implemented for DEVICE memory type.");
  }

#pragma omp parallel for schedule(static, 64)
  for (unsigned long i = 0; i < size; i++) {
    if (x[i] < (alpha + 1.0) / (alpha + beta + 2.0)) {
      /* Apply continued fraction directly. */
      double epsabs = 0.;

      double cf = cuda_beta_cont_frac(alpha, beta, x[i], epsabs);

      y[i] = y[i] * cf / alpha;
    } else {
      /* Apply continued fraction after hypergeometric transformation. */
      double epsabs =
          fabs(1. / (y[i] / beta)) * CUDA_DBL_EPSILON;
      double cf = cuda_beta_cont_frac(beta, alpha, 1.0 - x[i], epsabs);
      double term = y[i] * cf / beta;

      y[i] = 1 - term;
    }
  }

  return;
}
