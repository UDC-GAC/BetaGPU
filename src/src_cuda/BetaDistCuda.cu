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

// Define a type for the function that launches the kernel
typedef void (*KernelLauncher)(double*, double*, double, double, int, int);
typedef void (*KernelLauncherFloat)(float*, float*, float, float, int, int);


/* --------------- Beta PDF Kernels --------------- */


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


/* --------------- Beta CDF Kernels --------------- */


// TODO: Implement the beta distribution CDF using the continued fraction
__device__ double cuda_beta_cont_frac(double alpha, double beta, double x, double epsabs){
    const unsigned int max_iter = 512;
    const double cutoff = 2. * CUDA_DBL_MIN;
    double cf;
    double delta_frac;

    double num_term = 1.;
    double denom_term = 1. - (alpha + beta) * x / (alpha + 1.);

    if (fabs(denom_term) < cutoff)
        denom_term = nan("");

    denom_term = 1. / denom_term;
    cf = denom_term;

    for (unsigned int iter = 0; iter < max_iter; iter++){
        
        const unsigned int k = iter + 1;
        double coeff = k * (beta - k) * x / (((alpha - 1.) + 2 * k) * (alpha + 2 * k));
        

        /* first step */
        denom_term = 1. + coeff * denom_term;
        num_term = 1. + coeff / num_term;

        if (fabs(denom_term) < cutoff)
            denom_term = nan("");

        if (fabs(num_term) < cutoff)
            num_term = nan("");

        denom_term = 1. / denom_term;

        delta_frac = denom_term * num_term;
        cf *= delta_frac;

        coeff = -(alpha + k) * (alpha + beta + k) * x / ((alpha + 2 * k) * (alpha + 2 * k + 1.));
    
        /* second step */
        denom_term = 1. + coeff * denom_term;
        num_term = 1. + coeff / num_term;

        if (fabs(denom_term) < cutoff)
            denom_term = nan("");

        if (fabs(num_term) < cutoff)
            num_term = nan("");

        denom_term = 1. / denom_term;

        delta_frac = denom_term * num_term;
        cf *= delta_frac;

        /* last iteration checks */
        //if (fabs(delta_frac - 1.) < 2. * CUDA_DBL_EPSILON)
        //    break;

        //if (cf * fabs(delta_frac - 1.) < epsabs)
        //    break;
    
    }

    // These checks are originally done within the loop
    // If this logic within the loop is modified, this should be modified as well
    //if (fabs(delta_frac - 1.) < 2. * CUDA_DBL_EPSILON || cf * fabs(delta_frac - 1.) < epsabs)
    //    return nan("");
        
    return cf;

}

__global__ void betacdf_dirCF_kernel(double *x, double *y, double alpha, double beta, size_t size){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    double a = alpha;
    double b = beta;
    if (idx < size){
        double my_x = x[idx];

        double ln_beta = lgamma(a + b) - lgamma(a) - lgamma(b);
        double ln_pre = -ln_beta + a * log(my_x) + b * log1p(-my_x);
        double prefactor = exp(ln_pre);

        double epsabs = 0.;
        double cf = cuda_beta_cont_frac(a, b, my_x, epsabs);

        y[idx] = prefactor * cf / a;
        
    }
}

__global__ void betacdf_hypergeoCF_kernel(double *x, double *y, double alpha, double beta, size_t size){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    double a = alpha;
    double b = beta;
    if (idx < size){
        double my_x = x[idx];

        double ln_beta = lgamma(a + b) - lgamma(a) - lgamma(b);
        double ln_pre = -ln_beta + a * log(my_x) + b * log1p(-my_x);
        double prefactor = exp(ln_pre);

        double epsabs = 1. / (prefactor / b) * CUDA_DBL_EPSILON;
        double cf = cuda_beta_cont_frac(b, a, 1. - my_x, epsabs);

        double term = prefactor * cf / b;

        y[idx] = 1. - term;
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

__global__ void betacdf_prefix_only_kernel(double *x, double *y, double alpha, double beta, size_t size){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    double a = alpha;
    double b = beta;
    if (idx < size){
        double my_x = x[idx];

        double ln_beta = lgamma(a + b) - lgamma(a) - lgamma(b);
        double ln_pre = -ln_beta + a * log(my_x) + b * log1p(-my_x);
        double prefactor = exp(ln_pre);

        y[idx] = prefactor;
    }
}


/* --------------- Kernel launchers --------------- */


inline void launch_betapdf_kernel(double *d_x, double *d_y, double alpha, double beta, int size, int block_size) {
    int n_blocks = size / block_size + (size % block_size == 0 ? 0 : 1);
    betapdf_kernel<<<n_blocks, block_size>>>(d_x, d_y, alpha, beta, size);
}

inline void launch_betapdf_kernel_f(float *d_x, float *d_y, float alpha, float beta, int size, int block_size) {
    int n_blocks = size / block_size + (size % block_size == 0 ? 0 : 1);
    betapdf_kernel_f<<<n_blocks, block_size>>>(d_x, d_y, alpha, beta, size);
}

inline void launch_betacdf_prefactor_only_kernel(double *d_x, double *d_y, double alpha, double beta, int size, int block_size) {
    int n_blocks = size / block_size + (size % block_size == 0 ? 0 : 1);
    betacdf_prefix_only_kernel<<<n_blocks, block_size>>>(d_x, d_y, alpha, beta, size);
}


/* --------------- Auxiliar encapsulation functions --------------- */


void beta_array_cuda(const double *x, double *y, const double alpha, const double beta, unsigned long size, KernelLauncher kernel_launcher){
    
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
    double *d_x, *d_y;

    cudaMalloc(&d_x, size * sizeof(double));
    cudaMalloc(&d_y, size * sizeof(double));

    // Copy the data to the device
    cudaMemcpy(d_x, x, size * sizeof(double), cudaMemcpyHostToDevice);

    #ifdef DEBUG
    cudaEventRecord(t2, 0);
    cudaEventSynchronize(t2);
    cudaEventElapsedTime(&elapsedMemcpyCG, t1, t2);
    #endif

    // Launch the kernel
    int block_size = 256;
    kernel_launcher(d_x, d_y, alpha, beta, size, block_size);

    #ifdef DEBUG
    cudaEventRecord(t3, 0);
    cudaEventSynchronize(t3);
    cudaEventElapsedTime(&elapsedKernel, t2, t3);
    #endif

    // Copy the result back to the host
    cudaMemcpy(y, d_y, size * sizeof(double), cudaMemcpyDeviceToHost);

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

void beta_array_cuda_float(const float *x, float *y, const float alpha, const float beta, unsigned long size, KernelLauncherFloat kernel_launcher){
    
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
    float *d_x, *d_y;

    cudaMalloc(&d_x, size * sizeof(float));
    cudaMalloc(&d_y, size * sizeof(float));

    // Copy the data to the device
    cudaMemcpy(d_x, x, size * sizeof(float), cudaMemcpyHostToDevice);

    #ifdef DEBUG
    cudaEventRecord(t2, 0);
    cudaEventSynchronize(t2);
    cudaEventElapsedTime(&elapsedMemcpyCG, t1, t2);
    #endif

    // Launch the kernel
    int block_size = 256;
    kernel_launcher(d_x, d_y, alpha, beta, size, block_size);

    #ifdef DEBUG
    cudaEventRecord(t3, 0);
    cudaEventSynchronize(t3);
    cudaEventElapsedTime(&elapsedKernel, t2, t3);
    #endif

    // Copy the result back to the host
    cudaMemcpy(y, d_y, size * sizeof(float), cudaMemcpyDeviceToHost);

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


/* ----- Auxiliar tmp functions ----- */


// Look https://github.com/ampl/gsl/blob/master/cdf/beta_inc.c#L26
double tmp_beta_cont_frac (const double a, const double b, const double x,
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


/* --------------- Export fuctions --------------- */


// CUDA kernel launch to compute the beta distribution
void betapdf_cuda(const double *x, double *y, const double alpha, const double beta, unsigned long size){
    
    beta_array_cuda(x, y, alpha, beta, size, launch_betapdf_kernel);

    return;
}

// CUDA kernel launch to compute the beta distribution
void betapdf_cuda(const float *x, float *y, const float alpha, const float beta, unsigned long size){

    beta_array_cuda_float(x, y, alpha, beta, size, launch_betapdf_kernel_f);

    return;
}

void betacdf_cuda(const double *x, double *y, const double alpha, const double beta, unsigned long size){
    
    beta_array_cuda(x, y, alpha, beta, size, launch_betacdf_prefactor_only_kernel);

    for (unsigned long i = 0; i < size; i++)
    {
        if (x[i] < (alpha + 1.0) / (alpha + beta + 2.0)) {
            /* Apply continued fraction directly. */
            double epsabs = 0.;

            double cf = tmp_beta_cont_frac(alpha, beta, x[i], epsabs);

            y[i] = y[i] * cf / alpha;
        } else {
            /* Apply continued fraction after hypergeometric transformation. */
            double epsabs =
                fabs(1. / (y[i] / beta)) * CUDA_DBL_EPSILON;
            double cf = tmp_beta_cont_frac(beta, alpha, 1.0 - x[i], epsabs);
            double term = y[i] * cf / beta;

            y[i] = 1 - term;
        }
    }

    return;
}
