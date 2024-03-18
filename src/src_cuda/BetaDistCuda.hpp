#include <vector>

#define CUDA_DBL_MIN        2.2250738585072014e-308
#define CUDA_DBL_EPSILON    2.2204460492503131e-16

enum class GPU_Type {
    HALF = 16,
    FLOAT = sizeof(float),
    DOUBLE = sizeof(double)
};

void betapdf_cuda(const double *x, double *y, const double alpha, const double beta, unsigned long size);

double* betapdf_cuda_pinned(const std::vector<double> &x, const double alpha, const double beta);

std::vector<float> betapdf_cuda(const std::vector<float> &x, const float alpha, const float beta);

std::vector<double> betacdf_cuda(std::vector<double> &x, double alpha, double beta);