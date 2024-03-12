#include <vector>

#define CUDA_DBL_MIN        2.2250738585072014e-308
#define CUDA_DBL_EPSILON    2.2204460492503131e-16

enum class GPU_Type {
    HALF = 16,
    FLOAT = sizeof(float),
    DOUBLE = sizeof(double)
};

std::vector<double> betapdf_cuda(std::vector<double> x, double alpha, double beta, GPU_Type precision=GPU_Type::DOUBLE);

#ifdef DEBUG
std::vector<double> betapdf_cuda_times(std::vector<double> x, double alpha, double beta, GPU_Type precision=GPU_Type::DOUBLE);
#endif

std::vector<double> betacdf_cuda(std::vector<double> x, double alpha, double beta);