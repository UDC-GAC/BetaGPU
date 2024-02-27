#include <vector>

enum class GPU_Type {
    HALF = 16,
    FLOAT = sizeof(float),
    DOUBLE = sizeof(double)
};

std::vector<double> betapdf_cuda(std::vector<double> x, double alpha, double beta, GPU_Type precision=GPU_Type::DOUBLE);

std::vector<double> betapdf_cuda_times(std::vector<double> x, double alpha, double beta, GPU_Type precision=GPU_Type::DOUBLE);

std::vector<double> betacdf_cuda(std::vector<double> x, double alpha, double beta);