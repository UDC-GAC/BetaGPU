#include "../src_cuda/BetaDistCuda.hpp"

#include <vector>
#include <iostream>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

std::vector<double> beta_pdf_cuda(const std::vector<double> &x, const double alpha, const double beta) {
    std::vector<double> y(x.size());
    betapdf_cuda(x.data(), y.data(), alpha, beta, x.size());
    return y;
}

std::vector<float> beta_pdf_cuda_f(const std::vector<float> &x, const float alpha, const float beta) {
    std::vector<float> y(x.size());
    betapdf_cuda(x.data(), y.data(), alpha, beta, x.size());
    return y;
}

std::vector<double> beta_cdf_cuda(const std::vector<double> &x, const double alpha, const double beta) {
    std::vector<double> y(x.size());
    betacdf_cuda(x.data(), y.data(), alpha, beta, x.size());
    return y;
}

std::vector<double> beta_cdf_cuda_GPU_CPU(const std::vector<double> &x, const double alpha, const double beta) {
    std::vector<double> y(x.size());
    betacdf_cuda_GPU_CPU(x.data(), y.data(), alpha, beta, x.size());
    return y;
}



PYBIND11_MODULE(BetaDistCuda, m) {
    m.doc() = "Beta distribution Cuda module";

    m.def("beta_pdf_cuda", &beta_pdf_cuda, "Beta PDF Cuda");
    m.def("beta_pdf_cuda_f", &beta_pdf_cuda_f, "Beta PDF Cuda");
    m.def("beta_cdf_cuda", &beta_cdf_cuda, "Beta CDF Cuda");
    m.def("beta_cdf_cuda_GPU_CPU", &beta_cdf_cuda_GPU_CPU, "Beta CDF Cuda");
}
