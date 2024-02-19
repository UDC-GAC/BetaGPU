#include "../src_ref/BetaDistGsl.hpp"

#include <chrono>
#include <cstdlib>
#include <ctime>
#include <iostream>
#include <vector>

#include <omp.h>

using std::vector;
using std::cerr;
using std::endl;

using profile_clock_t = std::chrono::high_resolution_clock;
using profile_duration_t = std::chrono::duration<double>;





struct CommandLineOptions {

  enum class ExecutionMode {
    SEQ,
    OMP,
    CUDA
  };

  enum class FunctionName {
    BETAPDF,
    BETACDF
  };

  int num_elements;
  int num_iterations;
  ExecutionMode exec_mode;
  FunctionName function_name;
};

__global__ void betapdf_kernel_self(double *x, double *y, double alpha, double beta, size_t size){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size){
        y[idx] = pow(x[idx], alpha - 1) * pow(1 - x[idx], beta - 1) * exp(lgamma(alpha + beta) - lgamma(alpha) - lgamma(beta));
    }
}

// CUDA kernel launch to compute the beta distribution
std::vector<double> betapdf_cuda_self(std::vector<double> x, double alpha, double beta){
    // Allocate memory on the device
    double *d_x, *d_y;
    cudaMalloc(&d_x, x.size() * sizeof(double));
    cudaMalloc(&d_y, x.size() * sizeof(double));

    // Copy the data to the device
    cudaMemcpy(d_x, x.data(), x.size() * sizeof(double), cudaMemcpyHostToDevice);

    // Launch the kernel
    int block_size = 256;
    int n_blocks = x.size() / block_size + (x.size() % block_size == 0 ? 0 : 1);
    betapdf_kernel_self<<<n_blocks, block_size>>>(d_x, d_y, alpha, beta, x.size());

    // Copy the result back to the host
    std::vector<double> y(x.size());
    cudaMemcpy(y.data(), d_y, x.size() * sizeof(double), cudaMemcpyDeviceToHost);

    // Free the memory on the device
    cudaFree(d_x);
    cudaFree(d_y);

    return y;
}

static std::string get_help_message(std::string prog_name) {
  return "Usage: " + prog_name + R"( [num_elements] [num_iterations] [exec_mode] [function_name]
  num_elements: Number of elements in the input vector
  num_iterations: Number of iterations to run the test
  exec_mode: Execution mode (seq, omp, cuda)
  function_name: Name of the function to test (betapdf, betacdf)
)";
}

int parse_positive_int(char* str) {
  int value = std::stoi(str);
  if (value <= 0) {
    throw std::invalid_argument("Value must be a positive integer");
  }
  return value;
}

CommandLineOptions::ExecutionMode parse_exec_mode(char* str) {
  std::string mode(str);
  if (mode == "seq")
    return CommandLineOptions::ExecutionMode::SEQ;
  if (mode == "omp")
    return CommandLineOptions::ExecutionMode::OMP;
  if (mode == "cuda")
    return CommandLineOptions::ExecutionMode::CUDA;
  
  throw std::invalid_argument("Invalid execution mode" + mode + "\n\t Valid execution modes are: seq, omp, cuda");
}

CommandLineOptions::FunctionName parse_function_name(char* str) {
  std::string mode(str);
  if (mode == "betapdf")
    return CommandLineOptions::FunctionName::BETAPDF;
  if (mode == "betacdf")
    return CommandLineOptions::FunctionName::BETACDF;
  
  throw std::invalid_argument("Invalid function name" + mode + "\n\t Valid function names are: betapdf, betacdf");
}
  
CommandLineOptions parse_command_line(int argc, char *argv[]) {
  CommandLineOptions options;
  if (argc != 5) {
    std::cerr << get_help_message(argv[0]) << std::endl;
    exit(EXIT_SUCCESS);
  }

  options.num_elements = parse_positive_int(argv[1]);
  options.num_iterations = parse_positive_int(argv[2]);
  options.exec_mode = parse_exec_mode(argv[3]);
  options.function_name = parse_function_name(argv[4]);

  return options;
}

std::string mode_to_text(CommandLineOptions::ExecutionMode mode) {
  switch (mode) {
  case CommandLineOptions::ExecutionMode::SEQ:
    return "Sequential";
  case CommandLineOptions::ExecutionMode::OMP:
    return "OpenMP";
  case CommandLineOptions::ExecutionMode::CUDA:
    return "CUDA";
  }
  return "Unknown";
}

std::string function_to_test(CommandLineOptions::FunctionName function) {
  switch (function) {
  case CommandLineOptions::FunctionName::BETAPDF:
    return "Beta PDF";
  case CommandLineOptions::FunctionName::BETACDF:
    return "Beta CDF";
  }
  return "Unknown";
}

void print_execution_parameters(const CommandLineOptions& options) {
  cerr << "+------------------------------------+" << endl;
  cerr << "|        Execution Parameters        |" << endl;
  cerr << "+------------------------------------+" << endl;
  cerr << "\tNumber of elements: " << options.num_elements << endl;
  cerr << "\tNumber of iterations: " << options.num_iterations << endl;
  cerr << "\tExecution mode: " << mode_to_text(options.exec_mode) << endl;
  cerr << "\tFunction name: " << function_to_test(options.function_name) << endl;
}

void execute_test(const CommandLineOptions& options, vector<double>& x, double alpha, double beta){
  vector<double> y(x.size());
  switch (options.exec_mode) {
  case CommandLineOptions::ExecutionMode::SEQ:
    switch (options.function_name) {
    case CommandLineOptions::FunctionName::BETAPDF:
      for (size_t i = 0; i < x.size(); i++) {
        y.at(i) = betapdf(x.at(i), alpha, beta);
      }
      break;
    case CommandLineOptions::FunctionName::BETACDF:
      for (size_t i = 0; i < x.size(); i++) {
        y.at(i) = betacdf(x.at(i), alpha, beta);
      }
      break;
    }
    break;
  case CommandLineOptions::ExecutionMode::OMP:
    switch (options.function_name) {
    case CommandLineOptions::FunctionName::BETAPDF:
      #pragma omp parallel for schedule(static, 64)
      for (size_t i = 0; i < x.size(); i++) {
        y.at(i) = betacdf(x.at(i), alpha, beta);
      }
      break;
    case CommandLineOptions::FunctionName::BETACDF:
      #pragma omp parallel for schedule(static, 64)
      for (size_t i = 0; i < x.size(); i++) {
        y.at(i) = betacdf(x.at(i), alpha, beta);
      }
      break;
    }
    break;
  case CommandLineOptions::ExecutionMode::CUDA:
    switch (options.function_name) {
    case CommandLineOptions::FunctionName::BETAPDF:
      cerr << "hello" << endl;
      y = betapdf_cuda_self(x, alpha, beta);
      break;
    case CommandLineOptions::FunctionName::BETACDF:
      y = betapdf_cuda_self(x, alpha, beta);
      break;
    }
    break;
  }
}

int 
main (int argc, char *argv[]) {

  CommandLineOptions options = parse_command_line(argc, argv);
  print_execution_parameters(options);
  
  vector<double> x(options.num_elements);
  

  for (int i = 0; i < options.num_elements; i++) {
    x[i] = rand() / (double)RAND_MAX;
  }

  auto full_start = profile_clock_t::now();
  for (int i = 1; i <= options.num_iterations; i++) {
    double alpha = 0.1 * i;
    double beta = 0.1 * i;
    
    auto start = profile_clock_t::now();
    execute_test(options, x, alpha, beta);
    auto end = profile_clock_t::now();

    cerr << "Itr[" << i << "]\t\tTime = \t\t" << profile_duration_t(end - start).count() << endl;
  }
  auto full_end = profile_clock_t::now();
  cerr << "Total time = " << profile_duration_t(full_end - full_start).count() << endl;
  
}
