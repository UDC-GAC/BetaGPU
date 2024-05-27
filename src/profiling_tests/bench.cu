#include "../src_ref/BetaDistGsl.hpp"
#include "../src_cuda/BetaDistCuda.hpp"

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <ctime>
#include <iostream>
#include <thread>
#include <vector>

#include <omp.h>

using std::vector;
using std::cerr;
using std::endl;

using profile_clock_t = std::chrono::high_resolution_clock;
using profile_duration_t = std::chrono::duration<double>;


const double DEFAULT_ALPHA = 9.34;
const double DEFAULT_BETA = 11.34;


struct CommandLineOptions {

  enum class ExecutionMode {
    SEQ,
    OMP,
    CUDA,
    CUDA_F,
    CUDA_OMP
  };

  enum class FunctionName {
    BETAPDF,
    BETACDF
  };

  int num_elements;
  int num_iterations;
  ExecutionMode exec_mode;
  FunctionName function_name;
  bool using_pinned_memory = false;
  bool using_sorted_data = false;
  double alpha = DEFAULT_ALPHA;
  double beta = DEFAULT_BETA;
};

static std::string get_help_message(std::string prog_name) {
  return "Usage: " + prog_name + R"( [num_elements] [num_iterations] [exec_mode] [function_name]
  num_elements: Number of elements in the input vector
  num_iterations: Number of iterations to run the test
  exec_mode: Execution mode (seq, omp, cuda, cuda_f)
  function_name: Name of the function to test (betapdf, betacdf)
  -p: Use pinned memory (only for CUDA modes)
  -s: Use sorted data
  -a [value]: Alpha parameter for the Beta distribution (default: 9.34)
  -b [value]: Beta parameter for the Beta distribution (default: 11.34)
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
  if (mode == "cuda_f")
    return CommandLineOptions::ExecutionMode::CUDA_F;
  if (mode == "cuda_omp")
    return CommandLineOptions::ExecutionMode::CUDA_OMP;
  
  throw std::invalid_argument("Invalid execution mode" + mode + "\n\t Valid execution modes are: seq, omp, cuda, cuda_f and cuda_omp");
}

CommandLineOptions::FunctionName parse_function_name(char* str) {
  std::string mode(str);
  if (mode == "betapdf")
    return CommandLineOptions::FunctionName::BETAPDF;
  if (mode == "betacdf")
    return CommandLineOptions::FunctionName::BETACDF;
  
  throw std::invalid_argument("Invalid function name" + mode + "\n\t Valid function names are: betapdf, betacdf");
}

void parse_leftover_args(int argc, char *argv[], CommandLineOptions& options) {
  for (int i = 0; i < argc; i++) {
    std::string arg(argv[i]);
    bool used = false;

    if (arg == "-p") {
      if (options.exec_mode != CommandLineOptions::ExecutionMode::CUDA && options.exec_mode != CommandLineOptions::ExecutionMode::CUDA_F && options.exec_mode != CommandLineOptions::ExecutionMode::CUDA_OMP) {
        exit(EXIT_FAILURE);
      }
      options.using_pinned_memory = true;
      used = true;
    }

    if (arg == "-s") {
      options.using_sorted_data = true;
      used = true;
    }

    if (arg == "-a") {
      if (i+1 >= argc) {
        throw std::invalid_argument("Missing argument for -a");
      }
      options.alpha = std::stod(argv[++i]);
      used = true;
    }

    if (arg == "-b") {
      if (i+1 >= argc) {
        throw std::invalid_argument("Missing argument for -b");
      }
      options.beta = std::stod(argv[++i]);
      used = true;
    }

    // ...

    if (!used) {
      throw std::invalid_argument("Invalid argument: " + arg);
    }
  }
}
  
CommandLineOptions parse_command_line(int argc, char *argv[]) {
  CommandLineOptions options;
  if (argc < 5) {
    std::cerr << get_help_message(argv[0]) << std::endl;
    exit(EXIT_SUCCESS);
  }

  options.num_elements = parse_positive_int(argv[1]);
  options.num_iterations = parse_positive_int(argv[2]);
  options.exec_mode = parse_exec_mode(argv[3]);
  options.function_name = parse_function_name(argv[4]);
  options.using_pinned_memory = false;
  options.using_sorted_data = false;
  if (argc > 5) {
    parse_leftover_args(argc-5, argv+5, options);
  }

  return options;
}

std::string mode_to_text(CommandLineOptions::ExecutionMode mode) {
  switch (mode) {
  case CommandLineOptions::ExecutionMode::SEQ:
    return "001.-Sequential";
  case CommandLineOptions::ExecutionMode::OMP:
    return "002.-OpenMP";
  case CommandLineOptions::ExecutionMode::CUDA:
    return "003.-CUDA";
  case CommandLineOptions::ExecutionMode::CUDA_F:
    return "004.-CUDA\\_F";
  case CommandLineOptions::ExecutionMode::CUDA_OMP:
    return "005.-CUDA\\_OMP";
  }
  return "000.-Unknown";
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
  std::string pinned_text = options.using_pinned_memory ? "^{pinned}" : "";
  std::string sorted_text = options.using_sorted_data ? "[sorted]" : "";
  cerr << "+------------------------------------+" << endl;
  cerr << "|        Execution Parameters        |" << endl;
  cerr << "+------------------------------------+" << endl;
  cerr << "\tNumber of elements: " << options.num_elements << endl;
  cerr << "\tNumber of iterations: " << options.num_iterations << endl;
  cerr << "\tExecution mode: " << mode_to_text(options.exec_mode) << pinned_text << endl;
  cerr << "\tFunction name: " << function_to_test(options.function_name) << endl;
  cerr << "\tAlpha: " << options.alpha << endl;
  cerr << "\tBeta: " << options.beta << endl;
  if (options.exec_mode == CommandLineOptions::ExecutionMode::OMP) {
    cerr << "\tNumber of threads: " << omp_get_max_threads() << endl;
  }
}

void execute_test(const CommandLineOptions& options, double *x, float *x_f, double *y, float *y_f, double alpha, double beta, size_t v_size){
  switch (options.exec_mode) {
  case CommandLineOptions::ExecutionMode::SEQ:
    switch (options.function_name) {
    case CommandLineOptions::FunctionName::BETAPDF:
      for (size_t i = 0; i < v_size; i++) {
        y[i] = betapdf(x[i], alpha, beta);
      }
      break;
    case CommandLineOptions::FunctionName::BETACDF:
      for (size_t i = 0; i < v_size; i++) {
        y[i] = betacdf(x[i], alpha, beta);
      }
      break;
    }
    break;
  case CommandLineOptions::ExecutionMode::OMP:
    switch (options.function_name) {
    case CommandLineOptions::FunctionName::BETAPDF:
      #pragma omp parallel for schedule(static, 64)
      for (size_t i = 0; i < v_size; i++) {
        y[i] = betapdf(x[i], alpha, beta);
      }
      break;
    case CommandLineOptions::FunctionName::BETACDF:
      #pragma omp parallel for schedule(static, 64)
      for (size_t i = 0; i < v_size; i++) {
        y[i] = betacdf(x[i], alpha, beta);
      }
      break;
    }
    break;
  case CommandLineOptions::ExecutionMode::CUDA:
    switch (options.function_name) {
    case CommandLineOptions::FunctionName::BETAPDF:
      betapdf_cuda(x, y, alpha, beta, v_size);
      break;
    case CommandLineOptions::FunctionName::BETACDF:
      betacdf_cuda(x, y, alpha, beta, v_size);
      break;
    }
    break;
  case CommandLineOptions::ExecutionMode::CUDA_F:
    switch (options.function_name) {
    case CommandLineOptions::FunctionName::BETAPDF:
      betapdf_cuda(x_f, y_f, static_cast<float>(alpha), static_cast<float>(beta), v_size);
      break;
    case CommandLineOptions::FunctionName::BETACDF:
      cerr << "CUDA FLOAT CDF not implemented" << endl;
      std::this_thread::sleep_for(std::chrono::milliseconds(1000));
      break;
    }
    break;
  case CommandLineOptions::ExecutionMode::CUDA_OMP:
    switch (options.function_name) {
    case CommandLineOptions::FunctionName::BETAPDF:
      cerr << "CUDA OMP PDF not implemented" << endl;
      std::this_thread::sleep_for(std::chrono::milliseconds(1000));
      break;
    case CommandLineOptions::FunctionName::BETACDF:
      betacdf_cuda_GPU_CPU(x, y, alpha, beta, v_size);
      break;
    }
    break;
  }
}

template <typename T>
void generate_data(T *data, size_t size, bool sorted = false) {
  if (sorted) {
    T step = 1.0 / (size+1.0);
    for (size_t i = 0; i < size; i++) {
      data[i] = (i+1) * step;
    }
    return;
  } else {
    for (size_t i = 0; i < size; i++) {
      data[i] = rand() / (T)RAND_MAX;
    }
  }
}

int 
main (int argc, char *argv[]) {

  srand(4135);

  CommandLineOptions options = parse_command_line(argc, argv);
  print_execution_parameters(options);
  double alpha = options.alpha;
  double beta = options.beta;

  if (!options.using_pinned_memory) {
  
    vector<double> x(options.num_elements), y(options.num_elements);
    vector<float> x_f, y_f;
    if (options.exec_mode == CommandLineOptions::ExecutionMode::CUDA_F) {
      x_f.resize(options.num_elements);
      y_f.resize(options.num_elements);
    }

    generate_data<double>(x.data(), options.num_elements, options.using_sorted_data);

    if (options.exec_mode == CommandLineOptions::ExecutionMode::CUDA_F)
      std::transform(x.begin(), x.end(), x_f.begin(), [](double x) { return (float)x; });

    auto full_start = profile_clock_t::now();
    for (int i = 1; i <= options.num_iterations; i++) {
      
      auto start = profile_clock_t::now();
      execute_test(options, x.data(), x_f.data(), y.data(), y_f.data(), alpha, beta, options.num_elements);
      auto end = profile_clock_t::now();

      cerr << "Itr[" << i << "]\t\tTime = \t\t" << profile_duration_t(end - start).count() << endl;
    }
    auto full_end = profile_clock_t::now();
    cerr << "Total time = " << profile_duration_t(full_end - full_start).count() << endl;

  } else {
     if (options.exec_mode == CommandLineOptions::ExecutionMode::CUDA || options.exec_mode == CommandLineOptions::ExecutionMode::CUDA_OMP) {
      double* x; 
      double* y;
      cudaMallocHost(&x, options.num_elements * sizeof(double));
      cudaMallocHost(&y, options.num_elements * sizeof(double));
      float *x_f, *y_f;

      generate_data<double>(x, options.num_elements, options.using_sorted_data);

      auto full_start = profile_clock_t::now();
      for (int i = 1; i <= options.num_iterations; i++) {
        
        auto start = profile_clock_t::now();
        execute_test(options, x, x_f, y, y_f, alpha, beta, options.num_elements);
        auto end = profile_clock_t::now();

        cerr << "Itr[" << i << "]\t\tTime = \t\t" << profile_duration_t(end - start).count() << endl;
      }
      auto full_end = profile_clock_t::now();
      cerr << "Total time = " << profile_duration_t(full_end - full_start).count() << endl;

      cudaFreeHost(x);
      cudaFreeHost(y);
     }
     if (options.exec_mode == CommandLineOptions::ExecutionMode::CUDA_F){
      double *x_d, *y_d;
      float* x; 
      float* y;
      cudaMallocHost(&x, options.num_elements * sizeof(float));
      cudaMallocHost(&y, options.num_elements * sizeof(float));

      generate_data<float>(x, options.num_elements, options.using_sorted_data);

      auto full_start = profile_clock_t::now();
      for (int i = 1; i <= options.num_iterations; i++) {
        
        auto start = profile_clock_t::now();
        execute_test(options, x_d, x, y_d, y, alpha, beta, options.num_elements);
        auto end = profile_clock_t::now();

        cerr << "Itr[" << i << "]\t\tTime = \t\t" << profile_duration_t(end - start).count() << endl;
      }
      auto full_end = profile_clock_t::now();
      cerr << "Total time = " << profile_duration_t(full_end - full_start).count() << endl;

      cudaFreeHost(x);
      cudaFreeHost(y);
     }
  }
  
}
