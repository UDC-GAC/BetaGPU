#include <cmath>
#include <iostream>
#include <vector>

#define CUDA_DBL_MIN        2.2250738585072014e-308
#define CUDA_DBL_EPSILON    2.2204460492503131e-16

using std::vector;

struct CommandLineOptions {

  double alpha = 0.2;
  double beta = 0.7;
  double num_elements = 1000;


};

void print_execution_parameters(const CommandLineOptions& options) {
  std::cerr << "+------------------------------------+" << std::endl;
  std::cerr << "|        Execution Parameters        |" << std::endl;
  std::cerr << "+------------------------------------+" << std::endl;
  std::cerr << "\tNumber of elements: " << options.num_elements << std::endl;
  std::cerr << "\tAlpha: " << options.alpha << std::endl;
  std::cerr << "\tBeta: " << options.beta << std::endl;
}

CommandLineOptions parse_command_line(int argc, char *argv[]) {
  CommandLineOptions options;
  for (int i = 1; i < argc; i++) {
    if (std::string(argv[i]) == "--alpha" || std::string(argv[i]) == "-a"){
      options.alpha = std::stod(argv[++i]);
    } else if (std::string(argv[i]) == "--beta" || std::string(argv[i]) == "-b"){
      options.beta = std::stod(argv[++i]);
    } else if (std::string(argv[i]) == "--num_elements" || std::string(argv[i]) == "-n"){
      options.num_elements = std::stod(argv[++i]);
    }
  }
  return options;
}


// Look https://github.com/ampl/gsl/blob/master/cdf/beta_inc.c#L26
double tmp_beta_cont_frac_measure (const double a, const double b, const double x,
                const double epsabs, size_t &iters) {
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
    iters = iter_count;

  if (iter_count >= max_iter)
    return nan("");

  return cf;
}

int
main (int argc, char *argv[])
{

  CommandLineOptions options = parse_command_line(argc, argv);
  print_execution_parameters(options);
  double epsabs = 1e-15;
  double alpha = options.alpha;
  double beta = options.beta;
  vector<double> x(options.num_elements);
  double step = 1.0 / (options.num_elements+1);
  for (int i = 0; i < options.num_elements; i++) {
    x[i] = (i+1) * step;
  }
  vector<double> y(x.size(), 1.0);
  vector<size_t> iter_count(x.size(), 0);

  for (double i = 0; i < x.size(); ++i){

    // Check X is in the range [0, 1]
    if (x[i] < 0.0 || x[i] > 1.0) {
      y[i] = nan("");
      continue;
    }
    if (x[i] == 0.0) {
      y[i] = 0.0;
      continue;
    }
    if (x[i] == 1.0) {
      y[i] = 1.0;
      continue;
    }

    // Check for asymptotic cases
    // We do not take those cases into account
    if (alpha > 1e5 && beta < 10 && x[i] > alpha / (alpha + beta))
      continue;

    if (beta > 1e5 && alpha < 10 && x[i] < beta / (alpha + beta))
      continue;

    // Execute the continued fraction
    // Measure the number of iterations
    if (x[i] < (alpha + 1.0) / (alpha + beta + 2.0))
    {
      /* Apply continued fraction directly. */
      epsabs = 0.;

      double cf = tmp_beta_cont_frac_measure(alpha, beta, x[i], epsabs, iter_count.at(i));

      y[i] = y[i] * cf / alpha;
    }
    else
    {
      /* Apply continued fraction after hypergeometric transformation. */
      epsabs = fabs(1. / (y[i] / beta)) * CUDA_DBL_EPSILON;
      double cf = tmp_beta_cont_frac_measure(beta, alpha, 1.0 - x[i], epsabs, iter_count.at(i));
      double term = y[i] * cf / beta;

      y[i] = 1 - term;
    }
  }

  // print the results to output_file
  std::cout << "X IterCount" << std::endl;
  std::cout << "0 0" << std::endl;
  for (double i = 0; i < x.size(); ++i)
    std::cout << x[i] << " " << iter_count[i] << std::endl;
  std::cout << "1 0" << std::endl;
  return 0;
}