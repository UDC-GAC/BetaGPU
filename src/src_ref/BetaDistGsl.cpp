#include "BetaDistGsl.hpp"

#include <gsl/gsl_cdf.h>
#include <gsl/gsl_randist.h>

// Probability density function
// Look https://github.com/ampl/gsl/blob/master/randist/beta.c
double betapdf(double x, double alpha, double beta) {
  return gsl_ran_beta_pdf(x, alpha, beta);
}

double betacdf(double x, double alpha, double beta) {
  return gsl_cdf_beta_P(x, alpha, beta);
}

