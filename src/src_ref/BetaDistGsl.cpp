#include "BetaDistGsl.hpp"

#include <gsl/gsl_cdf.h>
#include <gsl/gsl_randist.h>
#include <gsl/gsl_sf_gamma.h>
#include <cmath>
#include <limits>

// Probability density function
// Look https://github.com/ampl/gsl/blob/master/randist/beta.c
double betapdf(double x, double alpha, double beta) {
  return gsl_ran_beta_pdf(x, alpha, beta);
}

double betacdf(double x, double alpha, double beta) {
  return gsl_cdf_beta_P(x, alpha, beta);
}

double other_beta_cont_frac (const double a, const double b, const double x,
                const double epsabs)
{
  const unsigned int max_iter = 512;    /* control iterations      */
  const double cutoff = 2.0 * OTHER_DBL_MIN;      /* control the zero cutoff */
  unsigned int iter_count = 0;
  double cf;

  /* standard initialization for continued fraction */
  double num_term = 1.0;
  double den_term = 1.0 - (a + b) * x / (a + 1.0);

  if (fabs (den_term) < cutoff)
    den_term = std::numeric_limits<double>::quiet_NaN();

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
        den_term = std::numeric_limits<double>::quiet_NaN();

      if (fabs (num_term) < cutoff)
        num_term = std::numeric_limits<double>::quiet_NaN();

      den_term = 1.0 / den_term;

      delta_frac = den_term * num_term;
      cf *= delta_frac;

      coeff = -(a + k) * (a + b + k) * x / ((a + 2 * k) * (a + 2 * k + 1.0));

      /* second step */
      den_term = 1.0 + coeff * den_term;
      num_term = 1.0 + coeff / num_term;

      if (fabs (den_term) < cutoff)
        den_term = std::numeric_limits<double>::quiet_NaN();

      if (fabs (num_term) < cutoff)
        num_term = std::numeric_limits<double>::quiet_NaN();

      den_term = 1.0 / den_term;

      delta_frac = den_term * num_term;
      cf *= delta_frac;

      if (fabs (delta_frac - 1.0) < 2.0 * OTHER_DBL_EPSILON)
        break;

      if (cf * fabs (delta_frac - 1.0) < epsabs)
        break;

      ++iter_count;
    }

  if (iter_count >= max_iter)
    return std::numeric_limits<double>::quiet_NaN();

  return cf;
}

double other_beta_inc(double x, double a, double b, double A, double Y)
{
  if (x == 0.0)
  {
    return A * 0 + Y;
  }
  else if (x == 1.0)
  {
    return A * 1 + Y;
  }
  else if (a > 1e5 && b < 10 && x > a / (a + b))
  {
    /* Handle asymptotic regime, large a, small b, x > peak [AS 26.5.17] */
    double N = a + (b - 1.0) / 2.0;
    return A * gsl_sf_gamma_inc_Q(b, -N * log(x)) + Y;
  }
  else if (b > 1e5 && a < 10 && x < b / (a + b))
  {
    /* Handle asymptotic regime, small a, large b, x < peak [AS 26.5.17] */
    double N = b + (a - 1.0) / 2.0;
    return A * gsl_sf_gamma_inc_P(a, -N * log1p(-x)) + Y;
  }
  else
  {
    double ln_beta = gsl_sf_lnbeta(a, b);
    double ln_pre = -ln_beta + a * log(x) + b * log1p(-x);

    double prefactor = exp(ln_pre);

    if (x < (a + 1.0) / (a + b + 2.0))
    {
      /* Apply continued fraction directly. */
      double epsabs = fabs(Y / (A * prefactor / a)) * OTHER_DBL_EPSILON;

      double cf = other_beta_cont_frac(a, b, x, epsabs);

      return A * (prefactor * cf / a) + Y;
    }
    else
    {
      /* Apply continued fraction after hypergeometric transformation. */
      double epsabs =
          fabs((A + Y) / (A * prefactor / b)) * OTHER_DBL_EPSILON;
      double cf = other_beta_cont_frac(b, a, 1.0 - x, epsabs);
      double term = prefactor * cf / b;

      if (A == -Y)
      {
        return -A * term;
      }
      else
      {
        return A * (1 - term) + Y;
      }
    }
  }
}

double otherbetacdf(double x, double alpha, double beta) {
  if (x <= 0.0) {
    return 0.0;
  } else if (x >= 1.0) {
    return 1.0;
  } else {
    return other_beta_inc(x, alpha, beta, 1., 0.);
  }
}

