#include "src_ref/BetaDistGsl.hpp"
#include "src_cuda/BetaDistCuda.hpp"

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <ctime>
#include <iostream>
#include <vector>
#include <omp.h>

#include <gtest/gtest.h>

#define SMALL_SIZE 1e7
#define MID_SIZE 1e8
#define LARGE_SIZE 1e9

#define PRECISION_TOLERANCE_DOUBLE 1e-7
#define PRECISION_TOLERANCE_FLOAT 1e-2
#define PRECISION_TOLERANCE_VERY_CLOSE_EQ 1e-15
#define PRECISION_TOLERANCE_CLOSE_EQ 1e-13

using std::vector;
using std::cerr;
using std::endl;

using profile_clock_t = std::chrono::high_resolution_clock;
using profile_duration_t = std::chrono::duration<double>;

class BETA_TEST : public ::testing::Test {
protected:
  void SetUp() override {
    // Code here will be called immediately after the constructor (right
    // before each test).
    srand(seed);

    x.resize(SMALL_SIZE);

    for (int i = 0; i < SMALL_SIZE; i++) {
      x[i] = rand() / (double)RAND_MAX;
    }

    alpha = .1;
    beta = .1;
  }

const unsigned int seed = time(NULL);
vector<double> x;
double alpha;
double beta;
};

class BIG_BETA_TEST : public ::testing::Test {
protected:
  void SetUp() override {
    // Code here will be called immediately after the constructor (right
    // before each test).
    srand(seed);

    x.resize(LARGE_SIZE);

    for (int i = 0; i < LARGE_SIZE; i++) {
      x[i] = rand() / (double)RAND_MAX;
    }

    alpha = .1;
    beta = .1;
  }

const unsigned int seed = time(NULL);
vector<double> x;
double alpha;
double beta;
};

// Test case for #pdf
TEST_F(BETA_TEST, SmallGSLTestPDF) {
  vector<double> y1(SMALL_SIZE);
  cerr << "Size of x: " << x.size() << endl;
  for (int j = 0; j < SMALL_SIZE; j++) {
    y1.at(j) = betapdf(x.at(j), alpha, beta);
  }
}

// Test case for #pdf
TEST_F(BETA_TEST, SmallCUDATestPDF) {
  vector<double> y2(x.size());
  betapdf_cuda(x.data(), y2.data(), alpha, beta, x.size());
}

// Test case for #pdf
TEST_F(BETA_TEST, SmallCUDATestFloatPDF) {
  vector<float> x_f(SMALL_SIZE), y_f(SMALL_SIZE);
  std::transform(x.begin(), x.end(), x_f.begin(), [](double d) { return (float)d; });
  betapdf_cuda(x_f.data(), y_f.data(), static_cast<float>(alpha), static_cast<float>(beta) , SMALL_SIZE);
}

// Test case for #pdf
TEST_F(BETA_TEST, SmallComaprisonPDF) {

  vector<double> y1(SMALL_SIZE), y2(SMALL_SIZE);

  for (int j = 0; j < SMALL_SIZE; j++) {
    y1.at(j) = betapdf(x.at(j), alpha, beta);
  }
  betapdf_cuda(x.data(), y2.data(), alpha, beta, SMALL_SIZE);

  for (int j = 0; j < SMALL_SIZE; j++) {
    EXPECT_NEAR(y1.at(j), y2.at(j), PRECISION_TOLERANCE_DOUBLE);
  }
}

// Test case for #pdf
TEST_F(BETA_TEST, SmallComparisonFloatPDF) {

  vector<double> y1(SMALL_SIZE), y2(SMALL_SIZE);
  vector<float> x_f(SMALL_SIZE), y_f(SMALL_SIZE);
  std::transform(x.begin(), x.end(), x_f.begin(), [](double d) { return (float)d; });

  for (int j = 0; j < SMALL_SIZE; j++) {
    y1.at(j) = betapdf(x.at(j), alpha, beta);
  }
  betapdf_cuda(x_f.data(), y_f.data(), static_cast<float>(alpha), static_cast<float>(beta) , SMALL_SIZE);
  std::transform(y_f.begin(), y_f.end(), y2.begin(), [](float f) { return (double)f; });

  for (int j = 0; j < SMALL_SIZE; j++) {
    EXPECT_NEAR(y1.at(j), y2.at(j), PRECISION_TOLERANCE_FLOAT);
  }
}

// Test case for #cdf
TEST_F(BETA_TEST, SmallGSLTestCDF) {
  vector<double> y1(SMALL_SIZE);
  cerr << "Size of x: " << x.size() << endl;
  for (int j = 0; j < SMALL_SIZE; j++) {
    y1.at(j) = betacdf(x.at(j), alpha, beta);
  }
}

// Test case for #cdf
TEST_F(BETA_TEST, SmallCUDATestCDF) {
  vector<double> y2(x.size());
  betacdf_cuda(x.data(), y2.data(), alpha, beta, x.size());
}

// Test case for #cdf
TEST_F(BETA_TEST, SmallComaprisonCDF) {

  vector<double> y1(SMALL_SIZE), y2(SMALL_SIZE);

  for (int j = 0; j < SMALL_SIZE; j++) {
    y1.at(j) = betacdf(x.at(j), alpha, beta);
  }
  betacdf_cuda(x.data(), y2.data(), alpha, beta, SMALL_SIZE);

  for (int j = 0; j < SMALL_SIZE; j++) {
    EXPECT_NEAR(y1.at(j), y2.at(j), PRECISION_TOLERANCE_CLOSE_EQ);
  }
}

// Test case for #cdf
TEST_F(BIG_BETA_TEST, BigComaprisonCDF) {

  vector<double> y1(MID_SIZE), y2(MID_SIZE);

  size_t size = MID_SIZE;

  omp_set_num_threads( omp_get_num_procs() );
  #pragma omp parallel for
  for (size_t j = 0; j < size; j++) {
    y1.at(j) = betacdf(x.at(j), alpha, beta);
  }

  betacdf_cuda(x.data(), y2.data(), alpha, beta, MID_SIZE);

  for (int j = 0; j < 100; j++) {
    EXPECT_NEAR(y1.at(j), y2.at(j), PRECISION_TOLERANCE_CLOSE_EQ);
  }
}

int 
main (int argc, char *argv[]) {

  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
    
  //auto start = profile_clock_t::now();
  //auto end = profile_clock_t::now();
  //cerr << "Time = " << profile_duration_t(end - start).count() << " itr[" << i << "]" << endl;
}
