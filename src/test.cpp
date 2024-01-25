#include "src_ref/BetaDistGsl.hpp"

#include <chrono>
#include <cstdlib>
#include <ctime>
#include <iostream>
#include <vector>

#include <gtest/gtest.h>

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
    srand(time(NULL));

    x.resize(10e7);

    for (int i = 0; i < 10e7; i++) {
      x[i] = rand() / (double)RAND_MAX;
    }

    alpha = 0.1;
    beta = 0.1;
  }

vector<double> x;
double alpha;
double beta;
};


// Test case for #cdf
TEST_F(BETA_TEST, LoopTestCDF) {

  for (int j = 0; j < 10e7; j++) {
    betacdf(x.at(j), alpha, beta);
  }
}

// Test case for #pdf
TEST_F(BETA_TEST, LoopTestPDF) {

  for (int j = 0; j < 10e7; j++) {
    betapdf(x.at(j), alpha, beta);
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
