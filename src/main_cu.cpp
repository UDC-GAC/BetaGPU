#include "BetaDistCuda.hpp"

#include <chrono>
#include <cstdlib>
#include <ctime>
#include <iostream>
#include <vector>

using std::vector;
using std::cerr;
using std::endl;

using profile_clock_t = std::chrono::high_resolution_clock;
using profile_duration_t = std::chrono::duration<double>;

int 
main () {
  
  vector<double> x(10e7);
  vector <double> y(10e7);
  

  for (int i = 0; i < 10e7; i++) {
    x[i] = rand() / (double)RAND_MAX;
  }

  for (int i = 1; i <= 100; i++) {
    double alpha = 0.1 * i;
    double beta = 0.1 * i;
    
    auto start = profile_clock_t::now();
    betapdf_cuda(x.data(), y.data(), alpha, beta, x.size());
    auto end = profile_clock_t::now();

    cerr << "Time = " << profile_duration_t(end - start).count() << " itr[" << i << "]" << endl;
  }
}
