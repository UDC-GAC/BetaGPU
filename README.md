# Beta Distribution Library

This repository contains a C++/CUDA library for computing the Beta Probability Density Function (PDF) and Cumulative Distribution Function (CDF) using the regularized incomplete beta function. The library is optimized for high performance using CUDA for GPU acceleration and OpenMP for parallel processing on CPUs.

## Compilation Requirements

To compile the library, you need the following:

- A C++ compiler (e.g., `g++`)
- CUDA Toolkit (e.g., `nvcc`)
- GNU Scientific Library (GSL)
- OpenMP
- CMake (for building the project)

## Installing and Building Instructions

1. **Clone the repository**:
   ```sh
   git clone https://github.com/yourusername/beta_dev.git
   cd beta_dev
   ```

2. **Create a build directory and navigate to it**:
   ```sh
   mkdir build
   cd build
   ```

3. **Run CMake to configure the project**:
   ```sh
   cmake ..
   ```

4. **Build the project**:
   ```sh
   make
   ```

5. **Run the example**:
   ```sh
   ./example
   ```

## Functions Provided

The library provides the following functions:

- `betapdf_kernel`: Computes the Beta PDF for double precision inputs.
- `betapdf_kernel_f`: Computes the Beta PDF for single precision inputs.
- `betacdf_kernel`: Computes the Beta CDF for double precision inputs.
- `betacdf_kernel_f`: Computes the Beta CDF for single precision inputs.

## Usage Example

Here is an example of how to use the library to compute the Beta PDF:

```cpp
#include "BetaDistCuda.hpp"
#include <iostream>
#include <vector>

int main() {
    // Input data
    std::vector<double> x = {0.1, 0.5, 0.9};
    std::vector<double> y(x.size());
    double alpha = 2.0;
    double beta = 5.0;
    size_t size = x.size();

    // Allocate device memory
    double *d_x, *d_y;
    cudaMalloc(&d_x, size * sizeof(double));
    cudaMalloc(&d_y, size * sizeof(double));

    // Copy data to device
    cudaMemcpy(d_x, x.data(), size * sizeof(double), cudaMemcpyHostToDevice);

    // Launch kernel
    int blockSize = 256;
    int numBlocks = (size + blockSize - 1) / blockSize;
    betapdf_kernel<<<numBlocks, blockSize>>>(d_x, d_y, alpha, beta, size);

    // Copy result back to host
    cudaMemcpy(y.data(), d_y, size * sizeof(double), cudaMemcpyDeviceToHost);

    // Print result
    for (size_t i = 0; i < size; ++i) {
        std::cout << "Beta PDF(" << x[i] << ") = " << y[i] << std::endl;
    }

    // Free device memory
    cudaFree(d_x);
    cudaFree(d_y);

    return 0;
}
```

## License

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.