cmake_minimum_required(VERSION 3.20 FATAL_ERROR)
project(beta_dist_cuda LANGUAGES CXX CUDA)

enable_testing()

set (CMAKE_CXX_STANDARD 17
     CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_DEBUG_POSTFIX "d")

set(DEBUG_MACRO_FLAG "-DDEBUG")

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin)
 
find_package(GSL REQUIRED)
find_package(GTest REQUIRED)
find_package(OpenMP REQUIRED)

## Python binding
find_package(pybind11 REQUIRED)
set(PYBIND11_CPP_STANDARD -std=c++17)

include(GoogleTest)

include_directories(BEFORE ${CMAKE_SOURCE_DIR}/include/beta_cuda)

file(GLOB_RECURSE GSL_LIB_FILES "src/src_ref/*.cpp")
file(GLOB_RECURSE CUDA_LIB_FILES "src/src_cuda/*.cu")
file(GLOB_RECURSE CUDA_LIB_INCLUDES "include/beta_cuda/*.hpp")

# Compilation modes
if (NOT CMAKE_BUILD_TYPE)
  set (CMAKE_BUILD_TYPE Release)
endif()
set (CMAKE_CXX_FLAGS_DEBUG "-g -O0 ${DEBUG_MACRO_FLAG}")
set (CMAKE_CXX_FLAGS_RELEASE "-O3")

#Add flags to the C++ compilation
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra")
#set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -use_fast_math")
#NCU para sacar reporte y luego nsight en mi maquina

set(CMAKE_CUDA_FLAGS_DEBUG "-g -lineinfo ${DEBUG_MACRO_FLAG}")
set(CMAKE_CUDA_FLAGS_RELEASE "-O3 -use_fast_math --maxrregcount 32")

#
#
# library GSL
#
#
 
add_library(beta_gsl STATIC ${GSL_LIB_FILES} )
 
# Request that particles be built with -std=c++11
# As this is a public compile feature anything that links to 
# particles will also build with -std=c++11
target_compile_features(beta_gsl PUBLIC cxx_std_17)

# Link against the GSL library
target_link_libraries(beta_gsl PRIVATE GSL::gsl)

#
#
# library CUDA
#
#

add_library(beta_cuda STATIC ${CUDA_LIB_FILES} )
target_compile_features(beta_cuda PUBLIC cxx_std_17)
target_link_libraries(beta_cuda PRIVATE GSL::gsl OpenMP::OpenMP_CXX)
set_target_properties( beta_cuda
                       PROPERTIES 
                       POSITION_INDEPENDENT_CODE ON
                       CUDA_SEPARABLE_COMPILATION ON
                       CUDA_RESOLVE_DEVICE_SYMBOLS ON)
set_target_properties(beta_cuda PROPERTIES CUDA_ARCHITECTURES "75;80;86")

#move .hpp files to include
install(FILES ${CUDA_LIB_INCLUDES} DESTINATION ${CMAKE_INSTALL_PREFIX}/include/beta_cuda CONFIGURATIONS ${CMAKE_BUILD_TYPE})
install(TARGETS beta_cuda DESTINATION ${CMAKE_INSTALL_PREFIX}/lib CONFIGURATIONS ${CMAKE_BUILD_TYPE})
 
# We need to explicitly state that we need all CUDA files in the 
# particle library to be built with -dc as the member functions 
# could be called by other libraries and executables
#set_target_properties( beta_gsl
#                       PROPERTIES CUDA_SEPARABLE_COMPILATION ON)


#
#
# tests
#
#

add_executable(test_gtest src/test.cpp)
target_link_libraries(test_gtest PRIVATE beta_gsl beta_cuda OpenMP::OpenMP_CXX GTest::GTest GTest::Main )
set_target_properties( test_gtest
                       PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(test_gtest PROPERTIES CUDA_ARCHITECTURES "75;80;86")

gtest_discover_tests(test_gtest)
 
if(APPLE)
  # We need to add the path to the driver (libcuda.dylib) as an rpath, 
  # so that the static cuda runtime can find it at runtime.
  set_property(TARGET main_cuda_test 
               PROPERTY
               BUILD_RPATH ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})
endif()

#
#
# main_gsl
#
#

add_executable(main_gsl_test src/main_gsl.cpp)
 
#set_property(TARGET main_test 
#             PROPERTY CUDA_SEPARABLE_COMPILATION ON)
target_link_libraries(main_gsl_test PRIVATE beta_gsl)

#
#
# main_cu
#
#

add_executable(main_cuda_test src/main_cu.cpp)
set_target_properties( main_cuda_test
                       PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(main_cuda_test PROPERTIES CUDA_ARCHITECTURES "75;80;86")
target_link_libraries(main_cuda_test PRIVATE beta_cuda)
 
if(APPLE)
  # We need to add the path to the driver (libcuda.dylib) as an rpath, 
  # so that the static cuda runtime can find it at runtime.
  set_property(TARGET main_cuda_test 
               PROPERTY
               BUILD_RPATH ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})
endif()

#
#
# benchmark_test
#
#

add_executable(benchmark_test src/profiling_tests/benchmark_test.cpp)
set_target_properties( benchmark_test
                       PROPERTIES 
                       POSITION_INDEPENDENT_CODE ON
                       CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(benchmark_test PROPERTIES CUDA_ARCHITECTURES "75;80;86")
target_link_libraries(benchmark_test PRIVATE beta_cuda beta_gsl OpenMP::OpenMP_CXX)

if(APPLE)
  # We need to add the path to the driver (libcuda.dylib) as an rpath, 
  # so that the static cuda runtime can find it at runtime.
  set_property(TARGET benchmark_test 
               PROPERTY
               BUILD_RPATH ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})
endif()

#
#
# benchmark_cont_frac
#
#

add_executable(benchmark_cont_frac src/profiling_tests/benchmark_cont_frac.cpp)
set_target_properties( benchmark_cont_frac
                       PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(benchmark_cont_frac PROPERTIES CUDA_ARCHITECTURES "75;80;86")
target_link_libraries(benchmark_cont_frac PRIVATE beta_cuda beta_gsl OpenMP::OpenMP_CXX)

if(APPLE)
  # We need to add the path to the driver (libcuda.dylib) as an rpath, 
  # so that the static cuda runtime can find it at runtime.
  set_property(TARGET benchmark_cont_frac 
               PROPERTY
               BUILD_RPATH ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})
endif()

#
#
# benchmark_base_CUDA
#
#

add_executable(bench_base src/profiling_tests/bench.cu src/src_cuda/BetaDistCuda.cu)
set_target_properties( bench_base
                       PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(bench_base PROPERTIES CUDA_ARCHITECTURES "75;80;86")
target_compile_options(bench_base PRIVATE $<$<COMPILE_LANGUAGE:CUDA>: --compiler-options -fopenmp -Xcompiler=-fopenmp>)
target_link_libraries(bench_base PRIVATE beta_gsl OpenMP::OpenMP_CXX)

if(APPLE)
  # We need to add the path to the driver (libcuda.dylib) as an rpath, 
  # so that the static cuda runtime can find it at runtime.
  set_property(TARGET bench_base 
               PROPERTY
               BUILD_RPATH ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})
endif()


#
#
# Python binding
#
#

add_library(BetaDistCuda MODULE src/pybindings/BetaDistCuda.cpp)
set_target_properties(BetaDistCuda PROPERTIES PREFIX "${PYTHON_MODULE_PREFIX}"
                                           SUFFIX "${PYTHON_MODULE_EXTENSION}"
                                          CUDA_ARCHITECTURES "75;80;86")
target_link_libraries(BetaDistCuda PRIVATE beta_cuda OpenMP::OpenMP_CXX pybind11::module)
