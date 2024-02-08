cmake_minimum_required(VERSION 3.20 FATAL_ERROR)
project(cmake_and_cuda LANGUAGES CXX CUDA)

enable_testing()

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin)
 
find_package(GSL REQUIRED)
find_package(GTest REQUIRED)
find_package(OpenMP REQUIRED)

include(GoogleTest)

file(GLOB_RECURSE GSL_LIB_FILES "src/src_ref/*.cpp")
file(GLOB_RECURSE CUDA_LIB_FILES "src/src_cuda/*.cu")

#Add flags to the C++ compilation
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -O0 -g")
#set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -use_fast_math")
#NCU para sacar reporte y luego nsight en mi maquina

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
set_target_properties( beta_cuda
                       PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(beta_cuda PROPERTIES CUDA_ARCHITECTURES "86")

 
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
target_link_libraries(test_gtest PRIVATE beta_gsl beta_cuda GTest::GTest GTest::Main)
set_target_properties( test_gtest
                       PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(test_gtest PROPERTIES CUDA_ARCHITECTURES "86")

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
set_target_properties(main_cuda_test PROPERTIES CUDA_ARCHITECTURES "86")
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
                       PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(benchmark_test PROPERTIES CUDA_ARCHITECTURES "86")
target_link_libraries(benchmark_test PRIVATE beta_cuda beta_gsl OpenMP::OpenMP_CXX)

if(APPLE)
  # We need to add the path to the driver (libcuda.dylib) as an rpath, 
  # so that the static cuda runtime can find it at runtime.
  set_property(TARGET benchmark_test 
               PROPERTY
               BUILD_RPATH ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})
endif()
