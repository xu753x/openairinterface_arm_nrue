cmake_minimum_required(VERSION 2.8)
project(run)
FIND_PACKAGE(CUDA REQUIRED)
# Pass options to NVCC
# 由于cuda采用NVCC编译而不是gCC编译，因此需要将参数传递给NVCC
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -lcufft)
# For compilation ...
# Specify target & source files to compile it from
CUDA_ADD_EXECUTABLE(run cuFFT2.cu)
# For linking ...
# Specify target & libraries to link it with
CUDA_ADD_CUFFT_TO_TARGET(run)
# 添加对gdb的支持
# SET(CMAKE_BUILD_TYPE "Debug")
# SET(CMAKE_CXX_FLAGS_DEBUG "$ENV{CXXFLAGS} -O0 -Wall -g2 -ggdb")
# SET(CMAKE_CXX_FLAGS_RELEASE "$ENV{CXXFLAGS} -O3 -Wall")
