cmake_minimum_required(VERSION 3.4)
project(norch_C)

# Set path to pybind11
find_package(pybind11 REQUIRED)

# Add the pybind11 module directly
pybind11_add_module(norch_C ${CMAKE_CURRENT_SOURCE_DIR}/norch/ATen/tensor.cpp)