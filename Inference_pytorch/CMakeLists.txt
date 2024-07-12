cmake_minimum_required(VERSION 3.10)

# Project name
project(NeuroSimPy)

# Set C++ standard
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Locate pybind11
set(pybind11_DIR "/home/mole/.conda/envs/hdnn/lib/python3.10/site-packages/pybind11/share/cmake/pybind11")
find_package(pybind11 REQUIRED)

# Compiler options
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp -O3 -w")

# Find all source files
file(GLOB ALL_SRC "NeuroSIM/*.cpp")
list(REMOVE_ITEM ALL_SRC "${CMAKE_CURRENT_BINARY_DIR}/NeuroSIM/main.cpp")
message("${ALL_SRC}")

# Create the pybind11 module
pybind11_add_module(neurosim_cpp NeuroSIM/neurosim_cpp.cpp ${ALL_SRC})