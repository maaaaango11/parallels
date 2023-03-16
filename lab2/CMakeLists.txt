cmake_minimum_required(VERSION 3.21)
project(lab222)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -fopenmp")

add_executable(lab222 main.cpp)
