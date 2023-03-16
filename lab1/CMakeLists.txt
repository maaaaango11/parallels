cmake_minimum_required(VERSION 3.21)
project(lab1)

set(CMAKE_CXX_STANDARD 14)
find_package(MPI REQUIRED)

add_executable(lab1 main1.cpp)

target_link_libraries(lab1 PUBLIC MPI::MPI_CXX)