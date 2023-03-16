cmake_minimum_required(VERSION 3.21)
project(lab4GoL)

set(CMAKE_CXX_STANDARD 14)
find_package(MPI REQUIRED)


add_executable(lab4GoL main.cpp)

target_link_libraries(lab4GoL PUBLIC MPI::MPI_CXX)