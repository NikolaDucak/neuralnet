cmake_minimum_required(VERSION 3.10)

project(neural_net_lib)

find_package(Eigen3 3.3 REQUIRED NO_MODULE)
find_package(Boost 1.40 REQUIRED serialization)

add_library(neural_net_lib inc/neural_net.h src/neural_net.cpp inc/eigen_serialization.h)

include_directories(./inc)
include_directories(${Boost_INCLUDE_DIR})

target_link_libraries(neural_net_lib LINK_PUBLIC Boost::serialization Eigen3::Eigen)
target_include_directories(neural_net_lib PUBLIC ./inc)
