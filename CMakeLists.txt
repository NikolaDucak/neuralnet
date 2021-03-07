cmake_minimum_required(VERSION 3.17)
project(nncli)
set(CMAKE_CXX_STANDARD 17)

add_subdirectory(neural_net_lib)

add_executable(nncli main.cpp)

target_link_libraries(nncli neural_net_lib)
