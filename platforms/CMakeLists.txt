cmake_minimum_required(VERSION 3.0)

project(admp)

find_package(autodiff)

add_executable(admp main.cpp)

target_link_libraries(admp autodiff::autodiff)