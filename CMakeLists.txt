cmake_minimum_required(VERSION 2.8.3)
project(numerical_methods)

find_package(catkin_simple REQUIRED)
catkin_simple(ALL_DEPS_REQUIRED)

add_definitions(-std=c++11)

cs_add_library(${PROJECT_NAME} src/numerical-methods.cc)

cs_install()
cs_export()
