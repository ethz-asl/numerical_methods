cmake_minimum_required(VERSION 2.8.3)
project(numerical_methods)

find_package(catkin_simple REQUIRED)
catkin_simple(ALL_DEPS_REQUIRED)

add_definitions(-std=c++11)

cs_add_library(${PROJECT_NAME} src/numerical-methods.cc)

catkin_add_gtest(test_integration_methods test/test-integration-methods.cc)
target_link_libraries(test_integration_methods ${PROJECT_NAME})

cs_install()
cs_export()
