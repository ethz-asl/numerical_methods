cmake_minimum_required(VERSION 2.8.3)
project(numerical_methods)

set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

find_package(Eigen3 REQUIRED)
include_directories(EIGEN3_INCLUDE_DIR)

find_package(GTest REQUIRED)
include_directories(GTEST_INCLUDE_DIR)

add_definitions(-std=c++11)

add_library(${PROJECT_NAME} src/numerical-methods.cc)
target_link_libraries(${PROJECT_NAME} Eigen3 GTest)

enable_testing()

add_executable(test_integration_methods 
  test/test-integration-methods.cc
)
target_link_libraries(test_integration_methods 
  ${PROJECT_NAME} 
  gtest 
  pthread
)
add_test(test_integration_methods)

add_executable(test_log_integration_methods 
  test/test-log-integration-methods.cc
)
target_link_libraries(test_log_integration_methods 
  ${PROJECT_NAME} 
  gtest 
  pthread
)
add_test(test_log_integration_methods)

add_executable(test_optimization_methods 
  test/test-optimization-methods.cc
)
target_link_libraries(test_optimization_methods 
  ${PROJECT_NAME} 
  gtest 
  pthread
)
add_test(test_optimization_methods)

add_executable(test_incremental_statistics 
  test/test-incremental-statistics.cc
)
target_link_libraries(test_incremental_statistics 
  ${PROJECT_NAME} 
  gtest 
  pthread
)
add_test(test_incremental_statistics)