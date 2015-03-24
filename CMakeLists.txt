cmake_minimum_required(VERSION 2.8.3)
project(numerical_methods)

find_package(catkin_simple REQUIRED COMPONENTS
  glog_catkin
)

catkin_simple()

add_definitions(-std=c++11)

cs_add_library(${PROJECT_NAME} src/numerical-methods.cc)

catkin_add_gtest(test_direct_search test/test-direct-search.cc)
target_link_libraries(test_direct_search ${PROJECT_NAME})

catkin_add_gtest(test_extrapolation test/test-extrapolation.cc)
target_link_libraries(test_extrapolation ${PROJECT_NAME})

catkin_add_gtest(test_integration test/test-integration.cc)
target_link_libraries(test_integration ${PROJECT_NAME})

catkin_add_gtest(test_linear_regression test/test-linear-regression.cc)
target_link_libraries(test_linear_regression ${PROJECT_NAME})

catkin_add_gtest(test_utility test/test-utility.cc)
target_link_libraries(test_utility ${PROJECT_NAME})

cs_install()
cs_export()
