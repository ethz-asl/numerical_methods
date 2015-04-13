cmake_minimum_required(VERSION 2.8.3)
project(numerical_methods)

find_package(catkin_simple REQUIRED)
catkin_simple(ALL_DEPS_REQUIRED)

add_definitions(-std=c++11)

catkin_add_gtest(test_integration_methods test/test-integration-methods.cc)
target_link_libraries(test_integration_methods ${PROJECT_NAME} pthread)

catkin_add_gtest(test_log_integration_methods test/test-log-integration-methods.cc)
target_link_libraries(test_log_integration_methods ${PROJECT_NAME} pthread)

cs_install()
cs_export()
