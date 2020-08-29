cmake_minimum_required(VERSION 3.14)
project(optimization)

find_package(Eigen3 CONFIG REQUIRED)
find_package(glog CONFIG REQUIRED)

set(CMAKE_CXX_STANDARD_REQUIRED 17)
set(CMAKE_BUILD_TYPE Release)

include_directories(${PROJECT_SOURCE_DIR}/inc)
add_library(${PROJECT_NAME} src/edge.cpp src/loss.cpp src/problem.cpp src/vertex.cpp)
target_link_libraries(${PROJECT_NAME} Eigen3::Eigen glog::glog)

add_executable(CurveFitting test/CurveFitting.cpp)
target_link_libraries(CurveFitting ${PROJECT_NAME} glog::glog Eigen3::Eigen)
