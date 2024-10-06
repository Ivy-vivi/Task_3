cmake_minimum_required(VERSION 3.0.0)
project(task)

# 设置 C++ 标准为 C++11
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# 查找 OpenCV
find_package(OpenCV REQUIRED)

# 查找 Eigen
find_package(Eigen3 REQUIRED)

# 查找 Ceres
find_package(Ceres REQUIRED)

# 包含头文件目录
include_directories(include ${OpenCV_INCLUDE_DIRS} ${EIGEN3_INCLUDE_DIR} ${CERES_INCLUDE_DIRS})

# 查找源文件
aux_source_directory(./src ALL_SRCS)

# 添加可执行文件
add_executable(task main.cpp ${ALL_SRCS})

# 链接库
target_link_libraries(task ${OpenCV_LIBS} ${CERES_LIBRARIES})
