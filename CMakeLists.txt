cmake_minimum_required(VERSION 3.15...3.31)
project(
    FractalDimensionComputation_CPU
    VERSION 1.0
    LANGUAGES CXX)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(
    FractalDimensionComputation_CPU
    main.cpp bcCPU.cpp FastDBC.cpp)
target_link_libraries(FractalDimensionComputation_CPU PRIVATE ${OpenCV_LIBS})
