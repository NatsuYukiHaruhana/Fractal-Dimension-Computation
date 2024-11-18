cmake_minimum_required(VERSION 3.15...3.31)
project(
    FractalDimensionComputation
    VERSION 1.0
    LANGUAGES CXX)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(
    FractalDimensionComputation
    main.cpp bcCPU.cpp)
target_link_libraries(FractalDimensionComputation PRIVATE ${OpenCV_LIBS})
