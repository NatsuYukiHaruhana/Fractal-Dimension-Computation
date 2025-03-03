cmake_minimum_required(VERSION 3.15...3.31)
project(
    FractalDimensionComputation
    VERSION 1.0
    LANGUAGES CXX)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

set(EXE_CPU FractalDimensionComputation_CPU)
add_executable(${EXE_CPU} main.cpp bcCPU.cpp FastDBC.cpp)
target_link_libraries(${EXE_CPU} PRIVATE ${OpenCV_LIBS})

if (CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(${EXE_CPU} PRIVATE -Wall -Wextra -Wpedantic)
elseif (MSVC)
    target_compile_options(${EXE_CPU} PRIVATE /W4)
endif()
