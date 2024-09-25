cmake_minimum_required(VERSION 3.24...3.30)
project(cuda-sample1 VERSION 1.0 LANGUAGES CXX CUDA)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -pedantic -O3")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Wall -O3")
set(CMAKE_CUDA_USE_RESPONSE_FILE_FOR_INCLUDES OFF)

add_subdirectory(cuda-sample1-lib)

if("${TESTING}" STREQUAL 1)
    message(STATUS "Testing is enabled")
    add_subdirectory(tests)
endif()

if("${BENCHMARKING}" STREQUAL 1)
    message(STATUS "Benchmarking is enabled")
    add_subdirectory(benchmarks)
endif()

