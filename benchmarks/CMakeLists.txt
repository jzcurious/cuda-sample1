cmake_minimum_required(VERSION 3.24...3.30)
project(cuda-sample1-perf VERSION 1.0 LANGUAGES CXX CUDA)

include(FetchContent)

FetchContent_Declare(
  indicators
  GIT_REPOSITORY https://github.com/p-ranav/indicators.git
  GIT_TAG v2.3
)
FetchContent_MakeAvailable(indicators)

FetchContent_Declare(
  argparse
  GIT_REPOSITORY https://github.com/p-ranav/argparse.git
  GIT_TAG v3.1
)
FetchContent_MakeAvailable(argparse)

add_executable(dotbench
    ${CMAKE_CURRENT_SOURCE_DIR}/src/timers.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/src/dotbench.cu
)

target_include_directories(dotbench PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../cuda-sample1-lib/include
    ${CMAKE_CURRENT_SOURCE_DIR}/../tests/include
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/../build/_deps/indicators-src/include
)

find_package(CUDAToolkit)
target_link_libraries(dotbench PRIVATE
    cuda-sample1-lib
    CUDA::curand
    argparse
)

configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/scripts/vis.py
    ${CMAKE_CURRENT_BINARY_DIR}/vis.py
    COPYONLY
)

