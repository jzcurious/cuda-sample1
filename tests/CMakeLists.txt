cmake_minimum_required(VERSION 3.24...3.30)
project(cuda-sample1-test VERSION 1.0 LANGUAGES CXX CUDA)

include(GoogleTest)
include(FetchContent)

FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG v1.15.2
)

FetchContent_MakeAvailable(googletest)

add_executable(cuda-sample1-test
    ${CMAKE_CURRENT_SOURCE_DIR}/src/tests.cu
)

target_include_directories(cuda-sample1-test PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../cuda-sample1-lib/include
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

find_package(CUDAToolkit)

target_link_libraries(cuda-sample1-test PRIVATE
    cuda-sample1-lib
    GTest::gtest_main
    CUDA::curand
)

enable_testing()
gtest_discover_tests(cuda-sample1-test)

