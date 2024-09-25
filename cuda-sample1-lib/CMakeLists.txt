cmake_minimum_required(VERSION 3.24...3.30)
project(cuda-sample1-lib VERSION 1.0 LANGUAGES CUDA)

add_library(cuda-sample1-lib STATIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src/dotprod.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/src/dynblock.cu
)

target_include_directories(cuda-sample1-lib PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

set_target_properties(cuda-sample1-lib PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(cuda-sample1-lib PROPERTIES CUDA_ARCHITECTURES "native")

if(CMAKE_BUILD_TYPE MATCHES "DebugDevice")
    message(STATUS "Device debugging mode is enabled")
    target_compile_options(cuda-sample1-lib
        PUBLIC $<$<COMPILE_LANGUAGE:CUDA>:-g -G>
    )
endif()

if(CMAKE_BUILD_TYPE MATCHES "DebugHost")
    message(STATUS "Host debugging mode is enabled")
    target_compile_options(cuda-sample1-lib PUBLIC
        $<$<COMPILE_LANGUAGE:CUDA>:-g -Xcompiler -fsanitize=address -fsanitize=undefined>
    )
    target_link_libraries(cuda-sample1-lib PUBLIC
        asan
    )
    target_link_options(cuda-sample1-lib
        BEFORE PUBLIC -fsanitize=undefined
        PUBLIC -fsanitize=address
    )
endif()

if(CMAKE_BUILD_TYPE MATCHES "Release")
    message(STATUS "Release mode is enabled")
endif()

