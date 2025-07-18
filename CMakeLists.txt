# MIT License
# Copyright (c) Meta Platforms, Inc. and affiliates.
# See LICENSE file for details.

cmake_minimum_required(VERSION 3.18)
project(cutracer LANGUAGES CXX CUDA)

# Set C++ and CUDA standards
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# CUDA architecture
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES "all")
endif()

# Check NVCC version
execute_process(
    COMMAND nvcc --version
    OUTPUT_VARIABLE NVCC_VERSION_OUTPUT
    ERROR_QUIET
)

string(REGEX MATCH "release ([0-9]+\\.[0-9]+)" NVCC_VERSION_MATCH "${NVCC_VERSION_OUTPUT}")
if(NVCC_VERSION_MATCH)
    set(NVCC_VERSION ${CMAKE_MATCH_1})
    if(NVCC_VERSION VERSION_LESS "10.1")
        message(FATAL_ERROR "ERROR: nvcc version >= 10.1 required to compile an nvbit tool! Found version ${NVCC_VERSION}")
    endif()
else()
    message(FATAL_ERROR "Could not determine NVCC version")
endif()

# Check PTXAS version for maxrregcount flag
execute_process(
    COMMAND ptxas --version
    OUTPUT_VARIABLE PTXAS_VERSION_OUTPUT
    ERROR_QUIET
)

string(REGEX MATCH "release ([0-9]+\\.[0-9]+)" PTXAS_VERSION_MATCH "${PTXAS_VERSION_OUTPUT}")
set(MAXRREGCOUNT_FLAG "")
if(PTXAS_VERSION_MATCH)
    set(PTXAS_VERSION ${CMAKE_MATCH_1})
    if(PTXAS_VERSION VERSION_LESS "12.3")
        set(MAXRREGCOUNT_FLAG "-maxrregcount=24")
    endif()
endif()

# Debug or Release build
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -g -O0")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -O0")
else()
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 -g")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -g")
endif()

# Directory structure
set(SRC_DIR "${CMAKE_SOURCE_DIR}/src")
set(INCLUDE_DIR "${CMAKE_SOURCE_DIR}/include")
set(NVBIT_PATH "${CMAKE_SOURCE_DIR}/third_party/nvbit/core")

# Check if NVBIT exists
if(NOT EXISTS "${NVBIT_PATH}/libnvbit.a")
    message(FATAL_ERROR "NVBIT library not found at ${NVBIT_PATH}/libnvbit.a")
endif()

# Include directories
include_directories(${NVBIT_PATH})
include_directories(${INCLUDE_DIR})

# Find CUDA
find_package(CUDAToolkit REQUIRED)

# Get CUDA library path
get_filename_component(CUDA_LIB_DIR ${CUDAToolkit_LIBRARY_DIR} ABSOLUTE)

# Source files
file(GLOB REGULAR_CU_SRCS "${SRC_DIR}/*.cu")
list(REMOVE_ITEM REGULAR_CU_SRCS "${SRC_DIR}/inject_funcs.cu")

# Create object library for regular CUDA files
add_library(cutracer_regular_objs OBJECT ${REGULAR_CU_SRCS})

target_compile_options(cutracer_regular_objs PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-dc>
    $<$<COMPILE_LANGUAGE:CUDA>:-Xptxas -cloning=no>
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler -Wall>
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler -fPIC>
)

# Create object library for inject_funcs.cu with special flags
add_library(inject_funcs_obj OBJECT "${SRC_DIR}/inject_funcs.cu")

target_compile_options(inject_funcs_obj PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:${MAXRREGCOUNT_FLAG}>
    $<$<COMPILE_LANGUAGE:CUDA>:-Xptxas -astoolspatch>
    $<$<COMPILE_LANGUAGE:CUDA>:--keep-device-functions>
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler -Wall>
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler -fPIC>
)

# Main shared library
add_library(cutracer SHARED 
    $<TARGET_OBJECTS:cutracer_regular_objs>
    $<TARGET_OBJECTS:inject_funcs_obj>
)

# Set output directory and filename
set_target_properties(cutracer PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/lib"
    OUTPUT_NAME "cutracer"
    PREFIX ""
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON
)

# Link libraries
target_link_libraries(cutracer PRIVATE
    ${NVBIT_PATH}/libnvbit.a
    CUDA::cuda_driver
    CUDA::cudart_static
)

# Link directories
target_link_directories(cutracer PRIVATE ${CUDA_LIB_DIR})

# Custom target for cleaning
add_custom_target(clean-all
    COMMAND ${CMAKE_COMMAND} -E remove_directory "${CMAKE_SOURCE_DIR}/obj"
    COMMAND ${CMAKE_COMMAND} -E remove_directory "${CMAKE_SOURCE_DIR}/lib"
    COMMENT "Cleaning obj and lib directories"
)

# Install target
install(TARGETS cutracer
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION bin
)

# Print configuration information
message(STATUS "Project: ${PROJECT_NAME}")
message(STATUS "NVCC Version: ${NVCC_VERSION}")
message(STATUS "PTXAS Version: ${PTXAS_VERSION}")
message(STATUS "CUDA Architectures: ${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "Build Type: ${CMAKE_BUILD_TYPE}")
message(STATUS "Max register count flag: ${MAXRREGCOUNT_FLAG}")
