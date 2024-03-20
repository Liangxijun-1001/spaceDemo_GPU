# cmake_minimum_required(VERSION 3.0)

# project(main CXX)

# set(CMAKE_BUILD_TYPE Debug)

# find_package(HIP REQUIRED)

# if (NOT DEFINED ROCM_PATH)
#     set(ROCM_PATH "/opt/rocm/" CACHE STRING "ROCM installation direcotry.")
# endif()

# # Search for rocm in common locations
# list(APPEND CMAKE_PREFIX_PATH ${ROCM_PATH}/hip ${ROCM_PATH})

# # find hip
# find_package(hip PATHS /opt/rocm/lib/cmake/hip NO_DEFAULT_PATH_REQUIRED)

# # set compiler and linker
# message("111111111111111111111")
# message(${HIP_HIPCC_EXECUTABLE})

# set (CMAKE_CXX_COMPILER ${HIP_HIPCC_EXECUTABLE})

# set (CMAKE_CXX_LINKER ${HIP_HIPCC_EXECUTABLE})

# set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")

# include_directories(
#     "${CMAKE_CURRENT_SOURCE_DIR}",
#     "/opt/rocm/include"
#     # "/home/liangnus_docker/workspace/rocm/rocm/include"
# )

# file(GLOB srcs
#     "${CMAKE_CURRENT_SOURCE_DIR}/*"
# )

# link_directories(
#     "/opt/rocm/lib"
#     #"/home/liangnus/workspace/rocm/rocm/hip/lib"
#     # "/home/liangnus_docker/workspace/rocm/rocm/hip/lib"
# )

# link_directories(
#     #"/home/liangnus/workspace/rocm/rocm/lib"
#     # "/home/liangnus_docker/workspace/rocm/rocm/lib"
# )

# link_directories(
#     "/opt/rocm-5.7.0/lib"
# )

# add_executable(main  spacecode_2d.cpp main_test.cpp)

# target_link_libraries(main amdhip64 amd_comgr hsa-runtime64)

cmake_minimum_required(VERSION 3.0)

find_package(CUDA REQUIRED)

set(CUDA_NVCC_FLAGS -g;-G)

SET(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}; -gencode arch=compute_86,code=sm_86;)

cuda_add_executable(main spacecode_2d.cu main_test.cpp)
#cuda_add_executable(main test.cu)
