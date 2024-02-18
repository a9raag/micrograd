cmake_minimum_required(VERSION 3.0.0)
project(micrograd VERSION 0.1.0 LANGUAGES C CXX CUDA)
set(CMAKE_CXX_COMPILER /usr/local/cuda-12.3/bin/nvcc) 
set(CMAKE_CUDA_ARCHITECTURES 86)
set(CMAKE_CUDA_COMPILER  /usr/local/cuda-12.3/bin/nvcc)
include(CTest)
enable_testing()
include_directories(${CMAKE_SOURCE_DIR}/include)
add_library(engine SHARED engine.cu)
add_library(tensor SHARED tensor.cu)
target_include_directories(engine PUBLIC ${CMAKE_SOURCE_DIR}/include)

add_executable(main main.cu)


set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
target_include_directories(main PRIVATE ${CMAKE_SOURCE_DIR}/include)

target_link_libraries(main engine)
include(CPack)
