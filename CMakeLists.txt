cmake_minimum_required(VERSION 3.16)
project(micrograd VERSION 0.1.0 LANGUAGES C CXX CUDA)
set(CMAKE_CXX_COMPILER /usr/local/cuda-12.3/bin/nvcc) 
set(CMAKE_CUDA_ARCHITECTURES 86)
set(CMAKE_CUDA_COMPILER  /usr/local/cuda-12.3/bin/nvcc)
set(CMAKE_CUDA_FLAGS ${CMAKE_CUDA_FLAGS} "-g -G")  # enable cuda-gdb
set(CMAKE_BUILD_TYPE Debug)

include(CTest)
enable_testing()
include_directories(${CMAKE_SOURCE_DIR}/include)
# add_library(engine SHARED engine.cu)
# add_library(tensor SHARED tensor.cu)
# add_library(compute1d SHARED compute1d.cu)
# add_library(compute2d SHARED compute2d.cu)
# target_include_directories(engine PUBLIC ${CMAKE_SOURCE_DIR}/include)
# target_include_directories(tensor PUBLIC ${CMAKE_SOURCE_DIR}/include)
# target_include_directories(compute1d PUBLIC ${CMAKE_SOURCE_DIR}/include)
# target_include_directories(compute2d PUBLIC ${CMAKE_SOURCE_DIR}/include)



add_executable(main main.cu)
target_sources(main PRIVATE ${CMAKE_SOURCE_DIR}/main.cu)
# set_target_properties(${PROJECT_NAME} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
# target_link_libraries(main engine)
# target_link_libraries(main tensor)
# target_link_libraries(main compute1d)
# target_link_libraries(main compute2d)

include(CPack)
