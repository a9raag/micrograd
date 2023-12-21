cmake_minimum_required(VERSION 3.0.0)
project(micrograd VERSION 0.1.0 LANGUAGES C CXX)

include(CTest)
enable_testing()
include_directories(${CMAKE_SOURCE_DIR}/include)
add_library(engine SHARED engine.cpp)
target_include_directories(engine PUBLIC ${CMAKE_SOURCE_DIR}/include)
add_executable(nn nn.cpp)


set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
target_include_directories(nn PRIVATE ${CMAKE_SOURCE_DIR}/include)

target_link_libraries(nn engine)
include(CPack)
