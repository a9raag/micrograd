# CMake 3.18 is required for check_language(CUDA), which lets configure succeed
# on systems where the CUDA toolkit is not installed.
cmake_minimum_required(VERSION 3.18)
project(micrograd VERSION 0.1.0 LANGUAGES CXX)

include(CTest)
enable_testing()

option(MICROGRAD_ENABLE_CUDA "Build the CUDA micrograd executable" ON)

if(MICROGRAD_ENABLE_CUDA)
    include(CheckLanguage)
    check_language(CUDA)

    if(CMAKE_CUDA_COMPILER)
        enable_language(CUDA)

        add_executable(micrograd src/main.cu)
        target_include_directories(micrograd PRIVATE
            ${CMAKE_SOURCE_DIR}/src
            ${CMAKE_SOURCE_DIR}/src/include
        )
        target_compile_features(micrograd PRIVATE cxx_std_17)
        set_target_properties(micrograd PROPERTIES
            CUDA_SEPARABLE_COMPILATION ON
            CUDA_STANDARD 17
            CUDA_STANDARD_REQUIRED ON
        )

        add_test(NAME micrograd_tensor2d_smoke COMMAND micrograd tensor2d)
        add_test(NAME micrograd_grad_check COMMAND micrograd grad-check)
    else()
        message(WARNING "CUDA compiler not found; skipping the micrograd executable target.")
    endif()
endif()

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)
