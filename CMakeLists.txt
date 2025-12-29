CMAKE_MINIMUM_REQUIRED(VERSION 3.14)

set(CMAKE_CUDA_ARCHITECTURES 80)

project(code_demo LANGUAGES CXX CUDA)
if(CUDA_ENABLED)
    enable_language(CUDA)
endif()
set(CMAKE_CXX_STANDARD 17)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
add_subdirectory(chapter-2)
add_subdirectory(chapter-3)