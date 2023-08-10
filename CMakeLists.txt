cmake_minimum_required(VERSION 3.16)
project(cuda_layer LANGUAGES CUDA CXX)
set(CMAKE_CXX_STANDARD 17)
include_directories(./include/layer)

set(PROJECT_ROOT ${CMAKE_CURRENT_SOURCE_DIR})
set(CMAKE_CUDA_ARCHITECTURES 75)

find_package(glog REQUIRED)

include(CheckLanguage)
check_language(CUDA)


aux_source_directory(./source/layer DIR_LAYER)

message("DIR_LAYER" ${DIR_LAYER})

set(link_lib glog::glog)
IF (!WIN32)
    set(link_lib ${link_lib} pthread)
ENDIF ()


set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)
add_library(cuda_layer ${DIR_LAYER})

target_link_libraries(cuda_layer ${link_lib})

target_include_directories(cuda_layer PUBLIC ${glog_INCLUDE_DIR})


add_subdirectory(test)
