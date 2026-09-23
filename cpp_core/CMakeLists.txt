cmake_minimum_required(VERSION 3.20)
project(graph_core LANGUAGES CXX)

add_executable(graph_core src/main.cpp)
target_compile_features(graph_core PRIVATE cxx_std_17)
target_include_directories(graph_core PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/third_party)

if(MSVC)
  target_compile_options(graph_core PRIVATE /W4 /permissive- /EHsc)
else()
  target_compile_options(graph_core PRIVATE -Wall -Wextra -Wpedantic)
endif()

set_target_properties(graph_core PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
  RUNTIME_OUTPUT_DIRECTORY_RELEASE ${CMAKE_BINARY_DIR}/bin
  RUNTIME_OUTPUT_DIRECTORY_DEBUG ${CMAKE_BINARY_DIR}/bin)
