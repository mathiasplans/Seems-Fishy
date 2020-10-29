cmake_minimum_required(VERSION 3.12)

project(seems_fishy CXX CUDA)
include(cmake/StandardProjectSettings.cmake)

# find_package(CUDA REQUIRED)

# include(CheckLanguage)
# check_language(CUDA)
# SET(CUDA_SEPARABLE_COMPILATION ON)
# cuda_select_nvcc_arch_flags(ARCH_FLAGS)

# Link this 'library' to set the c++ standard / compile-time options requested
add_library(project_options INTERFACE)
target_compile_features(project_options INTERFACE cxx_std_20)

# Link this 'library' to use the warnings specified in CompilerWarnings.cmake
add_library(project_warnings INTERFACE)

# standard compiler warnings
include(cmake/CompilerWarnings.cmake)
set_project_warnings(project_warnings)

set(CONAN_EXTRA_REQUIRES ${CONAN_EXTRA_REQUIRES})

include(cmake/Conan.cmake)
run_conan()

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

add_executable(
    seems_fishy
    src/main.cpp
)

# set_target_properties(seems_fishy PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# target_include_directories(seems_fishy SYSTEM PUBLIC ${CUDA_INCLUDE_DIRS})
target_link_libraries(seems_fishy PRIVATE project_options project_warnings)

find_package(Vulkan REQUIRED FATAL_ERROR)

target_link_libraries_system(
  seems_fishy PRIVATE
  Vulkan::Vulkan
  CONAN_PKG::glfw
  CONAN_PKG::glm
  CONAN_PKG::eigen
)
