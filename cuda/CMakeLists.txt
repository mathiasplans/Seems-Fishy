cmake_minimum_required(VERSION 3.12)

project(sfml_cuda_project CXX CUDA)
include(cmake/StandardProjectSettings.cmake)

find_package(CUDA REQUIRED)

include(CheckLanguage)
check_language(CUDA)
SET(CUDA_SEPARABLE_COMPILATION ON)
cuda_select_nvcc_arch_flags(ARCH_FLAGS)

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

add_subdirectory(src)
