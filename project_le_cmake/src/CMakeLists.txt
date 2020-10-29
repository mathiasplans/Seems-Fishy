add_executable(seems_fishy
    main.cpp
)

# set_target_properties(seems_fishy PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# target_include_directories(seems_fishy SYSTEM PUBLIC ${CUDA_INCLUDE_DIRS})
target_link_libraries(seems_fishy PRIVATE project_options project_warnings)

find_package(Vulkan)

target_link_libraries_system(
  seems_fishy PRIVATE 
  Vulkan::Vulkan
  CONAN_PKG::glfw
  CONAN_PKG::glm
  CONAN_PKG::eigen

)

