cmake_minimum_required(VERSION 3.12.4)

project(fvm_nabla LANGUAGES CXX)

find_package(atlas REQUIRED)
find_package(Python3 COMPONENTS Interpreter REQUIRED)

add_custom_command(OUTPUT fvm_nabla.hpp
                   COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/fvm_nabla.py > ${CMAKE_CURRENT_BINARY_DIR}/fvm_nabla.hpp
                   DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/fvm_nabla.py
                   COMMENT "Generating ${CMAKE_CURRENT_BINARY_DIR}/fvm_nabla.hpp"
)

add_executable(fvm_nabla fvm_nabla_atlas_driver.cpp fvm_nabla.hpp)
target_link_libraries(fvm_nabla PRIVATE atlas eckit)
target_include_directories(fvm_nabla PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/dawn)
target_include_directories(fvm_nabla PRIVATE ${CMAKE_CURRENT_BINARY_DIR})
target_compile_features(fvm_nabla PRIVATE cxx_std_17)
