cmake_minimum_required(VERSION 3.14.5)
project(unstructured_interface LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)

include(FetchContent)
FetchContent_Declare(GridTools
    GIT_REPOSITORY https://github.com/GridTools/gridtools.git
    GIT_TAG        master
    # URL https://github.com/GridTools/gridtools/archive/???.tar.gz
)
FetchContent_MakeAvailable(GridTools)

add_executable(prototype prototype.cc)
target_link_libraries(prototype GridTools::gridtools)
target_include_directories(prototype PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/gencode/include)

find_package(eckit)
find_package(Atlas)

add_executable(atlas_mesh_adapter atlas_mesh_adapter.cc)
target_link_libraries(atlas_mesh_adapter atlas GridTools::gridtools)
target_include_directories(atlas_mesh_adapter PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/gencode/include)
