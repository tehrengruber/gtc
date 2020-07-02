cmake_minimum_required(VERSION 3.14.5)
project(unstructured_interface LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)

include(FetchContent)
FetchContent_Declare(GridTools
    URL https://github.com/GridTools/gridtools/archive/v2.0.0rc1.tar.gz
)
FetchContent_MakeAvailable(GridTools)

add_executable(prototype prototype.cc)
target_link_libraries(prototype GridTools::gridtools)
target_include_directories(prototype PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/gencode/include)
