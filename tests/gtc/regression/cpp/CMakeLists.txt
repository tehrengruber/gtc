cmake_minimum_required(VERSION 3.14.5)
project(gtc_regression LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)

add_compile_options(-Wall)

include(CheckLanguage)
check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
endif()

include(FetchContent)
FetchContent_Declare(GridTools
    GIT_REPOSITORY https://github.com/GridTools/gridtools.git
    GIT_TAG        master
    # URL https://github.com/GridTools/gridtools/archive/???.tar.gz
)
FetchContent_MakeAvailable(GridTools)

FetchContent_Declare(cpputil_unstructured
    SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../../../../cpputil/unstructured
)
FetchContent_MakeAvailable(cpputil_unstructured)


function(add_regression_test name)
    add_executable(${name} ${name}_driver.cu)
    # target_link_libraries(cell2cell PRIVATE
    # target_link_libraries(${ARG_NAME} gencode)
    # target_link_libraries(${ARG_NAME} test_util)
    target_link_libraries(${name} GridTools::gridtools gtest gtest_main gmock gtnext)
    target_link_libraries(${name} GridTools::stencil_gpu)
    gridtools_setup_target(${name} CUDA_ARCH sm_50)
endfunction()

include(CTest)
if(BUILD_TESTING)
    include(FetchGoogleTest.cmake)
    fetch_googletest()

    add_regression_test(cell2cell)
    add_regression_test(vertex2edge)
endif()
