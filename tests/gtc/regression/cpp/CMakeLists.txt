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


function(add_regression_test_executable)
    set(options)
    set(oneValueArgs NAME)
    set(multiValueArgs SOURCES LIBRARIES)
    cmake_parse_arguments(ARG "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

    add_executable(${ARG_NAME} ${ARG_SOURCES})
    target_link_libraries(${ARG_NAME} ${ARG_LIBRARIES})
endfunction()

function(add_regression_test name)
    set(_sources ${name}_driver.cc)
    set(default_libraries GridTools::gridtools gtest gtest_main gmock gtnext)

    add_regression_test_executable(NAME ${name}_unaive SOURCES ${_sources} LIBRARIES ${default_libraries})

    add_regression_test_executable(NAME ${name}_ugpu SOURCES ${_sources} LIBRARIES ${default_libraries} GridTools::stencil_gpu)
    gridtools_setup_target(${name}_ugpu CUDA_ARCH sm_50)
endfunction()

include(CTest)
if(BUILD_TESTING)
    include(FetchGoogleTest.cmake)
    fetch_googletest()

    add_regression_test(cell2cell)
    add_regression_test(vertex2edge)
endif()
