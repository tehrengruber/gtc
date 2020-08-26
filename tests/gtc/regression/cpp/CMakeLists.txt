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
    set(oneValueArgs NAME INPUT MODE)
    set(multiValueArgs SOURCES LIBRARIES)
    cmake_parse_arguments(ARG "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

    message(STATUS "${ARG_NAME}")
    set(generated_stencil_source ${CMAKE_CURRENT_SOURCE_DIR}/generated_${ARG_NAME}.hpp)
    add_custom_command(OUTPUT ${generated_stencil_source}
        COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/${ARG_INPUT} ${ARG_MODE}
        DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/${ARG_INPUT})
    add_custom_target(generated_${ARG_NAME} DEPENDS ${generated_stencil_source})
    add_executable(${ARG_NAME} ${ARG_SOURCES}) # TODO fix GridTools appending .cu to .hpp, then we don't need the custom_target
    target_compile_definitions(${ARG_NAME} PRIVATE STENCIL_IMPL_SOURCE="${generated_stencil_source}")
    target_link_libraries(${ARG_NAME} ${ARG_LIBRARIES})
    add_dependencies(${ARG_NAME} generated_${ARG_NAME})
    add_test(NAME ${ARG_NAME} COMMAND $<TARGET_FILE:${ARG_NAME}>)
endfunction()

function(add_regression_test name)
    set(options)
    set(oneValueArgs)
    set(multiValueArgs ADDITIONAL_LIBRARIES)
    cmake_parse_arguments(PARSE_ARGV 1 ARG "${options}" "${oneValueArgs}" "${multiValueArgs}")

    set(_sources ${name}_driver.cc)
    set(libraries GridTools::gridtools gtest gtest_main gmock gtnext)
    list (APPEND libraries ${ARG_ADDITIONAL_LIBRARIES})

    add_regression_test_executable(NAME ${name}_unaive INPUT ${name}.py MODE unaive SOURCES ${_sources} LIBRARIES ${libraries})

    if(TARGET GridTools::stencil_gpu)
        add_regression_test_executable(NAME ${name}_ugpu INPUT ${name}.py MODE ugpu SOURCES ${_sources} LIBRARIES ${libraries} GridTools::stencil_gpu)
        gridtools_setup_target(${name}_ugpu CUDA_ARCH sm_50)
    endif()
endfunction()

include(CTest)
if(BUILD_TESTING)
    find_package(Python3 COMPONENTS Interpreter REQUIRED)

    include(FetchGoogleTest.cmake)
    fetch_googletest()

    find_package(eckit REQUIRED)
    find_package(Atlas REQUIRED)

    add_regression_test(cell2cell)
    add_regression_test(vertex2edge)
    add_regression_test(fvm_nabla ADDITIONAL_LIBRARIES atlas eckit)
endif()
