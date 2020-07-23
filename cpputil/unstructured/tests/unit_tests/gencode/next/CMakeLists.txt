function(add_gtnext_atlas_test)
    set(options)
    set(oneValueArgs NAME)
    set(multiValueArgs SOURCES)
    cmake_parse_arguments(ARG "${options}" "${oneValueArgs}"
                          "${multiValueArgs}" ${ARGN})


    add_executable(${ARG_NAME} ${ARG_SOURCES})
    target_link_libraries(${ARG_NAME} gencode)
    target_link_libraries(${ARG_NAME} test_util)
    target_link_libraries(${ARG_NAME} GridTools::gridtools atlas eckit gtest gtest_main)

    add_test(NAME ${ARG_NAME} COMMAND $<TARGET_FILE:${ARG_NAME}>)
endfunction()

add_gtnext_atlas_test(NAME test_atlas_array_view_adapter SOURCES test_atlas_array_view_adapter.cc)
add_gtnext_atlas_test(NAME test_atlas_field_util SOURCES test_atlas_field_util.cc)
