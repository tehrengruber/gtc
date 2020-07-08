add_executable(fvm_nabla driver.cc)
target_link_libraries(fvm_nabla PRIVATE atlas eckit GridTools::gridtools gencode gtest gtest_main)
add_test(NAME fvm_nabla COMMAND $<TARGET_FILE:fvm_nabla>)
