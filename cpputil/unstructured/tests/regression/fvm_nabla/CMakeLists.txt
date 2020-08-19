add_executable(fvm_nabla driver.cc)
target_link_libraries(fvm_nabla PRIVATE atlas eckit GridTools::gridtools gtnext gtest gtest_main)
add_test(NAME fvm_nabla COMMAND $<TARGET_FILE:fvm_nabla>)

if(CMAKE_CUDA_COMPILER)
    add_executable(fvm_nabla_cuda driver.cc)
    target_link_libraries(fvm_nabla_cuda PRIVATE atlas eckit GridTools::gridtools GridTools::stencil_gpu gtnext gtest gtest_main)
    gridtools_setup_target(fvm_nabla_cuda CUDA_ARCH sm_50)
    target_compile_options(fvm_nabla_cuda PRIVATE "-std=c++17")

    add_test(NAME fvm_nabla_cuda COMMAND $<TARGET_FILE:fvm_nabla_cuda>)
endif()
