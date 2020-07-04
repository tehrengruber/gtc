add_executable(fvm_nabla driver.cc)
target_link_libraries(fvm_nabla PRIVATE atlas eckit)
target_include_directories(fvm_nabla PRIVATE ${PROJECT_SOURCE_DIR}/gencode/include)
