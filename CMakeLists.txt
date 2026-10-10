cmake_minimum_required(VERSION 3.20)
project(sava VERSION 0.1 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(SAVA_BUILD_TESTS "Build and register tests" ON)

find_package(Python3 REQUIRED COMPONENTS Interpreter)

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  add_compile_options(-Wall -Wextra -Wno-unused-parameter)
endif()

if(SAVA_BUILD_TESTS)
  enable_testing()
endif()

add_subdirectory(sim)

if(SAVA_BUILD_TESTS)
  # Python tests (assembler, runtime, compiler, end-to-end) run against the
  # simulator library built here.
  add_test(NAME python_tests
           COMMAND Python3::Interpreter -m pytest -q ${PROJECT_SOURCE_DIR}/python/tests)
  set_tests_properties(python_tests PROPERTIES
    ENVIRONMENT "SAVA_SIM_LIB=$<TARGET_FILE:sava_sim>;PYTHONPATH=${PROJECT_SOURCE_DIR}/python")
endif()
