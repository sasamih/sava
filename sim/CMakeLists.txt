# ISA header generated from arch/*.json (single source of truth).
set(SAVA_GEN_DIR ${CMAKE_CURRENT_BINARY_DIR}/generated)
set(SAVA_ISA_HEADER ${SAVA_GEN_DIR}/sava/isa_gen.h)
add_custom_command(
  OUTPUT ${SAVA_ISA_HEADER}
  COMMAND ${CMAKE_COMMAND} -E make_directory ${SAVA_GEN_DIR}/sava
  COMMAND Python3::Interpreter ${PROJECT_SOURCE_DIR}/tools/gen_isa_header.py
          --isa ${PROJECT_SOURCE_DIR}/arch/isa.json
          --config ${PROJECT_SOURCE_DIR}/arch/sava-v0.json
          -o ${SAVA_ISA_HEADER}
  DEPENDS ${PROJECT_SOURCE_DIR}/tools/gen_isa_header.py
          ${PROJECT_SOURCE_DIR}/arch/isa.json
          ${PROJECT_SOURCE_DIR}/arch/sava-v0.json
  COMMENT "Generating SAVA ISA header"
  VERBATIM)
add_custom_target(sava_isa_gen DEPENDS ${SAVA_ISA_HEADER})

add_library(sava_sim_objs OBJECT
  src/c_api.cpp
  src/device.cpp
  src/executable.cpp)
target_include_directories(sava_sim_objs
  PUBLIC include ${SAVA_GEN_DIR}
  PRIVATE src)
set_target_properties(sava_sim_objs PROPERTIES POSITION_INDEPENDENT_CODE ON)
add_dependencies(sava_sim_objs sava_isa_gen)

# Shared library with the C API: loaded by the Python runtime via ctypes
# (and later by an IREE HAL driver).
add_library(sava_sim SHARED)
target_link_libraries(sava_sim PUBLIC sava_sim_objs)

add_library(sava_sim_static STATIC)
target_link_libraries(sava_sim_static PUBLIC sava_sim_objs)

add_executable(sava-sim tools/sava_sim_main.cpp tools/npy.cpp)
target_link_libraries(sava-sim PRIVATE sava_sim_static)

if(SAVA_BUILD_TESTS)
  add_executable(sava_sim_tests tests/sim_tests.cpp)
  target_link_libraries(sava_sim_tests PRIVATE sava_sim_static)
  add_test(NAME sim_tests COMMAND sava_sim_tests)
endif()
