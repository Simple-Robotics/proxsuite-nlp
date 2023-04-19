#
# Copyright (C) 2022 LAAS-CNRS, INRIA
#

add_project_dependency(Boost COMPONENTS unit_test_framework REQUIRED)
add_project_dependency(benchmark REQUIRED PKG_CONFIG_REQUIRES "benchmark >= 1.5.0")

# create an object library just for cnpy
add_library(cnpy OBJECT cnpy.cpp)
target_link_libraries(cnpy Eigen3::Eigen)
target_include_directories(cnpy PUBLIC $<INSTALL_INTERFACE:./>)
set_target_properties(cnpy PROPERTIES PUBLIC_HEADER cnpy.hpp)

install(
  TARGETS cnpy
  EXPORT ${TARGETS_EXPORT_NAME}
  OBJECTS DESTINATION lib/
  PUBLIC_HEADER DESTINATION include/${PROJECT_NAME}/tests)

macro(ADD_TEST_CFLAGS test_name flag)
  set_property(
    TARGET ${test_name}
    APPEND_STRING
    PROPERTY COMPILE_FLAGS " ${flag}")
endmacro()

function(get_cpp_test_name name dir out_var)
  string(REPLACE "${CMAKE_CURRENT_SOURCE_DIR}" "test-cpp" prefix ${dir})
  set(${out_var}
      "${prefix}-${name}"
      PARENT_SCOPE)
endfunction(get_cpp_test_name)

function(add_proxnlp_test name)
  get_cpp_test_name(${name} ${CMAKE_CURRENT_SOURCE_DIR} test_name)
  set(test_file ${name}.cpp)

  add_unit_test(${test_name} ${test_file})
  set_target_properties(${test_name} PROPERTIES LINKER_LANGUAGE CXX)
  target_include_directories(${test_name} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

  # define macros required by boost_test
  # see: https://www.boost.org/doc/libs/1_78_0/libs/test/doc/html/boost_test/usage_variants.html
  # define module name, replace '-' by '_'
  set(MODULE_NAME "${name}Test")
  string(REGEX REPLACE "-" "_" MODULE_NAME ${MODULE_NAME})

  add_test_cflags(${test_name} "-DBOOST_TEST_DYN_LINK")
  add_test_cflags(${test_name} "-DBOOST_TEST_MODULE=${MODULE_NAME}")

  target_link_libraries(${test_name} ${PROJECT_NAME})
  target_link_libraries(${test_name} Boost::unit_test_framework)
  target_link_libraries(${test_name} benchmark::benchmark)
  target_link_libraries(${test_name} cnpy)
endfunction(add_proxnlp_test)

# symlink test data file
function(symlink_data_file filename)
  # filename should be relative
  execute_process(COMMAND ${CMAKE_COMMAND} -E ${LINK} ${CMAKE_CURRENT_SOURCE_DIR}/${filename}
                          ${CMAKE_CURRENT_BINARY_DIR}/${filename})
endfunction()

add_proxnlp_test(block-chol)
add_proxnlp_test(tri-solve)
add_proxnlp_test(constraints)
add_proxnlp_test(costs)
add_proxnlp_test(finite-diff)
add_proxnlp_test(math)
add_proxnlp_test(functions)
add_proxnlp_test(manifolds)
add_proxnlp_test(solver)

add_proxnlp_test(cnpy-load)
symlink_data_file("npy_payload.npy")
symlink_data_file("npy_payload2.npy")

# PYTHON TESTS
if(BUILD_PYTHON_INTERFACE)
  add_subdirectory(python)
endif()
