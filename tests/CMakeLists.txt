#
# Copyright (C) 2022-2023 LAAS-CNRS, INRIA
#

add_project_dependency(Boost COMPONENTS unit_test_framework REQUIRED)
add_project_dependency(benchmark REQUIRED PKG_CONFIG_REQUIRES "benchmark >= 1.5.0")

include(FetchContent)
# TODO use main bab2min/EigenRand when PR is merged
FetchContent_Declare(EigenRand
  GIT_REPOSITORY https://github.com/ManifoldFR/EigenRand.git
  GIT_TAG 23d287e
  )

set(EIGENRAND_BUILD_TEST OFF CACHE INTERNAL "Enable EigenRand test")
set(EIGENRAND_BUILD_BENCHMARK OFF CACHE INTERNAL "Enable EigenRand benchmark")
FetchContent_MakeAvailable(EigenRand)

# create an object library just for cnpy
add_library(cnpy STATIC cnpy.cpp)
target_link_libraries(cnpy PUBLIC Eigen3::Eigen)
target_include_directories(cnpy PUBLIC $<INSTALL_INTERFACE:./>)
set_target_properties(cnpy PROPERTIES POSITION_INDEPENDENT_CODE ON PUBLIC_HEADER cnpy.hpp)

install(
  TARGETS cnpy
  EXPORT ${TARGETS_EXPORT_NAME}
  ARCHIVE DESTINATION lib/${PROJECT_NAME}
  PUBLIC_HEADER DESTINATION include/${PROJECT_NAME}/tests)

set(TEST_UTIL_LIB ${PROJECT_NAME}_test_util)
add_library(${TEST_UTIL_LIB} STATIC util.cpp)
target_link_libraries(${TEST_UTIL_LIB} PUBLIC ${PROJECT_NAME} EigenRand::EigenRand)
target_include_directories(${TEST_UTIL_LIB} PRIVATE ./)
set_target_properties(${TEST_UTIL_LIB} PROPERTIES LINKER_LANGUAGE CXX)

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
endfunction()

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

  target_link_libraries(${test_name} PRIVATE ${PROJECT_NAME})
  target_link_libraries(${test_name} PRIVATE Boost::unit_test_framework)
  target_link_libraries(${test_name} PRIVATE cnpy)
  target_link_libraries(${test_name} PRIVATE ${TEST_UTIL_LIB})
endfunction()

function(get_cpp_bench_name name dir out_var)
  string(REPLACE "${CMAKE_CURRENT_SOURCE_DIR}" "bench-cpp" prefix ${dir})
  set(${out_var}
      "${prefix}-${name}"
      PARENT_SCOPE)
endfunction()

function(add_proxnlp_bench name)
  if(BUILD_BENCHMARK)
    get_cpp_bench_name(${name} ${CMAKE_CURRENT_SOURCE_DIR} bench_name)
    set(bench_file ${name}.cpp)

    add_unit_test(${bench_name} ${bench_file})
    set_target_properties(${bench_name} PROPERTIES LINKER_LANGUAGE CXX)
    target_include_directories(${bench_name} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

    target_link_libraries(${bench_name} PRIVATE ${PROJECT_NAME})
    target_link_libraries(${bench_name} PRIVATE benchmark::benchmark)
    target_link_libraries(${bench_name} PRIVATE ${bench_UTIL_LIB})
  endif()
endfunction()

# symlink test data file
function(symlink_data_file filename)
  # filename should be relative
  execute_process(COMMAND ${CMAKE_COMMAND} -E ${LINK} ${CMAKE_CURRENT_SOURCE_DIR}/${filename}
                          ${CMAKE_CURRENT_BINARY_DIR}/${filename})
endfunction()

add_proxnlp_bench(cholesky-sparse-bench)
add_proxnlp_bench(cholesky-dense-bench)
add_proxnlp_bench(tri-solve-bench)

add_proxnlp_test(cholesky-sparse)
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
