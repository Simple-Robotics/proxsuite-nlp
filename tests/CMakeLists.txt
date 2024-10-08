#
# Copyright (C) 2022-2023 LAAS-CNRS, INRIA
#

add_project_dependency(Boost COMPONENTS unit_test_framework REQUIRED)

find_path(
  EigenRand_INCLUDE_DIR
  NAMES EigenRand
  PATHS ${EigenRand_PREFIX}
  PATH_SUFFIXES include/EigenRand
)

if(NOT EigenRand_INCLUDE_DIR)
  # Fetch private dependencies
  include(FetchContent)

  # Add eigenrand dependency
  # TODO use main bab2min/EigenRand when PR is merged
  FetchContent_Declare(
    eigenrand
    GIT_REPOSITORY https://github.com/ManifoldFR/EigenRand.git
    GIT_TAG 23d287e
    GIT_PROGRESS TRUE
    EXCLUDE_FROM_ALL SYSTEM
  )
  set(EIGENRAND_BUILD_TEST
      OFF
      CACHE INTERNAL "Enable EigenRand test"
  )
  set(EIGENRAND_BUILD_BENCHMARK
      OFF
      CACHE INTERNAL "Enable EigenRand benchmark"
  )
  FetchContent_MakeAvailable(eigenrand)
endif()

# Add benchmark dependency
if(BUILD_BENCHMARK)
  find_package(benchmark REQUIRED)
endif()

# create an object library just for cnpy
set(CNPY_LIB ${PROJECT_NAME}-cnpy)
add_library(${CNPY_LIB} STATIC cnpy.cpp)
target_link_libraries(${CNPY_LIB} PUBLIC Eigen3::Eigen)
target_include_directories(${CNPY_LIB} PUBLIC $<INSTALL_INTERFACE:./>)
set_target_properties(${CNPY_LIB} PROPERTIES POSITION_INDEPENDENT_CODE ON PUBLIC_HEADER cnpy.hpp)
set_standard_output_directory(${CNPY_LIB})

install(
  TARGETS ${CNPY_LIB}
  EXPORT ${TARGETS_EXPORT_NAME}
  ARCHIVE DESTINATION lib/${PROJECT_NAME}
  PUBLIC_HEADER DESTINATION include/${PROJECT_NAME}/tests
)

set(TEST_UTIL_LIB ${PROJECT_NAME}-test_util)
add_library(${TEST_UTIL_LIB} STATIC util.cpp)
target_link_libraries(${TEST_UTIL_LIB} PUBLIC ${PROJECT_NAME})
modernize_target_link_libraries(
  ${TEST_UTIL_LIB}
  SCOPE
  PUBLIC
  TARGETS
  EigenRand::EigenRand
  INCLUDE_DIRS
  ${EigenRand_INCLUDE_DIR}
)
set_target_properties(${TEST_UTIL_LIB} PROPERTIES LINKER_LANGUAGE CXX)

macro(ADD_TEST_CFLAGS test_name flag)
  set_property(
    TARGET ${test_name}
    APPEND_STRING
    PROPERTY COMPILE_FLAGS " ${flag}"
  )
endmacro()

function(get_cpp_test_name name dir out_var)
  string(REPLACE "${CMAKE_CURRENT_SOURCE_DIR}" "test-cpp" prefix ${dir})
  set(${out_var}
      "${PROJECT_NAME}-${prefix}-${name}"
      PARENT_SCOPE
  )
endfunction()

function(add_proxsuite_nlp_test name)
  get_cpp_test_name(${name} ${CMAKE_CURRENT_SOURCE_DIR} test_name)
  set(test_file ${name}.cpp)

  add_unit_test(${test_name} ${test_file})
  set_standard_output_directory(${test_name})
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
  target_link_libraries(${test_name} PRIVATE ${CNPY_LIB})
  target_link_libraries(${test_name} PRIVATE ${TEST_UTIL_LIB})
endfunction()

function(get_cpp_bench_name name dir out_var)
  string(REPLACE "${CMAKE_CURRENT_SOURCE_DIR}" "bench-cpp" prefix ${dir})
  set(${out_var}
      "${prefix}-${name}"
      PARENT_SCOPE
  )
endfunction()

function(add_proxsuite_nlp_bench name)
  if(BUILD_BENCHMARK)
    get_cpp_bench_name(${name} ${CMAKE_CURRENT_SOURCE_DIR} bench_name)
    set(bench_file ${name}.cpp)

    add_executable(${bench_name} ${bench_file})
    set_standard_output_directory(${bench_name})
    set_target_properties(${bench_name} PROPERTIES LINKER_LANGUAGE CXX)
    target_include_directories(${bench_name} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

    target_link_libraries(${bench_name} PRIVATE ${PROJECT_NAME})
    target_link_libraries(${bench_name} PRIVATE benchmark::benchmark)
    target_link_libraries(${bench_name} PRIVATE ${TEST_UTIL_LIB})
  endif()
endfunction()

if(WIN32)
  set(LINK copy_if_different)
else(WIN32)
  set(LINK create_symlink)
endif(WIN32)

# symlink test data file
function(symlink_data_file filename)
  # filename should be relative
  execute_process(
    COMMAND ${CMAKE_COMMAND} -E ${LINK} ${CMAKE_CURRENT_SOURCE_DIR}/${filename}
            ${CMAKE_CURRENT_BINARY_DIR}/${filename}
  )
endfunction()

add_proxsuite_nlp_bench(cholesky-block-sparse-bench)
add_proxsuite_nlp_bench(cholesky-dense-bench)
add_proxsuite_nlp_bench(tri-solve-bench)

add_proxsuite_nlp_test(cholesky-block-sparse)
add_proxsuite_nlp_test(tri-solve)
add_proxsuite_nlp_test(constraints)
add_proxsuite_nlp_test(costs)
add_proxsuite_nlp_test(finite-diff)
add_proxsuite_nlp_test(math)
add_proxsuite_nlp_test(functions)
add_proxsuite_nlp_test(linesearch)
add_proxsuite_nlp_test(manifolds)
add_proxsuite_nlp_test(solver)

add_proxsuite_nlp_test(cnpy-load)
symlink_data_file("npy_payload.npy")
symlink_data_file("npy_payload2.npy")

# PYTHON TESTS
if(BUILD_PYTHON_INTERFACE)
  add_subdirectory(python)
endif()
