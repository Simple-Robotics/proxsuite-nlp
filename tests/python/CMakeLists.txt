function(CHECK_FOR_PYTHON_PACKAGE package_name)
  execute_process(
    COMMAND ${PYTHON_EXECUTABLE} -c "import ${package_name}"
    RESULT_VARIABLE _pypkg_found
    OUTPUT_QUIET ERROR_QUIET
  )
  if(_pypkg_found EQUAL 0)
    set(${package_name}_FOUND
        TRUE
        PARENT_SCOPE
    )
    message(STATUS "Found Python package ${package_name}.")
  else()
    set(${package_name}_FOUND
        FALSE
        PARENT_SCOPE
    )
    message(STATUS "Python package ${package_name} not found.")
  endif()
endfunction()

# Check for pytest
check_for_python_package(pytest)
if(NOT pytest_FOUND)
  message(FATAL_ERROR "Python pytest package missing. You can install it with: pip install pytest")
endif()

set(PYTHON_TESTS test_1d_funs.py test_costs.py test_functions.py test_manifolds.py
                 test_polymorphic.py test_print.py
)

message(STATUS "Python tests: ${PYTHON_TESTS}")

foreach(pyfile ${PYTHON_TESTS})
  get_filename_component(test_name ${pyfile} NAME_WE)
  string(REGEX REPLACE "^test_" "${PROJECT_NAME}-test-py-" test_name ${test_name})
  string(REGEX REPLACE "_" "-" test_name ${test_name})
  # We must add tests/python because in the future we might create extension modules
  # specifically for some tests
  add_python_unit_test(${test_name} "tests/python/${pyfile}" "tests/python" "bindings/python")
endforeach()

include(${JRL_CMAKE_MODULES}/python-helpers.cmake)

function(add_test_lib filename name)
  python3_add_library(${name} MODULE WITH_SOABI ${filename})
  target_link_libraries(${name} PRIVATE ${PROJECT_NAME} eigenpy::eigenpy)
  target_include_directories(${name} PRIVATE ../../bindings/python/include)
  target_compile_definitions(${name} PRIVATE MODULE_NAME=${name})
endfunction()

add_test_lib(polymorphic_test.cpp test_polymorphic_ext)
