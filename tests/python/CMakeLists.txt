set(PYTHON_TESTS test_1d_funs.py test_manifolds.py test_functions.py test_print.py)

message(STATUS "Python tests: ${PYTHON_TESTS}")

foreach(pyfile ${PYTHON_TESTS})
  get_filename_component(test_name ${pyfile} NAME_WE)
  string(REGEX REPLACE "^test_" "test-py-" test_name ${test_name})
  string(REGEX REPLACE "_" "-" test_name ${test_name})
  # We must add tests/python because in the future we might create extension modules
  # specifically for some tests
  add_python_unit_test(${test_name} "tests/python/${pyfile}" "tests/python" "bindings/python")
endforeach()
