# Copyright (C) 2022 LAAS-CNRS / INRIA

set(TEST_DIR ${CMAKE_CURRENT_SOURCE_DIR})


ADD_PROJECT_DEPENDENCY(Boost COMPONENTS unit_test_framework REQUIRED)


macro(ADD_TEST_CFLAGS test_name flag)
  set_property(TARGET ${test_name} APPEND_STRING PROPERTY COMPILE_FLAGS " ${flag}")
endmacro()


function(get_cpp_test_name name dir out_var)
  string(REPLACE "${TEST_DIR}" "" prefix ${dir})
  set(${out_var} "test-${prefix}${name}" PARENT_SCOPE)
endfunction(get_cpp_test_name)


macro(add_lienlp_test name)

  get_cpp_test_name(${name} ${CMAKE_CURRENT_SOURCE_DIR} test_name)
  ADD_UNIT_TEST(${test_name} ${name}.cpp)
  set_target_properties(${test_name} PROPERTIES LINKER_LANGUAGE CXX)
  target_include_directories(${test_name} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

  ADD_TEST_CFLAGS(${test_name} "-DBOOST_TEST_DYN_LINK")

  # define module name, replace '-' by '_'
  set(MODULE_NAME "${name}Test")
  string(REGEX REPLACE "-" "_" MODULE_NAME ${MODULE_NAME})

  ADD_TEST_CFLAGS(${test_name} "-DBOOST_TEST_MODULE=${MODULE_NAME}")

  target_link_libraries(${test_name} PUBLIC ${PROJECT_NAME})
  target_link_libraries(${test_name} PRIVATE Boost::unit_test_framework)

  unset(test_name)
endmacro(add_lienlp_test)


add_lienlp_test(manifold)
add_lienlp_test(merit)

