#
# Copyright (C) 2022 LAAS-CNRS, INRIA
#

set(EXAMPLES_HEADER
    "example-base.hpp")

file(GLOB EXAMPLES_CPP
     "*.cpp")


function(add_proxnlp_example exfile)
  get_filename_component(example_name ${exfile} NAME_WE)
  set(example_name "example-${example_name}")
  get_filename_component(exfile ${exfile} NAME)
  message(STATUS "Adding executable ${example_name} (file ${exfile})")

  ADD_UNIT_TEST(${example_name} ${exfile})
  set_target_properties(${example_name} PROPERTIES LINKER_LANGUAGE CXX)
  target_include_directories(${example_name} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

  target_link_libraries(${example_name} PUBLIC ${PROJECT_NAME})
endfunction()

foreach(exfile ${EXAMPLES_CPP})
  add_proxnlp_example(${exfile})
endforeach()


# Python examples
if(BUILD_PYTHON_INTERFACE)

  file(GLOB_RECURSE EXAMPLES_PY
       ${CMAKE_CURRENT_SOURCE_DIR}/*.py)

  foreach(pyfile ${EXAMPLES_PY})
    get_filename_component(pyfile_name ${pyfile} NAME)
     execute_process(
          COMMAND ${CMAKE_COMMAND} -E create_symlink
          ${pyfile}
          ${CMAKE_CURRENT_BINARY_DIR}/${pyfile_name}
     )
  endforeach()

endif(BUILD_PYTHON_INTERFACE)
