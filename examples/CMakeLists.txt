#
# Copyright (C) 2022 LAAS-CNRS, INRIA
#

set(EXAMPLES_HEADER "example-base.hpp")

function(add_proxsuite_nlp_example exfile)
  get_filename_component(example_name ${exfile} NAME_WE)
  set(example_name "${PROJECT_NAME}-example-${example_name}")
  get_filename_component(exfile ${exfile} NAME)
  message(STATUS "Adding executable ${example_name} (file ${exfile})")

  add_unit_test(${example_name} ${exfile})
  set_standard_output_directory(${example_name})
  set_target_properties(${example_name} PROPERTIES LINKER_LANGUAGE CXX)
  target_include_directories(${example_name} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

  target_link_libraries(${example_name} PUBLIC ${PROJECT_NAME})
endfunction()

add_proxsuite_nlp_example(circle.cpp)
add_proxsuite_nlp_example(equality-qp.cpp)
if(BUILD_WITH_PINOCCHIO_SUPPORT)
  add_project_private_dependency(example-robot-data REQUIRED)
  add_proxsuite_nlp_example(so2.cpp)
  add_proxsuite_nlp_example(ur5-ik.cpp)
  target_link_libraries(
    ${PROJECT_NAME}-example-ur5-ik PUBLIC example-robot-data::example-robot-data
  )
endif()

# Python examples
if(BUILD_PYTHON_INTERFACE)

  file(GLOB_RECURSE EXAMPLES_PY ${CMAKE_CURRENT_SOURCE_DIR}/*.py)

  foreach(pyfile ${EXAMPLES_PY})
    get_filename_component(pyfile_name ${pyfile} NAME)
    execute_process(
      COMMAND ${CMAKE_COMMAND} -E create_symlink ${pyfile}
              ${CMAKE_CURRENT_BINARY_DIR}/${pyfile_name}
    )
  endforeach()

endif()
