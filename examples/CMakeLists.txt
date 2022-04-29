# Copyright (C) 2022 LAAS-CNRS, INRIA

set(EXAMPLES_HEADER
    "example-base.hpp")

file(GLOB EXAMPLES_CPP
     "*.cpp")

message(STATUS "CPP EXAMPLES: ${EXAMPLES_CPP}")


function(add_proxnlp_example exfile)
  get_filename_component(example_name ${exfile} NAME_WE)
  set(example_name "example-${example_name}")
  get_filename_component(exfile ${exfile} NAME)
  message(STATUS "Adding executable ${example_name} (file ${exfile})")

  ADD_UNIT_TEST(${example_name} ${exfile})
  set_target_properties(${example_name} PROPERTIES LINKER_LANGUAGE CXX)
  target_include_directories(${example_name} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

  target_link_libraries(${example_name} PUBLIC ${PROJECT_NAME})
  target_link_libraries(${example_name} PRIVATE fmt::fmt)
endfunction()

foreach(exfile ${EXAMPLES_CPP})
  add_proxnlp_example(${exfile})
endforeach()


# Python examples
if(BUILD_PYTHON_INTERFACE)

  file(MAKE_DIRECTORY
       ${CMAKE_CURRENT_BINARY_DIR}/python)

  file(GLOB_RECURSE PY_EXAMPLES
       ${CMAKE_CURRENT_SOURCE_DIR}/python/*.py)

  message(STATUS "Python examples: ${PY_EXAMPLES}")
  foreach(pyfile ${PY_EXAMPLES})
  	get_filename_component(pyfile_name ${pyfile} NAME)
       execute_process(
            COMMAND ${CMAKE_COMMAND} -E create_symlink
            ${pyfile}
            ${CMAKE_CURRENT_BINARY_DIR}/python/${pyfile_name}
       )
  endforeach()

  install(FILES ${PY_EXAMPLES}
          DESTINATION ${CMAKE_INSTALL_PREFIX}/python
          PERMISSIONS OWNER_READ GROUP_READ WORLD_READ OWNER_WRITE)
endif(BUILD_PYTHON_INTERFACE)