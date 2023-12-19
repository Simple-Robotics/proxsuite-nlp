#
# Copyright (C) 2022 LAAS-CNRS, INRIA
#

# use underscore for Python to parse module name
set(PYLIB_NAME "pyproxsuite_nlp")
set(${PYLIB_NAME}_INSTALL_DIR ${PYTHON_SITELIB}/proxsuite_nlp)

include(${PROJECT_SOURCE_DIR}/cmake/python-helpers.cmake)
include(${PROJECT_SOURCE_DIR}/cmake/stubs.cmake)

# identify bindings C++ source files
file(GLOB_RECURSE ${PYLIB_NAME}_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)

file(GLOB_RECURSE ${PYLIB_NAME}_HEADERS ${CMAKE_CURRENT_SOURCE_DIR}/*.hpp)

set(PY_HEADERS ${${PYLIB_NAME}_HEADERS})
set(PY_SOURCES ${${PYLIB_NAME}_SOURCES})

add_header_group(PY_HEADERS)

function(MAKE_BINDINGS)
  add_library(${PYLIB_NAME} SHARED ${PY_SOURCES} ${PY_HEADERS})

  target_include_directories(${PYLIB_NAME} SYSTEM PRIVATE ${PYTHON_INCLUDE_DIRS})
  set_target_properties(
    ${PYLIB_NAME}
    PROPERTIES LIBRARY_OUTPUT_DIRECTORY "proxsuite_nlp"
               PREFIX ""
               SUFFIX ${PYTHON_EXT_SUFFIX}
               VERSION ${PROJECT_VERSION})
  if(UNIX)
    get_relative_rpath(${${PYLIB_NAME}_INSTALL_DIR} PYLIB_INSTALL_RPATH)
    set_target_properties(${PYLIB_NAME} PROPERTIES INSTALL_RPATH "${PYLIB_INSTALL_RPATH}")
  endif()

  target_compile_options(${PYLIB_NAME} PRIVATE -Wno-conversion -Wno-comment)
  target_compile_definitions(${PYLIB_NAME} PRIVATE MODULE_NAME=${PYLIB_NAME})

  # link solver, eigenpy, Boost.Python
  target_link_libraries(${PYLIB_NAME} PUBLIC ${PROJECT_NAME} eigenpy::eigenpy)
  install(
    TARGETS ${PYLIB_NAME}
    EXPORT ${TARGETS_EXPORT_NAME}
    DESTINATION ${${PYLIB_NAME}_INSTALL_DIR})
endfunction()

make_bindings()

file(MAKE_DIRECTORY ${PROJECT_BINARY_DIR}/include/${PROJECT_NAME}/python
     ${PROJECT_BINARY_DIR}/include/${PROJECT_NAME}/python/utils)

# copy headers to build/include/ directory
# TODO: fix include path
foreach(pyheader ${PY_HEADERS})
  string(REGEX REPLACE "${CMAKE_CURRENT_SOURCE_DIR}/" "" pyheader ${pyheader})
  get_filename_component(headerPath ${pyheader} PATH)
  execute_process(COMMAND ${CMAKE_COMMAND} -E ${LINK} ${CMAKE_CURRENT_SOURCE_DIR}/${pyheader}
                          ${PROJECT_BINARY_DIR}/include/${PROJECT_NAME}/python/${pyheader})
  install(
    FILES ${pyheader}
    DESTINATION ${CMAKE_INSTALL_PREFIX}/include/proxsuite-nlp/python/${headerPath}
    PERMISSIONS OWNER_READ GROUP_READ WORLD_READ OWNER_WRITE)
endforeach(pyheader)

# COPY PYTHON FILES INTO OUTPUT FOLDER
set(${PYLIB_NAME}_PYFILES __init__.py casadi_utils.py utils.py)

foreach(pyfile ${${PYLIB_NAME}_PYFILES})
  python_install_on_site(proxsuite_nlp ${pyfile})
endforeach()

# Generate stubs
if(GENERATE_PYTHON_STUBS)
  load_stubgen()
  generate_stubs(${CMAKE_CURRENT_BINARY_DIR} proxsuite_nlp ${PYTHON_SITELIB} ${PYLIB_NAME})
endif(GENERATE_PYTHON_STUBS)