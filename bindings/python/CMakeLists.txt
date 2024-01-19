#
# Copyright (C) 2022 LAAS-CNRS, INRIA
#

# use underscore for Python to parse module name
set(PYLIB_NAME "pyproxsuite_nlp")
set(${PYLIB_NAME}_INSTALL_DIR ${PYTHON_SITELIB}/proxsuite_nlp)

include(${PROJECT_SOURCE_DIR}/cmake/python-helpers.cmake)
include(${PROJECT_SOURCE_DIR}/cmake/stubs.cmake)

# identify bindings C++ source files
file(GLOB_RECURSE PY_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)
file(GLOB_RECURSE PY_HEADERS ${CMAKE_CURRENT_SOURCE_DIR}/*.hpp)

add_header_group(PY_HEADERS)

function(MAKE_BINDINGS)
  add_library(${PYLIB_NAME} SHARED ${PY_SOURCES} ${PY_HEADERS})
  add_library(proxsuite::nlp::python ALIAS ${PYLIB_NAME})

  set_target_properties(
    ${PYLIB_NAME}
    PROPERTIES LIBRARY_OUTPUT_DIRECTORY "proxsuite_nlp"
               RUNTIME_OUTPUT_DIRECTORY "proxsuite_nlp"
               PREFIX ""
               SUFFIX ${PYTHON_EXT_SUFFIX}
               VERSION ${PROJECT_VERSION})
  if(UNIX AND NOT APPLE)
    get_relative_rpath(${${PYLIB_NAME}_INSTALL_DIR} PYLIB_INSTALL_RPATH)
    set_target_properties(${PYLIB_NAME} PROPERTIES INSTALL_RPATH "${PYLIB_INSTALL_RPATH}")
  endif()

  set(GCC_CMD_OPTIONS -Wno-conversion -Wno-comment)
  set(GCC_CMD_ID AppleClang,Clang,GNU)
  target_compile_options(${PYLIB_NAME}
                         PRIVATE $<$<CXX_COMPILER_ID:${GCC_CMD_ID}>:${GCC_CMD_OPTIONS}>)
  target_compile_definitions(${PYLIB_NAME} PRIVATE MODULE_NAME=${PYLIB_NAME})
  target_include_directories(
    ${PYLIB_NAME} PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/>
                         $<INSTALL_INTERFACE:include/${PROJECT_NAME}>)

  # link solver, eigenpy, Boost.Python
  target_link_libraries(${PYLIB_NAME} PUBLIC ${PROJECT_NAME} eigenpy::eigenpy)
  install(
    TARGETS ${PYLIB_NAME}
    EXPORT ${TARGETS_EXPORT_NAME}
    DESTINATION ${${PYLIB_NAME}_INSTALL_DIR})
endfunction()

make_bindings()

install(
  DIRECTORY include/${PROJECT_NAME}
  DESTINATION "include"
  FILES_MATCHING
  PATTERN "*.hpp")

# COPY PYTHON FILES INTO OUTPUT FOLDER
set(${PYLIB_NAME}_PYFILES __init__.py casadi_utils.py utils.py windows_dll_manager.py)

foreach(pyfile ${${PYLIB_NAME}_PYFILES})
  python_install_on_site(proxsuite_nlp ${pyfile})
endforeach()

# Generate stubs
if(GENERATE_PYTHON_STUBS)
  load_stubgen()
  generate_stubs(${CMAKE_CURRENT_BINARY_DIR} proxsuite_nlp ${PYTHON_SITELIB} ${PYLIB_NAME}
                 ${PROJECT_NAME})
endif(GENERATE_PYTHON_STUBS)
