include(${PROJECT_SOURCE_DIR}/cmake/stubs.cmake)

# identify bindings C++ source files
file(GLOB_RECURSE ${PY_NAME}_SOURCES
     ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp
     )

file(GLOB_RECURSE ${PY_NAME}_HEADERS
     ${CMAKE_CURRENT_SOURCE_DIR}/*.hpp
     )

set(PY_HEADERS ${${PY_NAME}_HEADERS})
set(PY_SOURCES ${${PY_NAME}_SOURCES})

ADD_HEADER_GROUP(PY_HEADERS)

message(STATUS "Python headers: ${PY_HEADERS}")

function(MAKE_BINDINGS)
  add_library(${PY_NAME} SHARED ${PY_SOURCES} ${PY_HEADERS})

  target_include_directories(${PY_NAME} SYSTEM PRIVATE ${PYTHON_INCLUDE_DIRS})
  set_target_properties(${PY_NAME}
    PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${PROJECT_NAME}
    PREFIX ""
    SUFFIX ${PYTHON_EXT_SUFFIX}
    VERSION ${PROJECT_VERSION}
    )
  if(UNIX AND NOT APPLE)
    set_target_properties(${PY_NAME} PROPERTIES INSTALL_RPATH "\$ORIGIN/../../..")
  endif()

  target_compile_options(${PY_NAME} PRIVATE -Wno-conversion -Wno-comment)

  # link solver, eigenpy, Boost.Python
  target_link_libraries(${PY_NAME} PUBLIC ${PROJECT_NAME} eigenpy::eigenpy)
  target_link_libraries(${PY_NAME} PRIVATE fmt::fmt)
  target_link_boost_python(${PY_NAME} PUBLIC)
  install(TARGETS ${PY_NAME}
          EXPORT ${TARGETS_EXPORT_NAME}
          DESTINATION ${${PY_NAME}_INSTALL_DIR})
endfunction()

MAKE_BINDINGS()

file(MAKE_DIRECTORY
     ${${PROJECT_NAME}_BINARY_DIR}/include/${PROJECT_NAME}/python
     )

# copy headers to build/include/ directory
foreach(pyheader ${PY_HEADERS})
  string(REGEX REPLACE "${CMAKE_CURRENT_SOURCE_DIR}/" "" pyheader ${pyheader})
  execute_process(COMMAND ${CMAKE_COMMAND} -E ${LINK}
                  ${CMAKE_CURRENT_SOURCE_DIR}/${pyheader}
                  ${${PROJECT_NAME}_BINARY_DIR}/include/${PROJECT_NAME}/python/${pyheader})
endforeach(pyheader)

install(FILES ${PY_HEADERS}
        DESTINATION ${CMAKE_INSTALL_PREFIX}/include/${PROJECT_NAME}/python
        PERMISSIONS OWNER_READ GROUP_READ WORLD_READ OWNER_WRITE)

# COPY PYTHON FILES INTO OUTPUT FOLDER
set(${PY_NAME}_PYFILES
    __init__.py
    utils.py)

foreach(pyfile ${${PY_NAME}_PYFILES})
	python_install_on_site(${PROJECT_NAME} ${pyfile})
endforeach()

# Generate stubs
if(GENERATE_PYTHON_STUBS)
  load_stubgen()
  # generate_stubs(${CMAKE_CURRENT_BINARY_DIR} ${PROJECT_NAME} ${PYTHON_SITELIB})
endif(GENERATE_PYTHON_STUBS)
