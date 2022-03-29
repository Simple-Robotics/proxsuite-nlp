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

add_library(${PY_NAME} SHARED ${${PY_NAME}_SOURCES})

target_include_directories(${PY_NAME} SYSTEM PRIVATE ${PYTHON_INCLUDE_DIRS})
set_target_properties(${PY_NAME}
  PROPERTIES
  INSTALL_RPATH "\$ORIGIN/../../.."
  LIBRARY_OUTPUT_DIRECTORY ${PROJECT_NAME}
  PREFIX ""
  SUFFIX ${PYTHON_EXT_SUFFIX}
  VERSION ${PROJECT_VERSION}
  )

# link solver, eigenpy, Boost.Python
target_link_libraries(${PY_NAME} PUBLIC ${PROJECT_NAME} eigenpy::eigenpy)
target_link_libraries(${PY_NAME} PRIVATE fmt::fmt)
target_link_boost_python(${PY_NAME} PUBLIC)
install(TARGETS ${PY_NAME} DESTINATION ${${PY_NAME}_INSTALL_DIR})

message(STATUS "Python headers: ${PY_HEADERS}")
message(STATUS "Python sources: ${PY_SOURCES}")

foreach(pyheader ${PY_HEADERS})
  message(STATUS "Processing Python header ${pyheader}")
  string(REGEX REPLACE "${CMAKE_CURRENT_SOURCE_DIR}/" "" pyheader ${pyheader})
  execute_process(COMMAND ${CMAKE_COMMAND} -E ${LINK}
                  ${CMAKE_CURRENT_SOURCE_DIR}/${pyheader}
                  ${${PROJECT_NAME}_BINARY_DIR}/include/${PROJECT_NAME}/python/${pyheader})
endforeach(pyheader)

file(MAKE_DIRECTORY
     ${${PROJECT_NAME}_BINARY_DIR}/include/${PROJECT_NAME}/python
     )

# COPY PYTHON FILES INTO OUTPUT FOLDER
set(${PY_NAME}_PYFILES
    __init__.py)

file(GLOB_RECURSE PY_EXAMPLES
     ${CMAKE_CURRENT_SOURCE_DIR}/examples/*.py)

message(STATUS "Python examples: ${PY_EXAMPLES}")

foreach(pyfile ${${PY_NAME}_PYFILES})
	python_install_on_site(${PROJECT_NAME} ${pyfile})
endforeach()

foreach(pyfile ${PY_EXAMPLES})
	get_filename_component(pyfile_name ${pyfile} NAME)
	configure_file(
		${pyfile}
		${CMAKE_CURRENT_BINARY_DIR}/examples/${pyfile_name}
		COPYONLY
		)
endforeach()
