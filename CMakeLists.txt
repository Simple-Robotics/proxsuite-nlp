#
# Copyright (c) 2022 CNRS INRIA
#

CMAKE_MINIMUM_REQUIRED(VERSION 3.0)

SET(PROJECT_NAME lienlp)
SET(PROJECT_CUSTOM_HEADER_EXTENSION "hpp")
SET(PROJECT_USE_CMAKE_EXPORT TRUE)
SET(PROJECT_USE_KEYWORD_LINK_LIBRARIES TRUE)

# Disable -Werror on Unix for now.
SET(CXX_DISABLE_WERROR True)
SET(CMAKE_VERBOSE_MAKEFILE True)

# ----------------------------------------------------
# --- OPTIONS  ---------------------------------------
# Need to be set before including base.cmake
# ----------------------------------------------------
option(INSTALL_DOCUMENTATION "Generate and install the documentation" OFF)

# Check if the submodule cmake have been initialized
IF(NOT EXISTS "${CMAKE_CURRENT_LIST_DIR}/cmake/base.cmake")
  message(FATAL_ERROR "\nPlease run the following command first:\ngit submodule update --init\n")
ENDIF()

set(DOXYGEN_USE_MATHJAX YES)
set(DOXYGEN_USE_TEMPLATE_CSS YES)

include(${CMAKE_CURRENT_LIST_DIR}/cmake/base.cmake)
COMPUTE_PROJECT_ARGS(PROJECT_ARGS LANGUAGES CXX)
project(${PROJECT_NAME} ${PROJECT_ARGS})

include(${CMAKE_CURRENT_LIST_DIR}/cmake/boost.cmake)
include(${CMAKE_CURRENT_LIST_DIR}/cmake/python.cmake)
IF(APPLE) # Use the handmade approach
  set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_LIST_DIR}/cmake/find-external/OpenMP ${CMAKE_MODULE_PATH})
ENDIF(APPLE)
include(CMakeDependentOption)

# If needed, set CMake policy for APPLE systems
APPLY_DEFAULT_APPLE_CONFIGURATION()

IF(WIN32)
  set(LINK copy_if_different)
ELSE(WIN32)
  set(LINK create_symlink)
ENDIF(WIN32)

# --- OPTIONS ----------------------------------------
option(BUILD_UTILS "Build the utils" OFF)
OPTION(BUILD_PYTHON_INTERFACE "Build the Python bindings" ON)
option(BUILD_WITH_COMMIT_VERSION "Build libraries by setting specific commit version" OFF)

# --- OPTIONAL DEPENDENCIES -------------------------
option(BUILD_WITH_PINOCCHIO_SUPPORT, "Build the library with support for Pinocchio" OFF)
option(BUILD_WITH_CASADI_SUPPORT "Build the library with the support of CASADI" OFF)
option(BUILD_WITH_OPENMP_SUPPORT "Build the library with the OpenMP support" OFF)
cmake_dependent_option(LINK_PYTHON_INTERFACE_TO_OPENMP "Link OpenMP to the Python interface" ON BUILD_WITH_OPENMP_SUPPORT OFF)
cmake_dependent_option(BUILD_PYTHON_BINDINGS_WITH_BOOST_MPFR_SUPPORT "Build the Python interface with Boost.Multiprecision MPFR support" ON BUILD_PYTHON_INTERFACE OFF)
option(GENERATE_PYTHON_STUBS "Generate the Python stubs associated to the Python library" OFF)

option(INITIALIZE_WITH_NAN "Initialize Eigen entries with NaN" OFF)
option(CHECK_RUNTIME_MALLOC "Check if some memory allocations are performed at runtime" OFF)

option(ENABLE_TEMPLATE_INSTANTIATION "Instantiate templates with default type for library build" ON)

IF(INITIALIZE_WITH_NAN)
  message(STATUS "Initialize with NaN all the Eigen entries.")
  add_definitions(-DEIGEN_INITIALIZE_MATRICES_BY_NAN)
ENDIF(INITIALIZE_WITH_NAN)

IF(CHECK_RUNTIME_MALLOC)
  MESSAGE(STATUS "Check if some memory allocations are performed at runtime.")
  add_definitions(-DLIENLP_EIGEN_CHECK_MALLOC)
  add_definitions(-DEIGEN_RUNTIME_NO_MALLOC)
ENDIF(CHECK_RUNTIME_MALLOC)

MACRO(TAG_LIBRARY_VERSION target)
  set_target_properties(${target} PROPERTIES SOVERSION ${PROJECT_VERSION})
ENDMACRO(TAG_LIBRARY_VERSION)

# ----------------------------------------------------
# --- DEPENDENCIES -----------------------------------
# ----------------------------------------------------
ADD_PROJECT_DEPENDENCY(Eigen3 REQUIRED PKG_CONFIG_REQUIRES "eigen3 >= 3.0.5")

# Variable containing all the cflags definition relative to optional dependencies
SET(CFLAGS_DEPENDENCIES)

IF(BUILD_WITH_CASADI_SUPPORT)
  ADD_PROJECT_DEPENDENCY(casadi 3.4.5 REQUIRED PKG_CONFIG_REQUIRES "casadi >= 3.4.5")
ENDIF(BUILD_WITH_CASADI_SUPPORT)

IF(BUILD_WITH_OPENMP_SUPPORT)
  find_package(OpenMP REQUIRED)
ENDIF(BUILD_WITH_OPENMP_SUPPORT)

set(BOOST_REQUIRED_COMPONENTS filesystem serialization system)

SET_BOOST_DEFAULT_OPTIONS()
EXPORT_BOOST_DEFAULT_OPTIONS()
ADD_PROJECT_DEPENDENCY(Boost REQUIRED COMPONENTS ${BOOST_REQUIRED_COMPONENTS})

IF(BUILD_PYTHON_INTERFACE)
  MESSAGE(STATUS "The Python bindings of Pinocchio will be compiled along the main library. If you want to disable this feature, please set the option BUILD_PYTHON_INTERFACE to OFF.")
  FINDPYTHON(REQUIRED)
  SEARCH_FOR_BOOST_PYTHON(REQUIRED)
  ADD_PROJECT_DEPENDENCY(eigenpy 2.6.7 REQUIRED)

  IF(BUILD_WITH_AUTODIFF_SUPPORT)
    ADD_PROJECT_DEPENDENCY(pycppad 1.1.0 REQUIRED)
    add_definitions(-DPYCPPAD_EXCLUDE_EIGEN_NUMTRAITS_SPECIALIZATION)
  ENDIF(BUILD_WITH_AUTODIFF_SUPPORT)


  # Check wether this a PyPy Python
  EXECUTE_PROCESS(COMMAND ${PYTHON_EXECUTABLE} -c "import platform; print(platform.python_implementation())"
    OUTPUT_VARIABLE _python_implementation_value
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET)

  message(STATUS "Python compiler: ${_python_implementation_value}")
  IF(_python_implementation_value MATCHES "PyPy")
    set(BUILD_PYTHON_INTERFACE_WITH_PYPY TRUE)
  ENDIF()

ENDIF(BUILD_PYTHON_INTERFACE)

# Enforce the preprocessed version of boost::list and boost::vector
# but it avoids any compilation issue.
add_definitions(-DBOOST_MPL_LIMIT_LIST_SIZE=30)
add_definitions(-DBOOST_MPL_LIMIT_VECTOR_SIZE=30)

LIST(APPEND CFLAGS_DEPENDENCIES "-DBOOST_MPL_LIMIT_LIST_SIZE=30")

# Path to boost headers
include_directories(SYSTEM ${Boost_INCLUDE_DIRS})

# ----------------------------------------------------
# --- INCLUDE ----------------------------------------
# ----------------------------------------------------
FILE(GLOB_RECURSE HEADERS
     ${PROJECT_SOURCE_DIR}/src/*.hpp
     ${PROJECT_SOURCE_DIR}/src/*.hxx
     )

LIST(APPEND HEADERS macros.hpp)

set(BINARY_DIR "${PROJECT_NAME}_BINARY_DIR")

MAKE_DIRECTORY("${BINARY_DIR}/include/lienlp")



SET(HEADERS_)
FOREACH(header ${HEADERS})
  STRING(REGEX REPLACE "${PROJECT_SOURCE_DIR}/src/" "" header ${header})
  LIST(APPEND HEADERS_ ${header})
  GET_FILENAME_COMPONENT(headerName ${header} NAME)
  GET_FILENAME_COMPONENT(headerPath ${header} PATH)
  EXECUTE_PROCESS(COMMAND ${CMAKE_COMMAND} -E ${LINK}
    ${${PROJECT_NAME}_SOURCE_DIR}/src/${header}
    ${${PROJECT_NAME}_BINARY_DIR}/include/${PROJECT_NAME}/${header})
  INSTALL(FILES ${${PROJECT_NAME}_SOURCE_DIR}/src/${header}
          DESTINATION ${CMAKE_INSTALL_PREFIX}/include/${PROJECT_NAME}/${headerPath}
          PERMISSIONS OWNER_READ GROUP_READ WORLD_READ OWNER_WRITE)
ENDFOREACH(header)
SET(HEADERS ${HEADERS_})

add_subdirectory(src)
add_subdirectory(bindings)
add_subdirectory(unittest)

# --- PACKAGING ----------------------------------------------------------------
MACRO(EXPORT_VARIABLE var_name var_value)
  GET_DIRECTORY_PROPERTY(has_parent PARENT_DIRECTORY)
  IF(has_parent)
    SET(${var_name} ${var_value} PARENT_SCOPE)
  ELSE()
    SET(${var_name} ${var_value})
  ENDIF()
ENDMACRO(EXPORT_VARIABLE var_name var_value)

IF(BUILD_WITH_CASADI_SUPPORT)
  EXPORT_VARIABLE(PINOCCHIO_USE_CASADI ON)
  SET(PACKAGE_EXTRA_MACROS "${PACKAGE_EXTRA_MACROS}\nset(PINOCCHIO_USE_CASADI \"\")")
ENDIF()
IF(BUILD_PYTHON_INTERFACE)
  EXPORT_VARIABLE(PINOCCHIO_WITH_PYTHON_INTERFACE ON)
  SET(PACKAGE_EXTRA_MACROS "${PACKAGE_EXTRA_MACROS}\nset(PINOCCHIO_WITH_PYTHON_INTERFACE \"\")")
ENDIF()

PKG_CONFIG_APPEND_LIBS(${PROJECT_NAME})
PKG_CONFIG_APPEND_BOOST_LIBS(${BOOST_REQUIRED_COMPONENTS})

FOREACH(cflags ${CFLAGS_DEPENDENCIES})
  PKG_CONFIG_APPEND_CFLAGS(${cflags})
ENDFOREACH(cflags ${CFLAGS_DEPENDENCIES})

# Install catkin package.xml
INSTALL(FILES package.xml DESTINATION share/${PROJECT_NAME})
