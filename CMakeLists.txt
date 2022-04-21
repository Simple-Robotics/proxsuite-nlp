#
# Copyright (C) 2022 CNRS INRIA
#

cmake_minimum_required(VERSION 3.1)

set(PROJECT_NAME proxnlp)
set(PROJECT_DESCRIPTION "An augmented Lagrangian solver for nonlinear programming on manifolds.")
set(PROJECT_URL "https://github.com/Simple-Robotics/proxnlp")
set(PROJECT_CUSTOM_HEADER_EXTENSION "hpp")
set(PROJECT_USE_KEYWORD_LINK_LIBRARIES True)

# Disable -Werror on Unix for now.
set(CXX_DISABLE_WERROR True)

set(CMAKE_CXX_EXTENSIONS OFF)

option(INSTALL_DOCUMENTATION "Generate and install the documentation" ON)

set(DOXYGEN_USE_MATHJAX YES)
set(DOXYGEN_USE_TEMPLATE_CSS YES)

include(${CMAKE_CURRENT_LIST_DIR}/cmake/base.cmake)
COMPUTE_PROJECT_ARGS(PROJECT_ARGS LANGUAGES CXX)
project(${PROJECT_NAME} ${PROJECT_ARGS})

include(${CMAKE_CURRENT_LIST_DIR}/cmake/boost.cmake)
include(${CMAKE_CURRENT_LIST_DIR}/cmake/python.cmake)
include(${CMAKE_CURRENT_LIST_DIR}/cmake/ide.cmake)
include(${CMAKE_CURRENT_LIST_DIR}/cmake/apple.cmake)
include(CMakeDependentOption)

if(WIN32)
  set(LINK copy_if_different)
else(WIN32)
  set(LINK create_symlink)
endif(WIN32)

# --- OPTIONS ----------------------------------------
option(BUILD_UTILS "Build the utils" OFF)
option(BUILD_PYTHON_INTERFACE "Build the Python bindings" ON)
option(BUILD_WITH_VERSION_SUFFIX "Build libraries with version appended to suffix" OFF)

# --- OPTIONAL DEPENDENCIES -------------------------
option(BUILD_WITH_PINOCCHIO_SUPPORT "Build the library with support for Pinocchio" ON)
option(BUILD_WITH_CASADI_SUPPORT "Build the library with the support of CASADI" OFF)
option(BUILD_WITH_OPENMP_SUPPORT "Build the library with the OpenMP support" ON)
cmake_dependent_option(LINK_PYTHON_INTERFACE_TO_OPENMP "Link OpenMP to the Python interface" ON BUILD_WITH_OPENMP_SUPPORT OFF)
option(GENERATE_PYTHON_STUBS "Generate the Python stubs associated to the Python library" OFF)

option(INITIALIZE_WITH_NAN "Initialize Eigen entries with NaN" OFF)
option(CHECK_RUNTIME_MALLOC "Check if some memory allocations are performed at runtime" OFF)

if(INITIALIZE_WITH_NAN)
  message(STATUS "Initialize with NaN all the Eigen entries.")
  add_definitions(-DEIGEN_INITIALIZE_MATRICES_BY_NAN)
endif(INITIALIZE_WITH_NAN)

if(CHECK_RUNTIME_MALLOC)
  message(STATUS "Check if some memory allocations are performed at runtime.")
  add_definitions(-DPROXNLP_EIGEN_CHECK_MALLOC)
  add_definitions(-DEIGEN_RUNTIME_NO_MALLOC)
endif(CHECK_RUNTIME_MALLOC)

macro(TAG_LIBRARY_VERSION target)
  set_target_properties(${target} PROPERTIES SOVERSION ${PROJECT_VERSION})
endmacro(TAG_LIBRARY_VERSION)

# ----------------------------------------------------
# --- DEPENDENCIES -----------------------------------
# ----------------------------------------------------
ADD_PROJECT_DEPENDENCY(Eigen3 REQUIRED PKG_CONFIG_REQUIRES "eigen3 >= 3.4.0")
ADD_PROJECT_DEPENDENCY(fmt REQUIRED PKG_CONFIG_REQUIRES "fmt >= 8.0.0")

# Variable containing all the cflags definition relative to optional dependencies
set(CFLAGS_DEPENDENCIES)

if(BUILD_WITH_CASADI_SUPPORT)
  ADD_PROJECT_DEPENDENCY(casadi 3.4.5 REQUIRED PKG_CONFIG_REQUIRES "casadi >= 3.4.5")
endif(BUILD_WITH_CASADI_SUPPORT)

if(BUILD_WITH_PINOCCHIO_SUPPORT)
  message(STATUS "Building with Pinocchio support.")
  ADD_PROJECT_DEPENDENCY(pinocchio REQUIRED PKG_CONFIG_REQUIRES "pinocchio >= 2.9.2")
  add_definitions(-DWITH_PINOCCHIO)
endif(BUILD_WITH_PINOCCHIO_SUPPORT)

if(BUILD_WITH_OPENMP_SUPPORT)
  ADD_PROJECT_DEPENDENCY(OpenMP REQUIRED)
endif(BUILD_WITH_OPENMP_SUPPORT)

set(BOOST_REQUIRED_COMPONENTS filesystem serialization system)

SET_BOOST_DEFAULT_OPTIONS()
EXPORT_BOOST_DEFAULT_OPTIONS()
ADD_PROJECT_DEPENDENCY(Boost REQUIRED COMPONENTS ${BOOST_REQUIRED_COMPONENTS})

if(BUILD_PYTHON_INTERFACE)
  FINDPYTHON(REQUIRED)
  set(PY_NAME "py${PROJECT_NAME}")
  set(${PY_NAME}_INSTALL_DIR ${PYTHON_SITELIB}/${PROJECT_NAME})
  SEARCH_FOR_BOOST_PYTHON(REQUIRED)
  ADD_PROJECT_DEPENDENCY(eigenpy 2.7.1 REQUIRED)

  execute_process(COMMAND ${PYTHON_EXECUTABLE} -c "import platform; print(platform.python_implementation())"
    OUTPUT_VARIABLE _python_implementation_value
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET)
  message(STATUS "Python compiler: ${_python_implementation_value}")

endif(BUILD_PYTHON_INTERFACE)


# Path to boost headers
include_directories(SYSTEM ${Boost_INCLUDE_DIRS})

# ----------------------------------------------------
# --- INCLUDE ----------------------------------------
# ----------------------------------------------------
file(GLOB_RECURSE HEADERS
     ${PROJECT_SOURCE_DIR}/src/*.hpp
     ${PROJECT_SOURCE_DIR}/src/*.hxx
     )

file(MAKE_DIRECTORY
     ${${PROJECT_NAME}_BINARY_DIR}/include/${PROJECT_NAME}
     ${${PROJECT_NAME}_BINARY_DIR}/include/${PROJECT_NAME}/helpers
     ${${PROJECT_NAME}_BINARY_DIR}/include/${PROJECT_NAME}/meritfuncs
     ${${PROJECT_NAME}_BINARY_DIR}/include/${PROJECT_NAME}/modelling/autodiff
     ${${PROJECT_NAME}_BINARY_DIR}/include/${PROJECT_NAME}/modelling/spaces
     ${${PROJECT_NAME}_BINARY_DIR}/include/${PROJECT_NAME}/modelling/costs
     ${${PROJECT_NAME}_BINARY_DIR}/include/${PROJECT_NAME}/modelling/residuals
     ${${PROJECT_NAME}_BINARY_DIR}/include/${PROJECT_NAME}/modelling/constraints
     )

# Process headers
set(HEADERS_LIST)
foreach(header ${HEADERS})
  string(REGEX REPLACE "${PROJECT_SOURCE_DIR}/src/" "" header ${header})
  list(APPEND HEADERS_LIST ${header})
  get_filename_component(headerName ${header} NAME)
  get_filename_component(headerPath ${header} PATH)
  execute_process(COMMAND ${CMAKE_COMMAND} -E ${LINK}
    ${${PROJECT_NAME}_SOURCE_DIR}/src/${header}
    ${${PROJECT_NAME}_BINARY_DIR}/include/${PROJECT_NAME}/${header})
  install(FILES ${${PROJECT_NAME}_SOURCE_DIR}/src/${header}
          DESTINATION ${CMAKE_INSTALL_PREFIX}/include/${PROJECT_NAME}/${headerPath}
          PERMISSIONS OWNER_READ GROUP_READ WORLD_READ OWNER_WRITE)
endforeach(header)
set(HEADERS ${HEADERS_LIST})

# Source files
add_subdirectory(src)

# Python interface
if(BUILD_PYTHON_INTERFACE)
  add_subdirectory(python)
endif()
add_subdirectory(examples)

# Unit tests
add_subdirectory(unittest)

# --- PACKAGING ----------------------------------------------------------------
macro(EXPORT_VARIABLE var_name var_value)
  get_directory_property(has_parent PARENT_DIRECTORY)
  if(has_parent)
    set(${var_name} ${var_value} PARENT_SCOPE)
  else()
    set(${var_name} ${var_value})
  endif()
endmacro(EXPORT_VARIABLE var_name var_value)

if(BUILD_WITH_CASADI_SUPPORT)
  EXPORT_VARIABLE(PROXNLP_USE_CASADI ON)
  set(PACKAGE_EXTRA_MACROS "${PACKAGE_EXTRA_MACROS}\nset(PROXNLP_USE_CASADI \"\")")
endif()
if(BUILD_PYTHON_INTERFACE)
  EXPORT_VARIABLE(PROXNLP_WITH_PYTHON_INTERFACE ON)
  set(PACKAGE_EXTRA_MACROS "${PACKAGE_EXTRA_MACROS}\nset(PROXNLP_WITH_PYTHON_INTERFACE \"\")")
endif()

PKG_CONFIG_APPEND_LIBS(${PROJECT_NAME})
PKG_CONFIG_APPEND_BOOST_LIBS(${BOOST_REQUIRED_COMPONENTS})

foreach(cflags ${CFLAGS_DEPENDENCIES})
  PKG_CONFIG_APPEND_CFLAGS(${cflags})
endforeach(cflags ${CFLAGS_DEPENDENCIES})

# Install catkin package.xml
install(FILES package.xml DESTINATION share/${PROJECT_NAME})
