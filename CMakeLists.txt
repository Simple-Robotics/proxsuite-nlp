#
# Copyright (C) 2022-2023 LAAS-CNRS, INRIA
#

cmake_minimum_required(VERSION 3.12)

set(PROJECT_NAME proxnlp)
set(PROJECT_DESCRIPTION
    "A primal-dual augmented Lagrangian-type solver for nonlinear programming on manifolds.")
set(PROJECT_URL "https://github.com/Simple-Robotics/proxnlp")
set(PROJECT_CUSTOM_HEADER_EXTENSION "hpp")
set(PROJECT_USE_KEYWORD_LINK_LIBRARIES True)

set(CXX_DISABLE_WERROR True)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(PROJECT_USE_CMAKE_EXPORT True)

option(INSTALL_DOCUMENTATION "Generate and install the documentation" ON)

set(DOXYGEN_USE_MATHJAX YES)
set(DOXYGEN_USE_TEMPLATE_CSS YES)

include(${CMAKE_CURRENT_LIST_DIR}/cmake/base.cmake)
compute_project_args(PROJECT_ARGS LANGUAGES CXX)
project(${PROJECT_NAME} ${PROJECT_ARGS})
set(CMAKE_VERBOSE_MAKEFILE OFF)

include(${CMAKE_CURRENT_LIST_DIR}/cmake/boost.cmake)
include(${CMAKE_CURRENT_LIST_DIR}/cmake/ide.cmake)
include(${CMAKE_CURRENT_LIST_DIR}/cmake/apple.cmake)

if(APPLE) # Use the handmade approach
  set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_LIST_DIR}/cmake/find-external/OpenMP ${CMAKE_MODULE_PATH})
elseif(UNIX)
  if(${CMAKE_VERSION} VERSION_GREATER "3.20.0" OR ${CMAKE_VERSION} VERSION_EQUAL "3.20.0")
    set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_LIST_DIR}/cmake/find-external/OpenMP ${CMAKE_MODULE_PATH})
  endif()
endif(APPLE)
include(CMakeDependentOption)

apply_default_apple_configuration()

if(WIN32)
  set(LINK copy_if_different)
else(WIN32)
  set(LINK create_symlink)
endif(WIN32)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  message(STATUS "Setting build type to 'Release' as none was specified.")
  set(CMAKE_BUILD_TYPE
      Release
      CACHE STRING "Choose the type of build." FORCE)
  # Set the possible values of build type for cmake-gui
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel"
                                               "RelWithDebInfo")
endif()

# --- OPTIONS ----------------------------------------
option(BUILD_PYTHON_INTERFACE "Build the Python bindings" ON)
option(BUILD_WITH_VERSION_SUFFIX "Build libraries with version appended to suffix" OFF)
option(ENABLE_TEMPLATE_INSTANTIATION "Template instantiation of the main library" ON)
option(BUILD_EXAMPLES "Build the examples" ON)
OPTION(BUILD_BENCHMARK "Build the benchmarks" OFF)

# --- OPTIONAL DEPENDENCIES -------------------------
option(BUILD_WITH_PINOCCHIO_SUPPORT "Build the library with support for Pinocchio" ON)
option(BUILD_WITH_OPENMP_SUPPORT "Build the library with the OpenMP support" OFF)
option(BUILD_WITH_PROXSUITE "Build the library with ProxSuite's LDLT" OFF)
cmake_dependent_option(
  GENERATE_PYTHON_STUBS "Generate the Python stubs associated to the Python library" OFF
  BUILD_PYTHON_INTERFACE OFF)

option(INITIALIZE_WITH_NAN "Initialize Eigen entries with NaN" OFF)
option(CHECK_RUNTIME_MALLOC "Check if some memory allocations are performed at runtime" OFF)

if(INITIALIZE_WITH_NAN)
  message(STATUS "Initialize with NaN all the Eigen entries.")
  add_definitions(-DEIGEN_INITIALIZE_MATRICES_BY_NAN)
endif(INITIALIZE_WITH_NAN)

if(CHECK_RUNTIME_MALLOC)
  message(STATUS "Check if some memory allocations are performed at runtime.")
  add_compile_definitions(PROXNLP_EIGEN_CHECK_MALLOC)
  add_definitions(-DEIGEN_RUNTIME_NO_MALLOC)
endif(CHECK_RUNTIME_MALLOC)

if(ENABLE_TEMPLATE_INSTANTIATION)
  add_definitions(-DPROXNLP_ENABLE_TEMPLATE_INSTANTIATION)
endif(ENABLE_TEMPLATE_INSTANTIATION)

macro(TAG_LIBRARY_VERSION target)
  set_target_properties(${target} PROPERTIES SOVERSION ${PROJECT_VERSION})
endmacro(TAG_LIBRARY_VERSION)

# ----------------------------------------------------
# --- DEPENDENCIES -----------------------------------
# ----------------------------------------------------
add_project_dependency(Eigen3 3.3.7 REQUIRED PKG_CONFIG_REQUIRES "eigen3 >= 3.3.7")
add_project_dependency(fmt "9.1.0...<11" REQUIRED PKG_CONFIG_REQUIRES "fmt >= 9.1.0")

if(BUILD_WITH_PROXSUITE)
  add_project_dependency(proxsuite REQUIRED)
  add_definitions(-DPROXNLP_ENABLE_PROXSUITE_LDLT)
endif()

# Variable containing all the cflags definition relative to optional dependencies
set(CFLAGS_DEPENDENCIES)

if(BUILD_WITH_OPENMP_SUPPORT)
  find_package(OpenMP REQUIRED)
  add_compile_definitions(PROXNLP_MULTITHREADING)
endif(BUILD_WITH_OPENMP_SUPPORT)

set(BOOST_REQUIRED_COMPONENTS filesystem serialization system)

set_boost_default_options()
export_boost_default_options()
add_project_dependency(Boost REQUIRED COMPONENTS ${BOOST_REQUIRED_COMPONENTS})

if(BUILD_PYTHON_INTERFACE)
  set(PYTHON_COMPONENTS Interpreter Development.Module NumPy Development)
  set(PYLIB_NAME "py${PROJECT_NAME}")
  add_project_dependency(eigenpy 3.1.0 REQUIRED PKG_CONFIG_REQUIRES "eigenpy >= 3.1.0")
  set(${PYLIB_NAME}_INSTALL_DIR ${PYTHON_SITELIB}/${PROJECT_NAME})

  execute_process(
    COMMAND ${PYTHON_EXECUTABLE} -c "import platform; print(platform.python_implementation())"
    OUTPUT_VARIABLE _python_implementation_value
    OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)
  message(STATUS "Python compiler: ${_python_implementation_value}")

endif(BUILD_PYTHON_INTERFACE)

if(BUILD_WITH_PINOCCHIO_SUPPORT)
  message(STATUS "Building with Pinocchio support.")
  add_project_dependency(pinocchio REQUIRED PKG_CONFIG_REQUIRES "pinocchio >= 2.9.1")
  add_compile_definitions(PROXNLP_WITH_PINOCCHIO)
endif(BUILD_WITH_PINOCCHIO_SUPPORT)

# --- MAIN LIBRARY ----------------------------------------
set(LIB_HEADER_DIR ${PROJECT_SOURCE_DIR}/include/proxnlp)
file(GLOB_RECURSE LIB_HEADERS ${LIB_HEADER_DIR}/*.hpp ${LIB_HEADER_DIR}/*.hxx)

set(LIB_SOURCES ${PROJECT_SOURCE_DIR}/src/block-kind.cpp)

if(ENABLE_TEMPLATE_INSTANTIATION)

  set(LIB_TEMPLATE_SOURCES
      ${PROJECT_SOURCE_DIR}/src/block-ldlt.cpp
      ${PROJECT_SOURCE_DIR}/src/constraints.cpp
      ${PROJECT_SOURCE_DIR}/src/manifold-base.cpp
      ${PROJECT_SOURCE_DIR}/src/cartesian-product.cpp
      ${PROJECT_SOURCE_DIR}/src/function-base.cpp
      ${PROJECT_SOURCE_DIR}/src/squared-distance.cpp
      ${PROJECT_SOURCE_DIR}/src/quadratic-residual.cpp
      ${PROJECT_SOURCE_DIR}/src/cost-function.cpp
      ${PROJECT_SOURCE_DIR}/src/cost-sum.cpp
      ${PROJECT_SOURCE_DIR}/src/solver-base.cpp
      ${PROJECT_SOURCE_DIR}/src/linesearch.cpp
      ${PROJECT_SOURCE_DIR}/src/results.cpp
      ${PROJECT_SOURCE_DIR}/src/problem-base.cpp
      ${PROJECT_SOURCE_DIR}/src/workspace.cpp
      ${PROJECT_SOURCE_DIR}/src/pdal.cpp)
  set(LIB_TEMPLATE_DECLARATIONS
      ${LIB_HEADER_DIR}/linalg/block-ldlt.txx
      ${LIB_HEADER_DIR}/manifold-base.txx
      ${LIB_HEADER_DIR}/function-base.txx
      ${LIB_HEADER_DIR}/cost-function.txx
      ${LIB_HEADER_DIR}/cost-sum.txx
      ${LIB_HEADER_DIR}/solver-base.txx
      ${LIB_HEADER_DIR}/results.txx
      ${LIB_HEADER_DIR}/problem-base.txx
      ${LIB_HEADER_DIR}/workspace.txx
      ${LIB_HEADER_DIR}/pdal.txx
      ${LIB_HEADER_DIR}/linesearch.txx
      ${LIB_HEADER_DIR}/modelling/costs/squared-distance.txx
      ${LIB_HEADER_DIR}/modelling/costs/quadratic-residual.txx
      ${LIB_HEADER_DIR}/modelling/spaces/cartesian-product.txx)
  if(BUILD_WITH_PINOCCHIO_SUPPORT)
    list(APPEND LIB_TEMPLATE_SOURCES ${PROJECT_SOURCE_DIR}/src/multibody/rigid-transform-point.cpp
         ${PROJECT_SOURCE_DIR}/src/multibody/spaces.cpp)
    list(APPEND LIB_TEMPLATE_DECLARATIONS
         ${LIB_HEADER_DIR}/modelling/residuals/rigid-transform-point.txx
         ${LIB_HEADER_DIR}/modelling/spaces/multibody.txx)
  endif(BUILD_WITH_PINOCCHIO_SUPPORT)
  list(APPEND LIB_HEADERS ${LIB_TEMPLATE_DECLARATIONS})
  list(APPEND LIB_SOURCES ${LIB_TEMPLATE_SOURCES})

endif(ENABLE_TEMPLATE_INSTANTIATION)

function(create_library)
  # get the compile defs
  get_directory_property(CURRENT_COMPILE_DEFINITIONS COMPILE_DEFINITIONS)

  add_library(${PROJECT_NAME} SHARED ${LIB_HEADERS} ${LIB_SOURCES})
  set_target_properties(${PROJECT_NAME} PROPERTIES LINKER_LANGUAGE CXX)
  target_compile_definitions(${PROJECT_NAME} PUBLIC ${CURRENT_COMPILE_DEFINITIONS})

  set_target_properties(${PROJECT_NAME} PROPERTIES LINKER_LANGUAGE CXX VERSION ${PROJECT_VERSION})
  if(BUILD_LIBRARY_WITH_VERSION_SUFFIX)
    tag_library_version(${PROJECT_NAME})
  endif()

  if(BUILD_WITH_PINOCCHIO_SUPPORT)
    target_link_libraries(${PROJECT_NAME} PUBLIC pinocchio::pinocchio)
  endif(BUILD_WITH_PINOCCHIO_SUPPORT)

  target_link_libraries(${PROJECT_NAME} PUBLIC Eigen3::Eigen)
  target_link_libraries(${PROJECT_NAME} PUBLIC Boost::filesystem Boost::system Boost::serialization)
  target_link_libraries(${PROJECT_NAME} PUBLIC fmt::fmt)
  if(BUILD_WITH_PROXSUITE)
    target_link_libraries(${PROJECT_NAME} PUBLIC proxsuite::proxsuite-vectorized)
  endif(BUILD_WITH_PROXSUITE)
  if(BUILD_WITH_OPENMP_SUPPORT)
    target_link_libraries(${PROJECT_NAME} PUBLIC OpenMP::OpenMP_CXX)
  endif()
  target_include_directories(${PROJECT_NAME} PUBLIC $<INSTALL_INTERFACE:include>)

  if(MSVC)
    target_compile_options(${PROJECT_NAME} PUBLIC "/bigobj")
  endif(MSVC)

  foreach(headerFile ${LIB_HEADERS})
    string(REGEX REPLACE "${PROJECT_SOURCE_DIR}/" "" headerFileRelative ${headerFile})
    get_filename_component(headerPath ${headerFileRelative} PATH)
    install(
      FILES ${headerFileRelative}
      DESTINATION ${CMAKE_INSTALL_PREFIX}/${headerPath}
      PERMISSIONS OWNER_READ GROUP_READ WORLD_READ OWNER_WRITE)
  endforeach(headerFile ${LIB_HEADERS})

endfunction()

create_library()

add_header_group(LIB_HEADERS)

install(
  TARGETS ${PROJECT_NAME}
  EXPORT ${TARGETS_EXPORT_NAME}
  PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_FULL_INCLUDEDIR}
  INCLUDES
  DESTINATION ${CMAKE_INSTALL_FULL_INCLUDEDIR}
  LIBRARY DESTINATION ${CMAKE_INSTALL_FULL_LIBDIR}
  ARCHIVE DESTINATION ${CMAKE_INSTALL_FULL_LIBDIR}
  RUNTIME DESTINATION ${CMAKE_INSTALL_FULL_BINDIR})

# Python interface
if(BUILD_PYTHON_INTERFACE)
  add_subdirectory(python)
endif()

if(BUILD_EXAMPLES)
  add_subdirectory(examples)
endif()
if(BUILD_TESTING)
  add_subdirectory(tests)
endif()

# --- PACKAGING ----------------------------------------------------------------
macro(EXPORT_VARIABLE var_name var_value)
  get_directory_property(has_parent PARENT_DIRECTORY)
  if(has_parent)
    set(${var_name}
        ${var_value}
        PARENT_SCOPE)
  else()
    set(${var_name} ${var_value})
  endif()
  string(APPEND PACKAGE_EXTRA_MACROS "\nset(${var_name} ${var_value})")
endmacro(
  EXPORT_VARIABLE
  var_name
  var_value)

if(BUILD_WITH_PINOCCHIO_SUPPORT)
  export_variable(PROXNLP_WITH_PINOCCHIO_SUPPORT ON)
endif()
if(BUILD_WITH_CASADI_SUPPORT)
  export_variable(PROXNLP_USE_CASADI ON)
endif()
if(BUILD_PYTHON_INTERFACE)
  export_variable(PROXNLP_WITH_PYTHON_INTERFACE ON)
endif()

pkg_config_append_libs(${PROJECT_NAME})
pkg_config_append_boost_libs(${BOOST_REQUIRED_COMPONENTS})

foreach(cflags ${CFLAGS_DEPENDENCIES})
  pkg_config_append_cflags(${cflags})
endforeach(cflags ${CFLAGS_DEPENDENCIES})

# Install catkin package.xml
install(FILES package.xml DESTINATION share/${PROJECT_NAME})
