#! /bin/bash
# Activation script

# Remove flags setup from cxx-compiler
unset CFLAGS
unset CPPFLAGS
unset CXXFLAGS
unset DEBUG_CFLAGS
unset DEBUG_CPPFLAGS
unset DEBUG_CXXFLAGS
unset LDFLAGS

if [[ $host_alias == *"apple"* ]];
then
  # On OSX setting the rpath and -L it's important to use the conda libc++ instead of the system one.
  # If conda-forge use install_name_tool to package some libs, -headerpad_max_install_names is then mandatory
  export LDFLAGS="-Wl,-headerpad_max_install_names -Wl,-rpath,$CONDA_PREFIX/lib -L$CONDA_PREFIX/lib"
elif [[ $host_alias == *"linux"* ]];
then
  # On GNU/Linux, I don't know if these flags are mandatory with g++ but
  # it allow to use clang++ as compiler
  export LDFLAGS="-Wl,-rpath,$CONDA_PREFIX/lib -Wl,-rpath-link,$CONDA_PREFIX/lib -L$CONDA_PREFIX/lib"
fi

# Setup ccache
export CMAKE_CXX_COMPILER_LAUNCHER=ccache

# Create compile_commands.json for language server
export CMAKE_EXPORT_COMPILE_COMMANDS=1

# Activate color output with Ninja
export CMAKE_COLOR_DIAGNOSTICS=1

# Set default build value only if not previously set
export PROXSUITE_NLP_BUILD_TYPE=${PROXSUITE_NLP_BUILD_TYPE:=Release}
export PROXSUITE_NLP_PYTHON_STUBS=${PROXSUITE_NLP_PYTHON_STUBS:=ON}
export PROXSUITE_NLP_PINOCCHIO_SUPPORT=${PROXSUITE_NLP_PINOCCHIO_SUPPORT:=OFF}
export PROXSUITE_NLP_PROXSUITE_SUPPORT=${PROXSUITE_NLP_PROXSUITE_SUPPORT:=OFF}
export PROXSUITE_NLP_BENCHMARK=${PROXSUITE_NLP_BENCHMARK:=OFF}
export PROXSUITE_NLP_EXAMPLES=${PROXSUITE_NLP_EXAMPLES:=ON}