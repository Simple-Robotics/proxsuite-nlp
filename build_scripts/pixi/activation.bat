:: Setup ccache
set CMAKE_CXX_COMPILER_LAUNCHER=ccache

:: Create compile_commands.json for language server
set CMAKE_EXPORT_COMPILE_COMMANDS=1

:: Activate color output with Ninja
set CMAKE_COLOR_DIAGNOSTICS=1

:: Set default build value only if not previously set
if not defined PROXSUITE_NLP_BUILD_TYPE (set PROXSUITE_NLP_BUILD_TYPE=Release)
if not defined PROXSUITE_NLP_PYTHON_STUBS (set PROXSUITE_NLP_PYTHON_STUBS=ON)
if not defined PROXSUITE_NLP_PINOCCHIO_SUPPORT (set PROXSUITE_NLP_PINOCCHIO_SUPPORT=OFF)
if not defined PROXSUITE_NLP_PROXSUITE_SUPPORT (set PROXSUITE_NLP_PROXSUITE_SUPPORT=OFF)
if not defined PROXSUITE_NLP_BENCHMARK (set PROXSUITE_NLP_BENCHMARK=OFF)
if not defined PROXSUITE_NLP_EXAMPLES (set PROXSUITE_NLP_EXAMPLES=ON)
