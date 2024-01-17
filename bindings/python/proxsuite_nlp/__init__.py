"""
Copyright (C) 2022 LAAS-CNRS, INRIA
"""

# On Windows, if proxsuite-nlp.dll is not in the same directory than
# the .pyd, it will not be loaded.
# We first try to load proxsuite-nlp, then, if it fail and we are on Windows:
#  1. We add all paths inside proxsuite-nlp_WINDOWS_DLL_PATH to DllDirectory
#  2. If proxsuite-nlp_WINDOWS_DLL_PATH we add the relative path from the
#     package directory to the bin directory to DllDirectory
# This solution is inspired from:
#  - https://github.com/PixarAnimationStudios/OpenUSD/pull/1511/files
#  - https://stackoverflow.com/questions/65334494/python-c-extension-packaging-dll-along-with-pyd
# More resources on https://github.com/diffpy/pyobjcryst/issues/33
try:
    from .pyproxsuite_nlp import *
    from .pyproxsuite_nlp import __version__
except ImportError:
    import platform

    if platform.system() == "Windows":
        from .windows_dll_manager import get_dll_paths, build_directory_manager

        with build_directory_manager() as dll_dir_manager:
            for p in get_dll_paths():
                dll_dir_manager.add_dll_directory(p)
            from .pyproxsuite_nlp import *
            from .pyproxsuite_nlp import __version__
    else:
        raise

from . import utils


def __process():
    import sys
    import inspect
    from . import pyproxsuite_nlp

    lib_name = "proxsuite_nlp"

    submodules = inspect.getmembers(pyproxsuite_nlp, inspect.ismodule)
    for mod_info in submodules:
        mod_name = "{}.{}".format(lib_name, mod_info[0])
        sys.modules[mod_name] = mod_info[1]
        mod_info[1].__file__ = pyproxsuite_nlp.__file__
        mod_info[1].__name__ = mod_name


__process()
