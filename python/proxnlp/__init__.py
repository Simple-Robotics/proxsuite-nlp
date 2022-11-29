"""
Copyright (C) 2022 LAAS-CNRS, INRIA
"""
from .pyproxnlp import *
from .pyproxnlp import __version__

from . import utils


def __process():
    import sys
    import inspect
    from . import pyproxnlp

    lib_name = "proxnlp"

    submodules = inspect.getmembers(pyproxnlp, inspect.ismodule)
    for mod_info in submodules:
        mod_name = "{}.{}".format(lib_name, mod_info[0])
        sys.modules[mod_name] = mod_info[1]
        mod_info[1].__file__ = pyproxnlp.__file__
        mod_info[1].__name__ = mod_name


__process()
