"""
Copyright (C) 2022 LAAS-CNRS, INRIA
"""
from .pyproxsuite_nlp import *
from .pyproxsuite_nlp import __version__

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
