"""
Copyright (C) 2022 LAAS-CNRS, INRIA
"""
from .pyproxnlp import *
from .pyproxnlp import __version__


def __process():
    import sys
    import inspect
    from . import pyproxnlp

    lib_name = 'proxnlp'

    submodules = inspect.getmembers(pyproxnlp, inspect.ismodule)
    for mod_info in submodules:
        sys.modules["{}.{}".format(lib_name, mod_info[0])] = mod_info[1]


__process()
