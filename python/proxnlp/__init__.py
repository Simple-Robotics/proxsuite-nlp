from .pyproxnlp import *
from .pyproxnlp import __version__

import sys
import inspect

lib_name = 'proxnlp'

submodules = inspect.getmembers(pyproxnlp, inspect.ismodule)
for mod_info in submodules:
    sys.modules["{}.{}".format(lib_name, mod_info[0])] = mod_info[1]
