from .pylienlp import *

import sys, inspect

lib_name = 'lienlp'

submodules = inspect.getmembers(pylienlp, inspect.ismodule)
for mod_info in submodules:
    sys.modules["{}.{}".format(lib_name, mod_info[0])] = mod_info[1]
