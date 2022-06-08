# A nonlinear programming solver on manifolds

This package is a C++11 implementation of a primal-dual augmented Lagrangian-type algorithm for nonlinear optimization on manifolds.

Copyright (C) 2022 LAAS-CNRS, INRIA

## Building from source

Clone this repo using

```bash
git clone [url-to-repo] --recursive
```

Create a build tree using CMake, build and install:

```bash
cd your/checkout/folder/
cmake -S . -B build
cmake --build build/ --config Release --target install
```

### Dependencies

* CMake (with the [JRL CMake modules](https://github.com/jrl-umi3218/jrl-cmakemodules))
* Eigenpy>=2.7.2 ([GitHub](https://github.com/stack-of-tasks/eigenpy) | [conda](https://anaconda.org/conda-forge/eigenpy))
* Pinocchio>=2.9.1
* Eigen>=3.3.7
* [fmtlib](https://github.com/fmtlib/fmt) version 8.1.1
* Boost>=1.71

For easy dependency management, I suggest using the [conda](https://github.com/conda/conda) package manager.

**Python dependencies:**

* numpy
* matplotlib
* typed-argument-parser
* tqdm
* meshcat-python
* [meshcat-utils](https://gitlab.inria.fr/wjallet/pin-meshcat-utils)

### Notes

* If you want to use Python and develop, I really advise managing your dependencies and environment using [conda](https://github.com/conda/conda). To build against a Conda environment, activate the environment and add `export CMAKE_PREFIX_PATH=$CONDA_PREFIX` before running CMake.
* To build the documentation:

    ```bash
    cd build/
    make doc
    ```
