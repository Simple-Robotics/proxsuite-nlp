# A nonlinear programming solver on Lie groups

This package is a C++ implementation of a primal-dual augmented Lagrangian-type algorithm for nonlinear
optimization on manifolds.

Copyright (C) 2022 LAAS-CNRS / Inria

## Building from source

Clone this repo using

```bash
git clone [url-to-repo] --recursive
```

**Dependencies**:

* CMake (with the [JRL CMake modules](https://github.com/jrl-umi3218/jrl-cmakemodules))
* Eigenpy >=2.7.1 ([GitHub](https://github.com/stack-of-tasks/eigenpy) | [conda](https://anaconda.org/conda-forge/eigenpy))
* Pinocchio 3
* Eigen >=3.4.0 (for now)
* [fmtlib](https://github.com/fmtlib/fmt)
* Boost >=1.71

Building:

```bash
cmake -S . -B build
cmake --build build/
```
