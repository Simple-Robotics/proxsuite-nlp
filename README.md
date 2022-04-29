# A nonlinear programming solver on manifolds

This package is a C++ implementation of a primal-dual augmented Lagrangian-type algorithm for nonlinear optimization on manifolds.

Copyright (C) 2022 LAAS-CNRS, INRIA

## Building from source

Clone this repo using

```bash
git clone [url-to-repo] --recursive
```

## Dependencies

* CMake (with the [JRL CMake modules](https://github.com/jrl-umi3218/jrl-cmakemodules))
* Eigenpy >=2.7.1 ([GitHub](https://github.com/stack-of-tasks/eigenpy) | [conda](https://anaconda.org/conda-forge/eigenpy))
* Pinocchio 3
* Eigen >=3.4.0 (for now)
* [fmtlib](https://github.com/fmtlib/fmt)
* Boost >=1.71

**Python dependencies:**

* numpy
* matplotlib
* typed-argument-parser
* tqdm
* meshcat-python
* [meshcat-utils](https://gitlab.inria.fr/wjallet/pin-meshcat-utils)

## Building

```bash
cmake -S . -B build
cmake --build build/
```
