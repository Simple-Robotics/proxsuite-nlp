# proxnlp: An augmented Lagrangian nonlinear solver on manifolds

`proxnlp` is a C++14 library, implementing a primal-dual augmented Lagrangian-type algorithm for nonlinear optimization on manifolds,
as well as some modelling tools.

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
* [eigenpy](https://github.com/stack-of-tasks/eigenpy)>=3.1.0 | [conda](https://anaconda.org/conda-forge/eigenpy)
* [pinocchio](https://github.com/stack-of-tasks/eigenpy)>=2.9.1 | [conda](https://anaconda.org/conda-forge/pinocchio)
* Eigen>=3.3.7
* [fmtlib](https://github.com/fmtlib/fmt)>=9.1.0, <11
* [Boost](https://www.boost.org/)>=1.71
* [Benchmark](https://github.com/google/benchmark)==1.5

For easy dependency management, I suggest using the [conda](https://github.com/conda/conda) package manager.

**Python dependencies:**

* numpy
* matplotlib
* typed-argument-parser
* tqdm
* meshcat-python

### Notes

* If you want to use Python and develop, I really advise managing your dependencies and environment using [conda](https://github.com/conda/conda). To build against a Conda environment, activate the environment and add `export CMAKE_PREFIX_PATH=$CONDA_PREFIX` before running CMake.
* To build the documentation:

    ```bash
    cd build/
    make doc
    ```

## Credits

The following people have been involved in the development of **proxsuite-nlp** and are warmly thanked for their contributions:

-   [Wilson Jallet](https://github.com/ManifoldFR) (LAAS-CNRS/Inria): main developer and manager of the project
-   [Sarah El Kazdadi](https://github.com/sarah-ek) (Inria): linear algebra modules developer
-   [Fabian Schramm](https://github.com/fabinsch) (Inria): core developper
-   [Joris Vaillant](https://github.com/jorisv) (Inria): core developer

## Acknowledgments

The development of **proxsuite-nlp** is actively supported by the [Willow team](https://www.di.ens.fr/willow/) [@INRIA](http://www.inria.fr) and the [Gepetto team](http://projects.laas.fr/gepetto/) [@LAAS-CNRS](http://www.laas.fr).
