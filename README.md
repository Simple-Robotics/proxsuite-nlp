> [!IMPORTANT]  
> This repository is archived. Development on nonlinear solvers for control (on Lie groups for robotics) continues on the [aligator](https://github.com/Simple-Robotics/aligator) repository.
>
> At time of archiving, the dense nonlinear solver in this repository is not ported over to aligator. This might change in the future if there is demand for it (and especially for a sparse Lie group NLP solver).


# proxsuite-nlp: a package for nonlinear optimization on manifolds [ARCHIVED]

**proxsuite-nlp** is a C++ library, implementing a primal-dual augmented Lagrangian-type algorithm for nonlinear optimization on manifolds, as well as some modelling tools.

## Installation

### From Conda

From [our channel](https://anaconda.org/simple-robotics/proxsuite-nlp)

```bash
conda install -c simple-robotics proxsuite-nlp
```

### From source with Pixi

To build **proxsuite-nlp** from source the easiest way is to use [Pixi](https://pixi.sh/latest/#installation).

[Pixi](https://pixi.sh/latest/) is a cross-platform package management tool for developers that
will install all required dependencies in `.pixi` directory.
It's used by our CI agent so you have the guarantee to get the right dependencies.

Run the following command to install dependencies, configure, build and test the project:

```bash
pixi run test
```

The project will be built in the `build` directory.
You can run `pixi shell` and build the project with `cmake` and `ninja` manually.

### From source

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

**Dependencies**

* CMake (with the [JRL CMake modules](https://github.com/jrl-umi3218/jrl-cmakemodules))
* Eigen>=3.3.7
* [fmtlib](https://github.com/fmtlib/fmt)>=9.1.0, <11
* [Boost](https://www.boost.org/)>=1.71
* (optional) [eigenpy](https://github.com/stack-of-tasks/eigenpy)>=3.2.0 | [conda](https://anaconda.org/conda-forge/eigenpy) (Python bindings)
* (optional) [pinocchio](https://github.com/stack-of-tasks/pinocchio) | [conda](https://anaconda.org/conda-forge/pinocchio)
* a C++-14 compliant compiler

**Python dependencies:**

* numpy
* matplotlib
* typed-argument-parser
* meshcat-python

### Notes

* To build against a Conda environment, activate the environment and add `export CMAKE_PREFIX_PATH=$CONDA_PREFIX` before running CMake.
* To build the documentation:

    ```bash
    cd build/
    make doc
    ```

## Credits

The following people have been involved in the development of **proxsuite-nlp** and are warmly thanked for their contributions:

* [Wilson Jallet](https://github.com/ManifoldFR) (LAAS-CNRS/Inria): main developer and manager of the project
* [Sarah El Kazdadi](https://github.com/sarah-ek) (Inria): linear algebra modules developer
* [Fabian Schramm](https://github.com/fabinsch) (Inria): core developper
* [Joris Vaillant](https://github.com/jorisv) (Inria): core developer
* [Justin Carpentier](https://github.com/jcarpent) (Inria): project coordinator
* [Nicolas Mansard](https://github.com/nmansard) (LAAS-CNRS): project coordinator

## Acknowledgments

The development of **proxsuite-nlp** is actively supported by the [Willow team](https://www.di.ens.fr/willow/) [@INRIA](http://www.inria.fr) and the [Gepetto team](http://projects.laas.fr/gepetto/) [@LAAS-CNRS](http://www.laas.fr).
