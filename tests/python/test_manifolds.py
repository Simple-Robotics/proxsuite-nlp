"""
Copyright (C) 2022 LAAS-CNRS, INRIA
"""
import pytest

from proxnlp.manifolds import SO2, SO3, SE2, CartesianProduct


def test_cartesian_product():
    space1 = SO2()
    space2 = SO3()
    space3 = SE2()

    prod = CartesianProduct(space1, space2)

    print(prod.nx)
    print(prod.ndx)
    print(prod.nx, prod.ndx)
    print(prod.getComponent(0))
    print(prod.getComponent(1))
    assert prod.num_components == 2

    x0 = prod.neutral()
    x1 = prod.rand()
    assert x0.size == prod.nx
    assert x1.size == prod.nx
    print("x0", x0)
    print("x1", x1)
    print("split(x1):", prod.split(x1).tolist())

    dx0 = prod.tangent_space().rand()
    print("dx0", dx0)
    print("split_vector(dx0):", prod.split_vector(dx0).tolist())
    try:
        prod.split_vector(x0)
    except RuntimeError as e:
        print(e)

    d0 = prod.difference(x0, x1)
    print(d0)
    assert d0.size == prod.ndx

    prod2 = space1 * space1 * space2
    print("prod2:")
    assert prod2.num_components == 3
    print(prod2.nx)
    print(prod2.ndx)
    x0 = prod2.neutral()
    print(x0)
    splitx0 = prod2.split(x0).tolist()
    print(splitx0)

    s32 = space3 * space3
    print(s32.__mul__)
    s33 = space3 * space3 * space3
    x0 = s33.rand()
    print("space3^3:", s33)
    print("numcomp:", s33.num_components)
    print("space3^3:", s33.ndx)
    print("space3^3:", s33.neutral())
    print("space3^3:", x0)
    s33_split = s33.split(x0).tolist()
    assert len(s33_split) == s33.num_components
    print("split s33:", s33_split)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(sys.argv))
