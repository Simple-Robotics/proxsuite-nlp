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
    print("left :", prod.left)
    print("right:", prod.right)

    x0 = prod.neutral()
    x1 = prod.rand()
    assert x0.size == prod.nx
    assert x1.size == prod.nx
    print("x0", x0)
    print("x1", x1)
    print("split(x1):", prod.split(x1).tolist())

    dx0 = prod.tangent_space().rand()
    print("dx0", dx0)
    print("split_vector(dx0):", prod.split_vector(dx0))
    try:
        prod.split_vector(x0)
    except RuntimeError as e:
        print(e)

    prod2 = space1 * space1 * space2
    print(prod2.nx)
    print(prod2.ndx)
    x0 = prod2.neutral()
    print(x0)

    s33 = space3 * space3 * space3
    print("space3^3:", s33.ndx)
    print("space3^3:", s33.neutral())
    print("space3^3:", s33.rand())


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(sys.argv))
