"""
Copyright (C) 2022 LAAS-CNRS, INRIA
"""
import pytest
import numpy as np

from proxnlp.manifolds import SO2, SO3, SE2, CartesianProduct
from proxnlp import autodiff, residuals


def test_cartesian_product():
    space1 = SO2()
    space2 = SO3()
    space3 = SE2()

    prod = CartesianProduct(space1, space2)
    assert prod.nx == (space1.nx + space2.nx)
    assert prod.ndx == (space1.ndx + space2.ndx)

    print(prod.nx, prod.ndx)
    print(prod.getComponent(0), prod.getComponent(1))
    assert prod.num_components == 2

    x0 = prod.neutral()
    x1 = prod.rand()
    assert x0.size == prod.nx
    assert x1.size == prod.nx

    dx0 = prod.tangent_space().rand()
    x2 = prod.integrate(x0, dx0)
    try:
        prod.split_vector(x0)
        prod.split(dx0)
    except RuntimeError as e:
        print(e)

    x0_spl = prod.split(x0)
    dx0_spl = prod.split_vector(dx0)
    x2_spl = prod.split(x2)
    print(x0_spl.tolist(), "<< x0_spl")
    print(dx0_spl.tolist(), "<< dx0_spl")
    print(dx0, "<< dx0")
    print(x2, "<< x2")
    print(x2_spl.tolist(), "<< x2_spl")
    for i in range(prod.num_components):
        c = prod.getComponent(i)
        _x2_i = c.integrate(x0_spl[i], dx0_spl[i])
        assert np.allclose(x2_spl[i], _x2_i)

    d0 = prod.difference(x0, x1)
    print(d0)
    assert d0.size == prod.ndx

    # prod2
    prod2 = space1 * space1 * space2
    print("prod2:")
    assert prod2.num_components == 3
    print(prod2.nx)
    print(prod2.ndx)
    x0 = prod2.rand()
    print(x0)
    splitx0 = prod2.split(x0).tolist()
    print(splitx0)
    remerge0 = prod2.merge(splitx0).tolist()
    assert np.allclose(remerge0, x0)

    # modify
    splitx0[0][1] = 1.42
    print("modify:", splitx0)
    print(x0)
    assert np.allclose(prod2.merge(splitx0), x0)

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

    # test derivatives
    def product_space_diff_test(prod):
        x0 = prod.rand()
        target = prod.rand()
        eps = 1e-7
        fun = residuals.ManifoldDifferenceToPoint(prod, target)

        r0 = fun(x0)
        print(r0)
        assert r0.shape[0] == fun.nr

        J_an = fun.getJacobian(x0)

        fun_autodiff = autodiff.FiniteDifferenceHelper(prod, fun, eps)
        J_nd = fun_autodiff.getJacobian(x0)
        print(J_an)
        print(J_nd)

        assert np.allclose(J_an, J_nd)

    product_space_diff_test(space1 * space2)
    product_space_diff_test(s33)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(sys.argv))
