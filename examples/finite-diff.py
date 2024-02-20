"""
Copyright (C) 2022 LAAS-CNRS, INRIA
"""

import numpy as np
import proxsuite_nlp

from proxsuite_nlp.manifolds import EuclideanSpace
from proxsuite_nlp.autodiff import FiniteDifferenceHelper, FiniteDifferenceHelperC2


def test_fd_one_dim():
    class MyFunction(proxsuite_nlp.BaseFunction):
        def __init__(self, nx):
            super().__init__(nx, nx, nx)

        def __call__(self, x):
            return x**2 - 1

    nx = 1
    space = EuclideanSpace(nx)
    f1 = MyFunction(nx)

    eps = 1e-4
    ATOL = eps**0.5
    f1_diff = FiniteDifferenceHelper(space, f1, eps)
    f1_c2 = FiniteDifferenceHelperC2(space, f1_diff, eps)

    xg = np.linspace(-1, 1, 201)
    xg = xg.reshape(-1, 1)
    f_vals = np.concatenate([f1_diff(x) for x in xg])
    Jf_vals = [f1_diff.getJacobian(x) for x in xg]
    Jf_vals = np.concatenate(Jf_vals)
    print("x:\n{}".format(xg))
    print(f_vals)
    print(Jf_vals)

    Hf_vals = np.concatenate([f1_c2.getVHP(x, np.ones(nx)) for x in xg])

    assert np.allclose(Jf_vals, 2 * xg, atol=ATOL)
    assert np.allclose(Hf_vals, 2 * np.ones_like(Hf_vals), atol=ATOL)
