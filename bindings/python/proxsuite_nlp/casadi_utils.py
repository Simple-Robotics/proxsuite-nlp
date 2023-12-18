"""
Copyright (C) 2022 LAAS-CNRS, INRIA
"""
import casadi
import proxsuite_nlp
import numpy as np


class CasadiFunction(proxsuite_nlp.C2Function):
    def __init__(
        self,
        nx: int,
        ndx: int,
        expression: casadi.SX,
        cx: casadi.SX,
        use_hessian: bool = True,
    ):
        nres = expression.shape[0]
        super().__init__(nx, ndx, nres)
        assert nx == cx.shape[0]
        dx = casadi.SX.sym("dx", ndx)
        # TODO: replace this using manifold operation
        xplus = cx + dx
        self.clam = casadi.SX.sym("lam", nres)
        self.expr = casadi.substitute(expression, cx, xplus)
        self.Jexpr = casadi.jacobian(self.expr, dx)
        self.use_hessian = use_hessian
        if use_hessian:
            self.Hexpr = casadi.jacobian(self.clam.T @ self.Jexpr, dx)
        else:
            self.Hexpr = casadi.SX.zeros(ndx, ndx)

        self.fun = casadi.Function("f", [cx, dx], [self.expr])
        self.Jfun = casadi.Function("Jf", [cx, dx], [self.Jexpr])
        self.Hfun = casadi.Function("Hf", [cx, dx, self.clam], [self.Hexpr])
        self._zero = np.zeros(ndx)

    def __call__(self, x):
        return np.asarray(self.fun(x, self._zero)).flatten()

    def computeJacobian(self, x, J):
        J[:] = np.asarray(self.Jfun(x, self._zero))

    def vectorHessianProduct(self, x, v, H):
        H[:, :] = np.asarray(self.Hfun(x, self._zero, v))
