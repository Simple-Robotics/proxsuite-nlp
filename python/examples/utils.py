import casadi
import lienlp
import numpy as np


class CasadiFunction(lienlp.C2Function):
    def __init__(self, nx, ndx, expression: casadi.SX, cx: casadi.SX):
        nres = expression.shape[0]
        super().__init__(nx, ndx, nres)
        assert nx == cx.shape[0]
        self.clam = casadi.SX.sym("lam", nres)
        self.expr = expression
        self.Jexpr = casadi.jacobian(expression, cx)
        self.Hexpr = casadi.jacobian(self.clam.T @ self.Jexpr, cx)

        self.fun = casadi.Function("f", [cx], [expression])
        self.Jfun = casadi.Function("Jf", [cx], [self.Jexpr])
        self.Hfun = casadi.Function("Hf", [cx, self.clam], [self.Hexpr])

    def __call__(self, x):
        return np.asarray(self.fun(x)).flatten()

    def computeJacobian(self, x, J):
        J[:] = np.asarray(self.Jfun(x))

    def vectorHessianProduct(self, x, v, H):
        H[:] = np.asarray(self.Hfun(x, v))

