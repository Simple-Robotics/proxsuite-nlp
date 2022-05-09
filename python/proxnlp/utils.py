"""
Copyright (C) 2022 LAAS-CNRS, INRIA
"""
import casadi
import proxnlp
import numpy as np

import matplotlib.pyplot as plt


class CasadiFunction(proxnlp.C2Function):
    def __init__(self, nx: int, ndx: int, expression: casadi.SX, cx: casadi.SX, use_hessian: bool = True):
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


_ROOT_10 = 10. ** .5


def plot_pd_errs(ax0: plt.Axes, prim_errs, dual_errs):
    prim_errs = np.asarray(prim_errs)
    dual_errs = np.asarray(dual_errs)
    ax0.plot(prim_errs, c='tab:blue')
    ax0.set_xlabel("Iterations")
    col2 = "tab:orange"
    ax0.plot(dual_errs, c=col2)
    ax0.spines['top'].set_visible(False)
    ax0.spines['right'].set_color(col2)
    ax0.yaxis.label.set_color(col2)
    ax0.set_yscale("log")
    yhigh = ax0.get_ylim()[1]
    ylim = min(np.min(prim_errs[prim_errs > 0]), np.min(dual_errs[dual_errs > 0]))
    ax0.set_ylim(ylim / _ROOT_10, yhigh)
    ax0.legend(["Primal error $p$", "Dual error $d$"])
    ax0.set_title("Solver primal-dual residuals")
