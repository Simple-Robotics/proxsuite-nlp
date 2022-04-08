"""
Copyright (C) LAAS-CNRS / INRIA

@file   In this file, we show how to overload function objects from lienlp.
"""
import lienlp
import numpy as np
import casadi as cas

from lienlp.constraints import NegativeOrthant
from lienlp.manifolds import EuclideanSpace


# 1: Overload a base function (not differentiable)

class MyFunction(lienlp.BaseFunction):
    def __init__(self, nx, ndx, nr):
        super().__init__(nx, ndx, nr)

    def __call__(self, x):
        return np.log(np.abs(x - 1.))


nx = 2
nr = 2
fun1 = MyFunction(nx, nx, nr)
assert fun1.nx == nx
assert fun1.ndx == nx
assert fun1.nr == nr

x0 = np.random.randn(nx)
print(fun1(x0))
assert np.allclose(fun1(x0), np.log(np.abs(x0 - 1.)))


# 2: Overload a twice-differentiable function


class DiffFunc(lienlp.C2Function):
    def __init__(self, nx, ndx):
        x = cas.SX.sym("x", nx)
        dx = cas.SX.sym("dx", ndx)

        x_dx = x + dx
        x_dx = cas.sqrt(x_dx ** 2 + 1e-10)
        self.expr = cas.sum1(x_dx * cas.log(x_dx))
        self.Jexpr = cas.jacobian(self.expr, dx)

        nr = self.expr.shape[0]
        super().__init__(nx, ndx, nr)

        self._zer = np.zeros(ndx)

        self.fun = cas.Function("f", [x, dx], [self.expr])
        self.Jfun = cas.Function("Jf", [x, dx], [self.Jexpr])

        cv = cas.SX.sym("v", nr)
        self.Hexpr = cas.jacobian(cv.T @ self.Jexpr, dx)
        self.Hfun = cas.Function("Hf", [x, cv, dx], [self.Hexpr])

    def __call__(self, x):
        return np.asarray(self.fun(x, self._zer)).flatten()

    def computeJacobian(self, x, J):
        J[:] = np.asarray(self.Jfun(x, self._zer))

    def vectorHessianProduct(self, x, v, H):
        H[:] = np.asarray(self.Hfun(x, v, self._zer))


fun2 = DiffFunc(nx, nx)
nr = 1
space = EuclideanSpace(nx)
x1 = space.rand()
cstr = NegativeOrthant(fun2)

print("Diff func:")
print("x0:  ", x0)
print("eval:", fun2(x0))
v0 = np.ones(nr)
J_ = np.zeros((nr, nx))
H_ = np.zeros((nx, nx))
fun2.computeJacobian(x0, J_)
print(f"Jacobian:\n{J_}")

fun2.vectorHessianProduct(x0, v0, H_)
print(f"Hessian: \n{H_}")
print(fun2.nx, fun2.ndx, fun2.nr)

cs2 = lienlp.costs.QuadraticDistanceCost(space, x0, np.eye(nx))
cs = lienlp.costs.CostFromFunction(fun2)
grad = np.zeros(nx)
cH = np.zeros((nx, nx))


def plot_fun():
    import matplotlib.pyplot as plt

    Ngx = 31
    xg = np.linspace(-1., 2., Ngx)
    grid = np.stack(np.meshgrid(xg, xg))

    zvals = np.stack([fun2(u) for u in grid.swapaxes(0, -1).reshape(-1, 2)])
    zvals = zvals.reshape(Ngx, Ngx)
    cs_ = plt.contour(grid[0], grid[1], zvals, levels=20)
    plt.colorbar(cs_)
    plt.show()
    print("shape:", zvals.shape)


plot_fun()

prob = lienlp.Problem(cs)
ws = lienlp.Workspace(nx, nx, prob)
rs = lienlp.Results(nx, prob)

solver = lienlp.Solver(space, prob, rho_init=10.)
flag = solver.solve(ws, rs, x0, rs.lamsopt)

print("Flag:", flag)

print("xopt:   ", rs.xopt)
