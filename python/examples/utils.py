import casadi
import lienlp
import numpy as np

from pinocchio.visualize import MeshcatVisualizer

import time
import tqdm


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


def display_trajectory(vizer: MeshcatVisualizer, qs, dt: float):
    nsteps = qs.shape[0] - 1
    for t in tqdm.trange(nsteps + 1):
        vizer.display(qs[t])
        time.sleep(dt)


def set_cam_angle(vizer: MeshcatVisualizer, value):
    viewer = vizer.viewer
    path = "/Cameras/default/rotated/<object>"
    viewer[path].set_property("position", value)
