"""
Copyright (C) 2022 LAAS-CNRS, INRIA
"""

import proxsuite_nlp
import numpy as np

from proxsuite_nlp import manifolds
from proxsuite_nlp.costs import CostSum, QuadraticDistanceCost
from proxsuite_nlp.utils import plot_pd_errs

import matplotlib.pyplot as plt


space = manifolds.SE2()
p0 = space.neutral()
p0[0] -= 0.4
p1 = space.rand()
p2 = space.rand()
print("p0:", p0)
print("p1:", p1)
print("p2:", p2)

ndx = space.ndx
weights = np.eye(ndx)

dist_to_p0 = QuadraticDistanceCost(space, p0, weights)
dist_to_p1 = QuadraticDistanceCost(space, p1, weights)
dist_to_p2 = QuadraticDistanceCost(space, p2, weights)
Hg0 = np.zeros((ndx, ndx))
Hg1 = np.zeros((ndx, ndx))

dist_to_p0.call(p0)
dist_to_p0.computeGradient(p0)
dist_to_p0.computeHessian(p0, Hg0)

dist_to_p1.call(p0)
dist_to_p1.computeGradient(p0)
dist_to_p1.computeHessian(p0, Hg1)
print("Hg0\n{}".format(Hg0))
print("Hg1\n{}".format(Hg1))

print("scalar * CostFunction:      ", 0.5 * dist_to_p1)
print("CostFunction + CostFunction:", dist_to_p0 + dist_to_p1)


def test_cost_sum():
    sum_1 = CostSum(space.nx, space.ndx)
    print("sum1 init:", sum_1)
    sum_1.add_component(dist_to_p0)
    print("add_comp :", sum_1)
    sum_1 += dist_to_p1
    print("after += :", sum_1)
    sum_1 *= 0.5
    print("after *= :", sum_1)
    print("sum1.weights:", sum_1.weights.tolist())
    Hs_1 = sum_1.computeHessian(p0)
    print("Hessian of sum_1\n{}".format(Hs_1))

    sum_2 = dist_to_p0 + dist_to_p1
    print("sum2 init:", sum_2)
    sum_2 *= 0.5
    print("after *= :", sum_2)
    Hs_2 = sum_2.computeHessian(p0)

    assert np.allclose(Hs_1, 0.5 * (Hg0 + Hg1)), "Should be\n{}".format(
        0.5 * (Hg0 + Hg1)
    )
    assert np.allclose(Hs_2, 0.5 * (Hg0 + Hg1)), "Should be\n{}".format(
        0.5 * (Hg0 + Hg1)
    )

    sum_3 = sum_2 + dist_to_p2
    print("sum3 init:", sum_3)

    problem = proxsuite_nlp.Problem(space, sum_3)
    solver = proxsuite_nlp.ProxNLPSolver(problem, mu_init=0.1)
    callback = proxsuite_nlp.helpers.HistoryCallback()
    solver.setup()
    solver.register_callback(callback)
    flag = solver.solve(p0)
    results = solver.getResults()
    print("Flag:", flag)
    print("xopt:", results.xopt)

    fig: plt.Figure = plt.figure(figsize=(8.4, 4.8))
    ax0: plt.Axes = fig.add_subplot(121)
    prim_errs = callback.storage.prim_infeas
    dual_errs = callback.storage.dual_infeas
    plot_pd_errs(ax0, prim_errs, dual_errs)
    ax: plt.Axes = fig.add_subplot(122)
    plot_pose(p0, ax, "blue", "$p_0$")
    plot_pose(p1, ax, "green", "$p_1$")
    plot_pose(p2, ax, "red", "$p_2$")

    plot_pose(
        results.xopt, ax, "orange", lab=r"$\bar{p} = \mathrm{Bary}(p_0, p_1, p_2)$"
    )

    ax.legend(fontsize=8, ncol=2)
    plt.xlim(-2, 2)
    plt.ylim(-2, 2)
    plt.axis("equal")
    plt.tight_layout()
    plt.show()


def plot_pose(p: np.ndarray, ax: plt.Axes, col, lab=None):
    """Plot a given element p€SE(2)."""
    from matplotlib.transforms import Affine2D

    x, y, cs, ss = p
    theta = np.arctan2(ss, cs)
    deg = np.rad2deg(theta)
    width, height = 1.0, 0.4
    plt.scatter(x, y, c=col, marker="o", label=lab)
    recmid = (x - 0.5 * width, y - 0.5 * height)
    trsf = Affine2D().rotate_deg_around(*(x, y), -deg) + ax.transData
    rect = plt.Rectangle(recmid, width, height, color=col, alpha=0.5)
    rect.set_transform(trsf)
    ax.add_patch(rect)
    return rect


if __name__ == "__main__":
    test_cost_sum()
