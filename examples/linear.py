"""
Copyright (C) 2022 LAAS-CNRS, INRIA
"""
import numpy as np
import proxnlp
from proxnlp.residuals import LinearFunction
from proxnlp.costs import QuadraticDistanceCost
from proxnlp.manifolds import EuclideanSpace
from proxnlp.constraints import createEqualityConstraint, createInequalityConstraint

import matplotlib.pyplot as plt

nx = 2
np.random.seed(42)
space = EuclideanSpace(nx)
nres = 2
A = np.random.randn(nres, nx)
b = np.random.randn(nres)
x0 = np.linalg.lstsq(A, -b)[0]
x1 = np.random.randn(nx) * 3
v0 = np.random.randn(nres) * 2

resdl = LinearFunction(A, b)
assert resdl.nx == nx
assert resdl.ndx == nx
assert resdl.nr == nres

print("x0:", x0, "resdl(x0):", resdl(x0))
print("x1:", x1, "resdl(x1):", resdl(x1))
J1 = np.zeros((nres, nx))
J2 = np.zeros((nres, nx))
resdl.computeJacobian(x0, J1)
print(A)
print(J1)
assert np.allclose(J1, A)
assert np.allclose(resdl(x0), 0.0)
assert np.allclose(resdl(np.zeros_like(x0)), b)

print("Residual nx :", resdl.nx)
print("Residual ndx:", resdl.ndx)
print("Residual nr :", resdl.nr)


cstr1 = createEqualityConstraint(resdl)
cstr2 = createInequalityConstraint(resdl)

# DEFINE A PROBLEM AND SOLVE IT
x_target = np.random.randn(nx) * 10
cost_ = QuadraticDistanceCost(space, x_target, np.eye(nx))
problem = proxnlp.Problem(cost_)
print("Problem:", problem)
print("Target :", x_target)
problem = proxnlp.Problem(cost_, [cstr1])


results = proxnlp.Results(problem)
workspace = proxnlp.Workspace(problem)


class DumbCallback(proxnlp.helpers.BaseCallback):
    def __init__(self):
        pass

    def call(self):
        print("Calling dumb callback!")


cb = proxnlp.helpers.HistoryCallback()
cb2 = DumbCallback()

solver = proxnlp.Solver(space, problem)
solver.register_callback(cb)
# solver.register_callback(cb2)
x_init = np.random.randn(nx) * 10
lams0 = [np.random.randn(resdl.nr)]
solver.solve(workspace, results, x_init, lams0)

solver.clear_callbacks()

print(" values:\n", cb.storage.values.tolist())

plt.rcParams["lines.linewidth"] = 1.0

xs_ = np.stack(cb.storage.xs.tolist())

plt.subplot(121)
plt.plot(*xs_.T, ls="--", marker=".", markersize=5)
for i, x in enumerate(xs_):
    plt.annotate(
        "$x_{{{}}}$".format(i),
        x,
        color="b",
        xytext=(10, 10),
        textcoords="offset pixels",
    )
plt.scatter(
    *x_target,
    label="target $\\bar{x}$",
    facecolor=(0.8, 0, 0, 0.5),
    edgecolors="k",
    zorder=2
)
plt.legend()

plt.title("Iterates")
plt.subplot(122)
values_ = cb.storage.values.tolist()
plt.plot(range(1, len(values_) + 1), cb.storage.values.tolist())
plt.xlabel("Iterate")
plt.yscale("log")
plt.title("Problem cost")

plt.show()
