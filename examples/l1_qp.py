"""
Solve an L1-penalized QP using proxsuite_nlp.
"""

import proxsuite_nlp
import numpy as np
import cvxpy

from proxsuite_nlp import manifolds, costs, constraints
from proxsuite_nlp.helpers import HistoryCallback

import matplotlib.pyplot as plt
import pprint


np.random.seed(10)

space = manifolds.R2()
target = np.array([1.2, -0.3])
w_x = np.eye(space.ndx) * 1.0
print("TARGET:", target)
rcost = costs.QuadraticDistanceCost(space, target, w_x)

A = np.array([[1.3, 0.3], [0.0, 1.0]])
lin_fun = proxsuite_nlp.residuals.LinearFunction(A)
assert np.allclose(A @ target, lin_fun(target))
uppr = np.array([1.0, 0.4])
lowr = np.array([0.0, -0.1])
# pset = constraints.BoxConstraint(lowr, uppr)
pset = constraints.NonsmoothPenaltyL1()
penalty = constraints.ConstraintObject(lin_fun, pset)


def soft_thresh(x):
    x = cvxpy.Variable(space.ndx, "x")
    e = x - target
    c = 0.5 * cvxpy.quad_form(e, w_x) + cvxpy.norm1(A @ x)
    p = cvxpy.Problem(cvxpy.Minimize(c))
    p.solve()
    return p.solution.primal_vars[1].copy()


sol = soft_thresh(target)
print("ANALYTICAL SOLUTION:", sol)
print()
x0 = target.copy()
print("x0:", x0)


problem = proxsuite_nlp.Problem(space, rcost, [penalty])

tol = 1e-5
mu_init = 0.01
rho_init = 0.001
solver = proxsuite_nlp.ProxNLPSolver(problem, tol, mu_init, rho_init)
solver.verbose = proxsuite_nlp.VERBOSE
print((solver.kkt_system, solver.mul_update_mode))

cb = HistoryCallback()
solver.register_callback(cb)
solver.setup()
xinit = x0 - (0.1, 1.0)
xinit = space.rand()
# xinit = x0
print("Initial guess: {}".format(xinit))
flag = solver.solve(xinit)
ws = solver.getWorkspace()
rs = solver.getResults()
print("FLAG: {}".format(flag))

print(rs)
print("xopt:", rs.xopt)
print("lopt:", rs.lamsopt.tolist())
print("cerrs:", rs.constraint_errs.tolist())
print("cstr_val:")
pprint.pp(ws.cstr_values.tolist(), indent=2)
print("Soft thresh of target:", sol)
print("CORRECT SOLUTION? {}".format(np.allclose(rs.xopt, sol, rtol=tol, atol=tol)))

cbstore: HistoryCallback.history_storage = cb.storage
prim_infeas = cbstore.prim_infeas

print("Dual residual:", ws.dual_residuals)
print("active set:", rs.activeset.tolist())


fig: plt.Figure = plt.figure()

gs = fig.add_gridspec(2, 1, height_ratios=[2, 1])
ax = fig.add_subplot(gs[0])

if isinstance(pset, constraints.NonsmoothPenaltyL1):
    plt.scatter(*sol, s=40, c="cyan", label="solution")
x_hist = np.array(cbstore.xs)
x_hist = np.insert(x_hist, 0, x0, axis=0)
ax: plt.Axes = plt.gca()

for i in range(len(x_hist)):
    x = x_hist[i]
    fontsize = 10
    ax.annotate("{:d}".format(i), x, x + (0, 0.01), fontsize=fontsize)
print("lams:", rs.lamsopt.tolist())

if isinstance(pset, constraints.BoxConstraint):
    size = uppr - lowr
    rect = plt.Rectangle(lowr, *size, zorder=-2, alpha=0.2)
    ax.add_patch(rect)
    lams_flat = np.array(rs.lamsopt).flatten()
    # check complementarity
    lams_flat *= 0.3 / np.linalg.norm(lams_flat)
    ar = plt.arrow(*rs.xopt, *lams_flat, width=0.005)
    arr_mid = rs.xopt + 0.5 * lams_flat
    ax.annotate("$\\lambda^*$", arr_mid, arr_mid + (0, 0.02))

plt.plot(*x_hist.T, ls="--", marker=".", lw=1.0, zorder=1)
plt.scatter(
    *x_hist[-1], s=20, zorder=2, alpha=0.7, edgecolors="k", c="g", label="$x^*$"
)
plt.scatter(
    *target, s=40, zorder=1, alpha=0.4, edgecolors="k", c="r", label="target $x^0$"
)

plt.legend()

ax = fig.add_subplot(gs[1])
plt.sca(ax)

plt.plot(cbstore.alphas, marker=".", ls="--", lw=1.0)
plt.yscale("log", base=2)
plt.tight_layout()

plt.show()
