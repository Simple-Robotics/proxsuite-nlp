"""
Solve an L1-penalized QP using PROXNLP.
"""
import proxnlp
import numpy as np
import cvxpy

from proxnlp import manifolds, costs, constraints
from proxnlp.helpers import HistoryCallback

import matplotlib.pyplot as plt
import pprint


np.random.seed(10)

space = manifolds.R2()
target = np.array([1.2, -0.3])
w_x = np.eye(space.ndx) * 1.0
print("TARGET:", target)
print("weights:\n{}".format(w_x))
rcost = costs.QuadraticDistanceCost(space, target, w_x)

A = np.eye(space.ndx)
lin_fun = proxnlp.residuals.LinearFunction(A)
assert np.allclose(A @ target, lin_fun(target))
uppr = np.array([1.0, 0.4])
lowr = np.array([0.0, -0.1])
# pset = constraints.BoxConstraint(lowr, uppr)
pset = constraints.NonsmoothPenaltyL1()
penalty = constraints.ConstraintObject(lin_fun, pset)


def soft_thresh(x):
    x = cvxpy.Variable(space.ndx, "x")
    e = x - target
    c = 0.5 * cvxpy.quad_form(e, w_x) + cvxpy.norm1(x)
    p = cvxpy.Problem(cvxpy.Minimize(c))
    p.solve()
    return p.solution.primal_vars[1]


sol = soft_thresh(target)
print("ANALYTICAL SOLUTION:", sol)
print()
x0 = target.copy()
print("x0:", x0)


problem = proxnlp.Problem(space, rcost, [penalty])

tol = 1e-6
mu_init = 0.001
rho_init = 0.0
solver = proxnlp.Solver(problem, tol, mu_init, rho_init)
solver.verbose = proxnlp.VERBOSE
solver.mul_update_mode = proxnlp.MUL_PRIMAL
solver.setDualPenalty(0.0)
solver.max_iters = 20
solver.reg_init = 0.1

cb = HistoryCallback()
solver.register_callback(cb)
solver.setup()
flag = solver.solve(x0)
ws = solver.getWorkspace()
rs = solver.getResults()
print("FLAG:", flag)

print(rs)
print("xopt:", rs.xopt)
print("lopt:", rs.lamsopt.tolist())
print("cerrs:", rs.constraint_errs.tolist())
print("cstr_val:")
pprint.pp(ws.cstr_values.tolist(), indent=2)
print("Soft thresh of target:", sol)

cbstore: HistoryCallback.history_storage = cb.storage
prim_infeas = cbstore.prim_infeas
print("Infeas:")
pprint.pp(prim_infeas.tolist())

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
