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

A = np.array([[1.3, 0.3], [0.1, 1.0]])
lin_fun = proxsuite_nlp.residuals.LinearFunction(A)
assert np.allclose(A @ target, lin_fun(target))
penalty = constraints.ConstraintObject(lin_fun, constraints.NonsmoothPenaltyL1())

TOL = 1e-6
MAX_ITER = 50


def soft_thresh(x):
    print("CVXPY SOLVE")
    x = cvxpy.Variable(space.ndx, "x")
    e = x - target
    c = 0.5 * cvxpy.quad_form(e, w_x) + cvxpy.norm1(A @ x)
    p = cvxpy.Problem(cvxpy.Minimize(c))
    p.solve(
        solver="OSQP",
        eps_abs=TOL,
        eps_rel=TOL,
        eps_prim_inf=TOL,
        eps_dual_inf=TOL,
        max_iter=MAX_ITER,
        verbose=True,
    )
    print(p.solver_stats)
    extra = p.solver_stats.extra_stats
    print(extra)
    return p.solution.primal_vars[1].copy()


sol = soft_thresh(target)
print("CVXPY SOLUTION:", sol)
print()
x0 = target.copy()
print("x0:", x0)


problem = proxsuite_nlp.Problem(space, rcost, [penalty])

mu_init = 0.01
rho_init = 1e-12
solver = proxsuite_nlp.ProxNLPSolver(problem, TOL, mu_init, rho_init)
solver.max_iters = MAX_ITER
solver.verbose = proxsuite_nlp.VERBOSE
print("Params:", (solver.kkt_system, solver.mul_update_mode))

cb = HistoryCallback()
solver.register_callback(cb)
solver.setup()
xinit = x0.copy()
print("Initial guess: {}".format(xinit))
flag = solver.solve(xinit)
ws = solver.workspace
rs = solver.results
print("FLAG: {}".format(flag))

print(rs)
print("xopt:", rs.xopt)
print("lopt:", rs.lamsopt.tolist())
print("cerrs:", rs.constraint_errs.tolist())
print("cstr_val:")
pprint.pp(ws.cstr_values.tolist(), indent=2)
print("Soft thresh of target:", sol)
print("CORRECT SOLUTION? {}".format(np.allclose(rs.xopt, sol, rtol=TOL, atol=TOL)))

cbstore: HistoryCallback.history_storage = cb.storage
prim_infeas = cbstore.prim_infeas

print("Dual residual:", ws.dual_residuals)
print("active set:", rs.activeset.tolist())


fig: plt.Figure = plt.figure()

gs = fig.add_gridspec(2, 1, height_ratios=[2, 1])
ax = fig.add_subplot(gs[0])

plt.scatter(*sol, s=40, c="cyan", label="solution")
x_hist = np.array(cbstore.xs)
x_hist = np.insert(x_hist, 0, x0, axis=0)
ax: plt.Axes = plt.gca()

for i in range(len(x_hist)):
    x = x_hist[i]
    fontsize = 10
    ax.annotate("{:d}".format(i), x, x + (0, 0.01), fontsize=fontsize)
print("lams:", rs.lamsopt.tolist())

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
plt.ylabel("Linesearch $\\alpha^k$")

plt.show()
