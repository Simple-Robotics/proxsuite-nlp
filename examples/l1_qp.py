"""
Solve an L1-penalized QP using PROXNLP.
"""
import proxnlp
import numpy as np

from proxnlp import manifolds, costs, constraints
from proxnlp.helpers import HistoryCallback

import matplotlib.pyplot as plt
import pprint


space = manifolds.R2()
target = np.random.randn(space.nx)
w_x_scalar = 1.1
w_x = np.eye(space.ndx) * w_x_scalar
print("TARGET:", target)
print("weights:\n{}".format(w_x))
rcost = costs.QuadraticDistanceCost(space, target, w_x)

A = np.eye(space.ndx)
lin_fun = proxnlp.residuals.LinearFunction(A)
assert np.allclose(A @ target, lin_fun(target))
low = np.array([-0.5, -1.0])
# pset = constraints.BoxConstraint(low, -low)
# pset = constraints.NegativeOrthant()
pset = constraints.L1Penalty()
penalty = constraints.ConstraintObject(lin_fun, pset)

problem = proxnlp.Problem(space, rcost, [penalty])

tol = 1e-4
mu_init = 0.01
solver = proxnlp.Solver(problem, tol, mu_init)
solver.verbose = proxnlp.VERBOSE
solver.max_iters = 40

ws = proxnlp.Workspace(problem)
rs = proxnlp.Results(problem)

x0 = target.copy()
cb = HistoryCallback()
solver.register_callback(cb)
flag = solver.solve(ws, rs, x0)
print("FLAG:", flag)

print(rs)
print("xopt:", rs.xopt)
print("lopt:", rs.lamsopt.tolist())
print("cerrs:", rs.constraint_errs.tolist())
print("cstr_val:")
pprint.pp(ws.cstr_values.tolist())

cbstore: HistoryCallback.history_storage = cb.storage
prim_infeas = cbstore.prim_infeas
print("Infeas:")
pprint.pp(prim_infeas.tolist())


def soft_thresh(x, lbda):
    mask = np.abs(x) <= lbda
    out = x - lbda * np.sign(x)
    out[mask] = 0.0
    return out


print("Dual residual:", ws.dual_residuals)
print("Jacobian:\n{}".format(ws.jacobians_data))
print("active set:", rs.activeset.tolist())

sol = soft_thresh(target, 1 / w_x_scalar)
print("Soft thresh of target:", sol)


plt.figure()
plt.scatter(*sol, s=40, c="cyan", label="solution")
x_hist = np.array(cbstore.xs)
plt.plot(*x_hist.T, ls="--", marker=".", lw=1.0, zorder=1)
plt.scatter(
    *x_hist[-1], s=20, zorder=2, alpha=0.7, edgecolors="k", c="g", label="$x^*$"
)
plt.scatter(
    *target,
    s=40,
    zorder=1,
    alpha=0.4,
    edgecolors="k",
    c="r",
    label="target $\\bar{x}^0$"
)

plt.legend()
plt.tight_layout()
plt.show()
