"""
L1-penalized QP.
"""
import proxnlp
import numpy as np

from proxnlp import manifolds, costs, constraints
from proxnlp.helpers import HistoryCallback


space = manifolds.R2()
target = space.rand()
w_x_scalar = 1.0
w_x = np.eye(space.ndx) * w_x_scalar
print("target:", target)
print("weights:\n{}".format(w_x))
rcost = costs.QuadraticDistanceCost(space, target, w_x)

A = np.eye(space.ndx)
lin_fun = proxnlp.residuals.LinearFunction(A)
assert np.allclose(A @ target, lin_fun(target))
penalty = constraints.ConstraintObject(lin_fun, constraints.L1Penalty())

problem = proxnlp.Problem(space, rcost, [penalty])

tol = 1e-6
mu_init = 0.01
solver = proxnlp.Solver(problem, tol, mu_init)
solver.verbose = proxnlp.VERBOSE

ws = proxnlp.Workspace(problem)
rs = proxnlp.Results(problem)

x0 = space.neutral()
cb = HistoryCallback()
solver.register_callback(cb)
flag = solver.solve(ws, rs, x0)
print("FLAG:", flag)

print(rs)
print("xopt:", rs.xopt)

cbstore: HistoryCallback.history_storage = cb.storage
for x in cbstore.xs:
    print(x)


def soft_thresh(x, lbda):
    mask = np.abs(x) <= lbda
    out = x - lbda * np.sign(x)
    out[mask] = 0.0
    return out


print("Soft thresh of target:", soft_thresh(target, w_x_scalar))
