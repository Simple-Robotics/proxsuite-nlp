import proxnlp

import numpy as np
from proxnlp import manifolds, constraints
from proxnlp import C2Function
from proxnlp import costs, autodiff


space = manifolds.R() * manifolds.SO2()

cost_weights = np.diag([1.0, 0.0])
target_ = space.neutral()
sq_cost = costs.QuadraticDistanceCost(space, target_, cost_weights)


class cstr_function_prototype(C2Function):
    def __init__(self, space):
        super().__init__(space.nx, space.ndx, 1)

    def __call__(self, x):
        return np.array([x[0] + x[1]]) - 2


inst = cstr_function_prototype(space)
eps = 1e-7
# get C2 function from helper
inst_good = autodiff.FiniteDifferenceHelperC2(space, inst, eps)

eq_cstr = constraints.createEqualityConstraint(inst_good)

prob = proxnlp.Problem(space, sq_cost, [eq_cstr])
ws = proxnlp.Workspace(prob)
rs = proxnlp.Results(prob)

mu_init = 1e-2
tol = 1e-4
solver = proxnlp.ProxNLPSolver(prob, tol, mu_init=mu_init)

x0 = space.neutral()
lam0 = np.zeros([1])
solver.solve(ws, rs, x0, [lam0])

print(rs)
print(rs.xopt)
