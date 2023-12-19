import proxsuite_nlp

import numpy as np
from proxsuite_nlp import manifolds, constraints
from proxsuite_nlp import C2Function
from proxsuite_nlp import costs, autodiff


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

prob = proxsuite_nlp.Problem(space, sq_cost, [eq_cstr])
ws = proxsuite_nlp.Workspace(prob)
rs = proxsuite_nlp.Results(prob)

mu_init = 1e-2
tol = 1e-4
solver = proxsuite_nlp.ProxNLPSolver(prob, tol, mu_init=mu_init)

x0 = space.neutral()
lam0 = np.zeros([1])
solver.solve(ws, rs, x0, [lam0])

print(rs)
print(rs.xopt)
