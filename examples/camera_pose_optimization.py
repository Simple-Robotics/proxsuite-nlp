from proxsuite_nlp import residuals, costs

import proxsuite_nlp
import pinocchio
import numpy as np


np.random.seed(42)
pinocchio.seed(42)

# Reference point we work on
point = np.random.randn(3)
base_fun = residuals.RigidTransformationPointAction(point)
view = np.random.randn(3)
# view = point
lin_op = residuals.LinearFunction(np.eye(3), -view)
resdl = proxsuite_nlp.compose(lin_op, base_fun)
space: proxsuite_nlp.manifolds.SE3() = base_fun.space
T0 = space.neutral()
print(resdl(T0))


def test_numdiff(eps=1e-6):
    J0 = np.zeros((3, 6))
    J1 = J0.copy()
    e = np.zeros(6)
    for i in range(6):
        e[i] = eps
        J0[:, i] = (base_fun(space.integrate(T0, e)) - base_fun(T0)) / eps
        e[i] = 0.0

    print(J0)
    base_fun.computeJacobian(T0, J1)
    print(J1)
    assert np.allclose(J0, J1)


test_numdiff()

total_cost: costs.CostSum = costs.QuadraticDistanceCost(
    space, T0, 1e-3 * np.eye(space.ndx)
) + costs.QuadraticResidualCost(resdl, np.eye(3))

assert total_cost.num_components == 2

print(total_cost.call(T0))


problem = proxsuite_nlp.Problem(space, total_cost, [])
solver = proxsuite_nlp.ProxNLPSolver(problem, 1e-4, 0.01, verbose=proxsuite_nlp.VERBOSE)
solver.setup()

flag = solver.solve(T0)
res = solver.getResults()
print("Converged:", flag)
print(res)

xopt = res.xopt
Mopt = pinocchio.XYZQUATToSE3(xopt)
print("Optimized pose:\n", Mopt, sep="")

print(resdl(xopt))
