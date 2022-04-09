import lienlp
import numpy as np

from lienlp import manifolds
from lienlp.costs import CostSum, QuadraticDistanceCost


space = manifolds.SE2()
p0 = space.neutral()
p1 = space.rand()
print("p0:", p0)
print("p1:", p1)

weights = np.eye(space.ndx)

dist_fun0 = QuadraticDistanceCost(space, p0, weights)
dist_fun1 = QuadraticDistanceCost(space, p1, weights)

Hg0 = dist_fun0.computeHessian(p0)
Hg1 = dist_fun1.computeHessian(p0)
print("Hg0\n{}".format(Hg0))
print("Hg1\n{}".format(Hg1))

print("scalar * CostFunction:      ", 0.5 * dist_fun1)
print("CostFunction + CostFunction:", dist_fun0 + dist_fun1)

sum_1 = CostSum(space.nx, space.ndx)
print("sum1 init:", sum_1)
sum_1.add_component(dist_fun0)
print("add_comp :", sum_1)
sum_1 += dist_fun1
print("after += :", sum_1)
sum_1 *= 0.5
print("after *= :", sum_1)
print("sum1.weights:", sum_1.weights.tolist())
Hs_1 = sum_1.computeHessian(p0)
print("Hessian of sum_1\n{}".format(Hs_1))

assert np.allclose(Hs_1, .5 * (Hg0 + Hg1)), "Should be\n{}".format(0.5 * (Hg0 + Hg1))