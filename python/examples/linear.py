import numpy as np
import lienlp
from lienlp.residuals import LinearResidual
from lienlp.costs import QuadraticResidualCost
from lienlp import EqualityConstraint, NegativeOrthant, Problem

nx = 3
A = np.random.randn(2, nx)
b = np.random.randn(2)
x0 = np.linalg.lstsq(A, -b)[0]
x1 = np.linalg.lstsq(A, np.abs(b) * .1)[0]

resdl = LinearResidual(A, b)

print("x0:", x0, "resdl(x0):", resdl(x0))
print("x1:", x1, "resdl(x1):", resdl(x1))
assert np.allclose(resdl.computeJacobian(x0), A)
assert np.allclose(resdl(x0), 0.)
assert np.allclose(resdl(np.zeros_like(x0)), b)

print("Residual nx :", resdl.nx)
print("Residual ndx:", resdl.ndx)
print("Residual nr :", resdl.nr)


cstr1 = EqualityConstraint(resdl)
cstr2 = NegativeOrthant(resdl)

print(cstr1.projection(x0), "should be zero")
print(cstr1.normalConeProjection(x0), "should be x0")

print("proj  x0:", cstr2.projection(x0))
print("proj  x1:", cstr2.projection(x1))
print("dproj x1:", cstr2.normalConeProjection(x1))


cost_ = QuadraticResidualCost(resdl, np.eye(nx))
problem = Problem(cost_)
# problem = Problem(cost_, [cstr2])
