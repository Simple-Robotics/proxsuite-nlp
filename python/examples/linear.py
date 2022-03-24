import numpy as np
from lienlp import LinearResidual

A = np.random.randn(2, 2)
b = np.random.randn(2)
x0 = np.linalg.solve(A, -b)

resdl = LinearResidual(A, b)

assert np.allclose(resdl.computeJacobian(x0), A)
assert np.allclose(resdl(x0), 0.)
assert np.allclose(resdl(np.zeros_like(b)), b)
