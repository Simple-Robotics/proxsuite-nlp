import numpy as np
import lienlp
from lienlp.residuals import ManifoldDifferenceToPoint
from lienlp.costs import QuadraticDistanceCost, QuadraticResidualCost
from lienlp.manifolds import EuclideanSpace
from lienlp.constraints import NegativeOrthant


nx = 2
p0 = np.array([-.4, .7])
p1 = np.array([1., .5])
space = EuclideanSpace(nx)

radius_ = .6
radius_sq = radius_ ** 2
weights = np.eye(nx)

cost_ = QuadraticDistanceCost(space, p0, weights)
circl_center = space.neutral()
inner_res_ = ManifoldDifferenceToPoint(space, circl_center)
print(cost_(p0))
print(cost_(p1))
print(inner_res_(p0))
print(inner_res_(p1))
w2 = 2 * np.eye(inner_res_.nr)
residual = QuadraticResidualCost(
    inner_res_,
    w2,
    np.zeros(inner_res_.nr),
    -radius_sq
)
print(residual(p0))
print(residual(p1))

cstr1 = NegativeOrthant(residual)
print(cstr1.projection(p0))
print(cstr1.projection(p1))

prob = lienlp.Problem(cost_, [cstr1])
results = lienlp.Results(nx, prob)
workspace = lienlp.Workspace(nx, nx, prob)

mu_init = 0.02
solver = lienlp.Solver(space, prob, mu_init=mu_init)
solver.use_gauss_newton = True
cb = lienlp.HistoryCallback()
solver.register_callback(cb)

lams0 = [np.random.randn(residual.nr)]
solver.solve(workspace, results, p1, lams0)


print("Result x:  ", results.xopt)
print("Target was:", p0)


import matplotlib.pyplot as plt


plt.rcParams["lines.linewidth"] = 1.

xs_ = np.stack(cb.storage.xs.tolist())

ax: plt.Axes = plt.axes()
ax.plot(*xs_.T, marker='o', markersize=3, alpha=.7)
a = plt.Circle(circl_center, radius_, facecolor='r', alpha=.4, edgecolor='k')
print(a)
ax.add_artist(a)
ax.set_aspect('equal')
ax.set_ylim(-.8, .8)
ax.set_xlim(-.6, 1.2)
plt.title("Optimization trajectory")
plt.show()

