import numpy as np
import lienlp
from lienlp.residuals import ManifoldDifferenceToPoint
from lienlp.costs import QuadraticDistanceCost, QuadraticResidualCost
from lienlp.manifolds import EuclideanSpace
from lienlp.constraints import NegativeOrthant, EqualityConstraint


nx = 2
p0 = np.array([-.4, .7])
p1 = np.array([1., .5])
space = EuclideanSpace(nx)

radius_ = .6
radius_sq = radius_ ** 2
weights = np.eye(nx)

cost_ = QuadraticDistanceCost(space, p0, weights)
circl_center = space.neutral()
print(cost_(p0))
print(cost_(p1))

res1_in = ManifoldDifferenceToPoint(space, circl_center)
w2 = 2 * np.eye(res1_in.nr)
res1 = QuadraticResidualCost(res1_in, w2,
                             np.zeros(res1_in.nr),
                             -radius_sq)
print(res1_in(p0))
print(res1_in(p1))
r0 = res1(p0)
r1 = res1(p1)
print(r0)
print(r1)

center2 = space.rand()
res2_in = ManifoldDifferenceToPoint(space, center2)
res2 = QuadraticResidualCost(res2_in, w2, np.zeros(res2_in.nr), -radius_sq)

cstr1 = NegativeOrthant(res1)
cstr2 = NegativeOrthant(res2)

print(cstr1.projection(r0))
print(cstr1.projection(r1))

prob = lienlp.Problem(cost_,
                      [
                          cstr1,
                          cstr2
                       ])
results = lienlp.Results(nx, prob)
workspace = lienlp.Workspace(nx, nx, prob)

mu_init = 0.02
solver = lienlp.Solver(space, prob, mu_init=mu_init, rho_init=1e-8, verbose=True, alpha_min=1e-7)
solver.use_gauss_newton = True
cb = lienlp.HistoryCallback()
solver.register_callback(cb)

lams0 = [np.random.randn(res1.nr)]
solver.solve(workspace, results, p1, lams0)


print("Result x:  ", results.xopt)
print("Target was:", p0)


import matplotlib.pyplot as plt


plt.rcParams["lines.linewidth"] = 1.

xs_ = np.stack(cb.storage.xs.tolist())

bound_xs = (np.min(xs_[:, 0]), np.max(xs_[:, 0]),
            np.min(xs_[:, 1]), np.max(xs_[:, 1]))
# left,right,bottom,up

ax: plt.Axes = plt.axes()
ax.plot(*xs_.T, marker='o', markersize=3, alpha=.7)
ar_c1 = plt.Circle(res1_in.target, radius_, facecolor='r', alpha=.4, edgecolor='k')
ar_c2 = plt.Circle(res2_in.target, radius_, facecolor='b', alpha=.4, edgecolor='k')
print(ar_c1)
ax.add_patch(ar_c1)
ax.add_patch(ar_c2)
ax.set_aspect('equal')

xlims = ax.get_xlim()
xlims = (min(bound_xs[0], xlims[0]), max(bound_xs[1], xlims[1]))
ylims = ax.get_ylim()
ylims = (min(bound_xs[2], ylims[0]), max(bound_xs[3], ylims[1]))
ax.set_xlim(*xlims)
ax.set_ylim(*ylims)
plt.title("Optimization trajectory")
plt.show()

