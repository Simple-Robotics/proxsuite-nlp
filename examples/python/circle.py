"""
Copyright (C) 2022 LAAS-CNRS, INRIA
"""
import numpy as np
import proxnlp
from proxnlp.residuals import ManifoldDifferenceToPoint
from proxnlp.costs import QuadraticDistanceCost, QuadraticResidualCost
from proxnlp.manifolds import VectorSpace
from proxnlp.constraints import create_inequality_constraint

import matplotlib.pyplot as plt

nx = 2
p0 = np.array([.7, .2])
p1 = np.array([1., .5])
space = VectorSpace(nx)

radius = .6
radius_sq = radius ** 2
weights = np.eye(nx)


# A_mat = np.array([[0., 1.],
#                   [0., 1.]])
# b = np.array([0., -0.2])
# reslin = proxnlp.residuals.LinearFunction(A_mat, b)


cost_ = QuadraticDistanceCost(space, p0, weights)
center1 = space.neutral()
print(cost_(p0))
print(cost_(p1))

res1_in = ManifoldDifferenceToPoint(space, center1)
nr = res1_in.nr
w2 = 2 * np.eye(nr)
slope_ = np.zeros(nr)
res1 = QuadraticResidualCost(res1_in, w2, slope_, -radius_sq)
print(res1_in(p0))
print(res1_in(p1))
r0 = res1(p0)
r1 = res1(p1)
print(r0)
print(r1)

np.random.seed(42)
center2 = np.random.randn(2)
res2_in = ManifoldDifferenceToPoint(space, center2)
res2 = QuadraticResidualCost(res2_in, w2, slope_, -radius_sq)

center3 = np.array([1., .2])
res3 = QuadraticResidualCost(ManifoldDifferenceToPoint(space, center3), w2, slope_, -radius_sq)

cstrs_ = [
    create_inequality_constraint(res1),
    create_inequality_constraint(res2),
    create_inequality_constraint(res3)
]

prob = proxnlp.Problem(cost_, cstrs_)
results = proxnlp.Results(nx, prob)
workspace = proxnlp.Workspace(nx, nx, prob)

mu_init = 0.05
rho_init = 0.
solver = proxnlp.Solver(space, prob, mu_init=mu_init, rho_init=rho_init,
                        verbose=proxnlp.VERBOSE)
solver.use_gauss_newton = True
callback = proxnlp.helpers.HistoryCallback()
solver.register_callback(callback)

lams0 = [np.zeros(cs.nr) for cs in cstrs_]
solver.solve(workspace, results, p1, lams0)


print("Result x:  ", results.xopt)
print("Target was:", p0)


plt.rcParams["lines.linewidth"] = 1.

xs_ = np.stack(callback.storage.xs.tolist())

bound_xs = (np.min(xs_[:, 0]), np.max(xs_[:, 0]),
            np.min(xs_[:, 1]), np.max(xs_[:, 1]))
# left,right,bottom,up

ax: plt.Axes = plt.axes()
ax.plot(*xs_.T, marker='.', markersize=2, alpha=.7, color='b')
ax.scatter(*xs_[-1], s=12, c='tab:red', marker='o', label="$x^*$", zorder=2)
circ_alpha = 0.3
ar_c1 = plt.Circle(center1, radius, facecolor='r', alpha=circ_alpha, edgecolor='k')
ar_c2 = plt.Circle(center2, radius, facecolor='b', alpha=circ_alpha, edgecolor='k')
ar_c3 = plt.Circle(center3, radius, facecolor='g', alpha=circ_alpha, edgecolor='k')

ax.add_patch(ar_c1)
ax.add_patch(ar_c2)
ax.add_patch(ar_c3)
ax.set_aspect('equal')

ax.scatter(*p0, c='green', marker='o')
ax.text(*p0, "$p_0$")

xlims = ax.get_xlim()
xlims = (min(bound_xs[0], xlims[0]), max(bound_xs[1], xlims[1]))
ylims = ax.get_ylim()
ylims = (min(bound_xs[2], ylims[0]), max(bound_xs[3], ylims[1]))
ax.set_xlim(*xlims)
ax.set_ylim(*ylims)
plt.legend()
plt.title("Optimization trajectory")


it_list = [1, 2, 3, 4, 5, 10, 20, 30]
it_list = [i for i in it_list if i < results.numiters]
for it in it_list:
    ls_alphas = callback.storage.ls_alphas[it].copy()
    ls_values = callback.storage.ls_values[it].copy()
    if len(ls_alphas) == 0:
        continue
    soidx = np.argsort(ls_alphas)
    ls_alphas = ls_alphas[soidx]
    ls_values = ls_values[soidx]
    d1 = callback.storage.d1_s[it]
    plt.figure()
    plt.plot(ls_alphas, ls_values)
    plt.plot(ls_alphas, ls_values[0] + ls_alphas * d1)
    plt.plot(ls_alphas, ls_values[0] + solver.armijo_c1 * ls_alphas * d1, ls='--')
    plt.title("Iteration %d" % it)

plt.show()
