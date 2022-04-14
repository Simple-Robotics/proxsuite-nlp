import pinocchio as pin
import pinocchio.casadi as cpin

import numpy as np
import casadi as cas
import lienlp

from lienlp.manifolds import MultibodyPhaseSpace, EuclideanSpace
from examples.utils import CasadiFunction

import example_robot_data as erd


robot = erd.load("double_pendulum")
model = robot.model

Tf = 1.
nsteps = 2
dt = Tf / nsteps
Tf = dt * nsteps
print("Time step: {:.3g}".format(dt))

nq = model.nq
nu = 1
B = np.array([[0.], [1.]])

xspace = MultibodyPhaseSpace(model)
uspace = EuclideanSpace(nsteps * nu)
pb_space = EuclideanSpace(nsteps * nu + (nsteps + 1) * (xspace.nx))
print("Space:", uspace)



def make_dynamics_expression(cmodel: cpin.Model, cdata: cpin.Data, x0, cxs, cus):
    resdls = [cxs[0] - x0]
    for t in range(nsteps):
        q = cxs[t][:nq]
        v = cxs[t][nq:]
        tau = B @ cus[t]
        acc = cpin.aba(cmodel, cdata, q, v, tau)
        vnext = v + dt * acc
        qnext = cpin.integrate(cmodel, q, dt * vnext)
        xnext = cas.vertcat(qnext, vnext)
        resdls.append(xnext - cxs[t + 1])
    expression = cas.vertcat(*resdls)
    return expression, resdls


u_bound = 0.55

class MultipleShootingProblem:
    """Multiple-shooting formulation."""
    def __init__(self, x0, x1):
        self.cmodel = cpin.Model(model)
        self.cdata = self.cmodel.createData()
        cxs = [cas.SX.sym("x%i" % i, xspace.nx) for i in range(nsteps + 1)]
        cus = [cas.SX.sym("u%i" % i, nu) for i in range(nsteps)]
        cX_s = cas.vertcat(*cxs)
        cU_s = cas.vertcat(*cus)
        cXU_s = cas.vertcat(cX_s, cU_s)
        x0 = cas.SX(x0)
        expression, states = make_dynamics_expression(self.cmodel, self.cdata, x0, cxs, cus)
        self.dynamics_fun = CasadiFunction(pb_space.nx, pb_space.ndx, expression, cXU_s)

        w_u = 1e-3
        w_x = 1e-3
        w_term = 1e-2
        ferr = cxs[-1] - x1
        cost_expression = (
            0.5 * w_u * cas.dot(cU_s, cU_s)
            + 0.5 * w_x * cas.dot(cX_s, cX_s)
            + 0.5 * w_term * cas.dot(ferr, ferr)
            )

        self.cost_fun = CasadiFunction(pb_space.nx, pb_space.ndx, cost_expression, cXU_s)

        control_bounds_ = []
        for t in range(nsteps):
            control_bounds_.append(cus[t] - u_bound)
        control_expr = cas.vertcat(*control_bounds_)
        self.control_bound_fun = CasadiFunction(pb_space.nx, pb_space.ndx, control_expr, cXU_s)



x0 = xspace.rand()
x1 = xspace.neutral()


print("Initial:", x0)
print("Final  :", x1)

u_init = pb_space.neutral()
probdef = MultipleShootingProblem(x0, x1)
cost_fun = lienlp.costs.CostFromFunction(probdef.cost_fun)
dyn_cstr = lienlp.constraints.EqualityConstraint(probdef.dynamics_fun)
bound_cstr = lienlp.constraints.NegativeOrthant(probdef.control_bound_fun)

# prob = lienlp.Problem(cost_fun, [dyn_cstr])
prob = lienlp.Problem(cost_fun, [dyn_cstr, bound_cstr])
print("No. of variables  :", pb_space.nx)
print("No. of constraints:", prob.total_constraint_dim)
ws = lienlp.Workspace(pb_space.nx, pb_space.ndx, prob)
rs = lienlp.Results(pb_space.nx, prob)

import ipdb; ipdb.set_trace()
rho_init = 1e-7
solver = lienlp.Solver(pb_space, prob, mu_init=1e-3, rho_init=rho_init, verbose=True)
solver.maxiters = 20
solver.use_gauss_newton = True
flag = solver.solve(ws, rs, u_init, [])

print("Results struct:\n{}".format(rs))

xus_opt = rs.xopt
xs_opt_flat = xus_opt[:(nsteps + 1) * xspace.nx]
us_opt_flat = xus_opt[(nsteps + 1) * xspace.nx:]
us_opt = us_opt_flat.reshape(nsteps, -1)
xs_opt = xs_opt_flat.reshape(nsteps + 1, -1)
qs_opt = xs_opt[:, :model.nq]
print("X shape:", xs_opt.shape)


import matplotlib.pyplot as plt

plt.rcParams['lines.linewidth'] = 1.

times = np.linspace(0., Tf, nsteps + 1)
labels_ = ["$x_{%i}$" % i for i in range(model.nq)]

plt.figure(figsize=(9.6, 4.8))
hlines_style = dict(colors='k', alpha=.7, ls='-', lw=2, zorder=-2)
plt.subplot(121)
plt.plot(times, qs_opt)
plt.hlines(x1[:model.nq], *times[[0, -1]], **hlines_style, label='$\\bar{x}$')
plt.legend()
plt.xlabel("Time $t$")
plt.title("State $x$")

plt.subplot(122)
plt.plot(times[:-1], us_opt)
plt.hlines((-u_bound, u_bound), *times[[0, -2]], **hlines_style)
plt.xlabel("Time $t$")
plt.title("Controls $u$")

plt.show()

