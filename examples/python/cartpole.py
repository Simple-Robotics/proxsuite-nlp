import proxnlp
import numpy as np

import pinocchio as pin
import pinocchio.casadi as cpin
import casadi as cas

from meshcat_utils import display_trajectory, ForceDraw

from proxnlp import manifolds
from proxnlp.constraints import EqualityConstraint
from proxnlp.utils import CasadiFunction

from .cartpole_utils import make_cartpole


model, coll_model, vis_model = make_cartpole(False)
data = model.createData()
vizer = pin.visualize.MeshcatVisualizer(model, coll_model, vis_model, data)
vizer.initViewer(loadModel=True)
vizer.viewer.open()

xspace = manifolds.MultibodyPhaseSpace(model)

xtarget = xspace.neutral()
x0 = xtarget.copy()
x0[1] = np.pi

vizer.display(x0[:model.nq])

nq = model.nq
nv = model.nv
dt = 0.01
Tf = 1.
nsteps = int(Tf / dt)
B = np.array([[1.], [0.]])
nu = B.shape[1]


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
        resdls.append(cxs[t + 1] - xnext)
    expression = cas.vertcat(*resdls)
    return expression


class ProblemDef():
    def __init__(self):
        self.cmodel = cpin.Model(model)
        self.cdata = self.cmodel.createData()

        cxs = [cas.SX.sym("x%i" % i, xspace.nx) for i in range(nsteps + 1)]
        cus = [cas.SX.sym("u%i" % i, nu) for i in range(nsteps)]
        cX_s = cas.vertcat(*cxs)
        cU_s = cas.vertcat(*cus)

        cXU_s = cas.vertcat(cX_s, cU_s)

        w_u = 1e-2
        w_term = 2e-1 * np.ones(xspace.ndx)
        w_term[2:] = 0.
        ferr = cxs[nsteps] - xtarget
        cost_expression = (
            0.5 * w_u * dt * cas.dot(cU_s, cU_s) +
            0.5 * cas.dot(ferr, w_term * ferr))

        self.cost_fun = CasadiFunction(pb_space.nx, pb_space.ndx, cost_expression, cXU_s, use_hessian=True)

        x0 = cas.SX(x0)
        expression = make_dynamics_expression(self.cmodel, self.cdata, x0, cxs, cus)
        self.dynamics_fun = CasadiFunction(pb_space.nx, pb_space.ndx, expression, cXU_s, use_hessian=False)

        control_bounds_ = []
        for t in range(nsteps):
            control_bounds_.append(cus[t] - u_bound)
            control_bounds_.append(-cus[t] - u_bound)
        control_expr = cas.vertcat(*control_bounds_)
        self.control_bound_fun = CasadiFunction(pb_space.nx, pb_space.ndx, control_expr, cXU_s, use_hessian=False)
