"""
Inverse kinematics with friction cone constraint.

min Dq, f || q - q0 ||**2 + || f ||**2

subject to  q = pin.integrate(q0, Dq)
            sum(f_i) == weight
            com_z >= avg(p_z) # com height higher than the average height of the feet
            f_i X (com_pos - p_i) == 0 # zero angular momentum at the com

            # The friction cone constraint can be one of the following:
            0)  || f_t ||**2 <= mu**2 * || f_n ||**2   # f_t and f_n are the tangential and orthogonal component of the contact force
            1)  || f @ k.T @ k - k @ k.T @ f ||**2 <= mu **2 k.T @ f @ f.T @ k @k.T @ k    # k is the vector normal to the ground, while f is the vector of contact force
            2)  || f.T @ k ||**2 >= (cos(alpha_k))**2 || f ||**2 * || k ||**2   # here alpha_k is the angle of the friction cone, alpha_k = tan^{-1} mu

Author:
    Alessandro Assirelli
"""

import pinocchio as pin
from pinocchio import casadi as cpin
import casadi
import numpy as np
import example_robot_data as robex
import matplotlib.pyplot as plt

import proxnlp
from proxnlp.manifolds import MultibodyPhaseSpace, VectorSpace
from proxnlp.utils import CasadiFunction

plt.style.use("seaborn")

### LOAD AND DISPLAY SOLO12
robot = robex.load("solo12")
model = robot.model
cmodel = cpin.Model(model)
data = model.createData()
cdata = cmodel.createData()

nq = robot.nq
nv = robot.nv
q0 = robot.q0
x0 = np.concatenate([q0, np.zeros(nv)])
cq0 = casadi.SX(q0)

# We adopt a second-order cone formulation for the friction cone
mu_friction = 0.8
alpha_k = casadi.atan(mu_friction)

try:
    viz = pin.visualize.MeshcatVisualizer(
        robot.model, robot.collision_model, robot.visual_model
    )
    viz.initViewer()
    viz.loadViewerModel()
    viz.display(robot.q0)
except:
    viz = None


def ground(xy):
    return (
        np.sin(xy[0] * 3) / 5
        + np.cos(xy[1] ** 2 * 3) / 20
        + np.sin(xy[1] * xy[0] * 5) / 10
    )


def vizGround(viz, elevation, space, name="ground", color=[1.0, 1.0, 0.6, 1.0]):
    from meshcat_utils import VizUtil

    drawer = VizUtil(viz)
    viz.viewer.open()
    prefix = "bg2"
    xg = np.arange(-1, 1, space)
    xy_g = np.meshgrid(xg, xg)
    xy_g = np.stack(xy_g)
    elev_g = ground(xy_g)
    xyz_g = np.concatenate([xy_g, elev_g[None]], axis=0)
    space = 1e-1
    xyz_g = xyz_g.reshape(3, -1).T
    npts = xyz_g.shape[0]
    colors = np.array([color] * npts)
    assert colors.shape[1] == 4
    drawer.draw_point_cloud(xyz_g, colors, prefix)
    # gv = viz.viewer.gui
    # for i,x in enumerate(np.arange(-1,1,space)):
    #     gv.deleteNode(f'world/pinocchio/visuals/{name}_cx{i}',True)
    #     gv.addCurve(f'world/pinocchio/visuals/{name}_cx{i}',
    #                 [ [x,y,elevation([x,y])] for y in np.arange(-1,1,space) ], color)
    # for i,y in enumerate(np.arange(-1,1,space)):
    #     gv.deleteNode(f'world/pinocchio/visuals/{name}_cy{i}',True)
    #     gv.addCurve(f'world/pinocchio/visuals/{name}_cy{i}',
    #                 [ [x,y,elevation([x,y])] for x in np.arange(-1,1,space) ], color)


if viz is not None:
    vizGround(viz, ground, 0.02)

cxyz = casadi.SX.sym("xyz", 3, 1)

groundNormal = casadi.Function(
    "groundNormal", [cxyz], [casadi.gradient(cxyz[2] - ground(cxyz[:2]), cxyz)]
)

feet = {idx: f.name for idx, f in enumerate(model.frames) if "FOOT" in f.name}
mass = sum([y.mass for y in model.inertias])
grav = np.linalg.norm(model.gravity.linear)


### ---------------------------------------------------------------------- ###

# Build utility functions
cq = casadi.SX.sym("q", model.nq, 1)
cdq = casadi.SX.sym("dq", model.nv, 1)

cpin.framesForwardKinematics(cmodel, cdata, cq)
feet = {
    idx: casadi.Function(f"{name}_pos", [cq], [cdata.oMf[idx].translation])
    for idx, name in feet.items()
}
com = casadi.Function("com", [cq], [cpin.centerOfMass(cmodel, cdata, cq)])
integrate = casadi.Function("integrate", [cq, cdq], [cpin.integrate(cmodel, cq, cdq)])


def friction_cone_constraint(q, fs):
    fc_constr = []
    for idx, force in zip(feet.keys(), fs):
        pos = feet[idx](q)
        perp = groundNormal(pos)
        fp = force @ perp.T @ perp - perp @ perp.T @ force
        fc_constr.append(
            fp.T @ fp
            - casadi.power(mu_friction, 2)
            * (perp.T @ force @ force.T @ perp @ perp.T @ perp)
        )

    return casadi.vertcat(*fc_constr)


def equality_constraints(q, fs):
    eq = []
    eq.append(sum(fs) - np.array([0, 0, mass * grav]))

    torque = 0
    for idx, force in zip(feet.keys(), fs):
        pos = feet[idx](q)
        torque += casadi.cross(force, pos - com(q))
    eq.append(torque)
    eq.append(com(q)[:2] - np.array([0.1, 0.2]))

    for f in feet.values():
        eq.append(f(q)[2] - ground(f(q)[:2]))

    return casadi.vertcat(*eq)


def inequality_constraints(q, which_solver):
    ineq = []
    ineq.append(-com(q)[2] + (sum([f(q)[2] for f in feet.values()])) / len(feet))
    if which_solver == "proxnlp":
        ineq.append(-com(q)[2] + (sum([f(q)[2] for f in feet.values()])) / len(feet))
    return casadi.vertcat(*ineq)


def compute_cost(q_ref, fs):
    cost = 0
    cost += sum([f.T @ f for f in fs]) / 10
    cost += casadi.sumsqr(q_ref)
    return cost


### -------------------------------------------------------------------------------------------- ###
### PROBLEM --- IPOPT

TOLERANCE = 1e-6
opti = casadi.Opti()

# Decision variables
# Note that the contact forces are optimization variables
dxs = opti.variable(2 * nv)
dq = dxs[:nv]
q = integrate(robot.q0, dq)
fs = [opti.variable(3) for _ in feet.values()]

q_ref = q[7:] - robot.q0[7:]

cost = compute_cost(q_ref, fs)
opti.subject_to(friction_cone_constraint(q, fs) <= 0)
opti.subject_to(equality_constraints(q, fs) == 0)  # Sum of forces is weight
opti.subject_to(
    inequality_constraints(q, "ipopt") <= 0
)  # Sum of torques around COM is 0


### SOLVE
p_opts = {}
s_opts = {
    "tol": 1.0,
    "dual_inf_tol": TOLERANCE,
    "constr_viol_tol": TOLERANCE,
    "compl_inf_tol": TOLERANCE,
}
opti.minimize(cost)
# set numerical backend and solver options
opti.solver("ipopt", p_opts, s_opts)
sol = opti.solve()
qopt_ipopt = opti.value(q)
dqopt_ipopt = opti.value(dq)
f_opt_ipopt = [opti.value(f) for f in fs]
du_ipopt = opti.stats()["iterations"]["inf_du"]
pr_ipopt = opti.stats()["iterations"]["inf_pr"]


if viz is not None:
    viz.display(qopt_ipopt)


### Problem PROXNLP

xspace = MultibodyPhaseSpace(model)
pb_space = VectorSpace(4 * 3 + (xspace.ndx))

Dxs = casadi.SX.sym("x_opt_0", xspace.ndx)
Dqs = Dxs[:nv]
q = cpin.integrate(cmodel, cq0, Dqs)
fs = [casadi.SX.sym("f_opt" + str(_), 3) for _ in feet.values()]

F_s = casadi.vertcat(*fs)
XF_s = casadi.vertcat(Dxs, F_s)

q_ref = q[7:] - q0[7:]

cost = compute_cost(q_ref, fs)
cost_fun = CasadiFunction(pb_space.nx, pb_space.ndx, cost, XF_s, use_hessian=True)

fc_fun = CasadiFunction(
    pb_space.nx, pb_space.ndx, friction_cone_constraint(q, fs), XF_s, use_hessian=True
)
eq_fun = CasadiFunction(
    pb_space.nx, pb_space.ndx, equality_constraints(q, fs), XF_s, use_hessian=False
)
ineq_fun = CasadiFunction(
    pb_space.nx,
    pb_space.ndx,
    inequality_constraints(q, "proxnlp"),
    XF_s,
    use_hessian=True,
)

### Solver setup

cost_fun_ = proxnlp.costs.CostFromFunction(cost_fun)
eq_constr1_ = proxnlp.constraints.create_equality_constraint(eq_fun)
ineq_constr1_ = proxnlp.constraints.create_inequality_constraint(fc_fun)
ineq_constr2_ = proxnlp.constraints.create_inequality_constraint(ineq_fun)

constraints = []
constraints.append(eq_constr1_)
constraints.append(ineq_constr1_)
constraints.append(ineq_constr2_)

prob = proxnlp.Problem(cost_fun_, constraints)

print("No. of variables  :", pb_space.nx)
print("No. of constraints:", prob.total_constraint_dim)
workspace = proxnlp.Workspace(pb_space.nx, pb_space.ndx, prob)
results = proxnlp.Results(pb_space.nx, prob)

callback = proxnlp.helpers.HistoryCallback()
rho_init = 1e-8
mu_init = 1e-2

solver = proxnlp.Solver(
    pb_space,
    prob,
    mu_init=mu_init,
    rho_init=rho_init,
    tol=TOLERANCE,
    dual_alpha=0.5,
    dual_beta=0.5,
    verbose=proxnlp.VERBOSE,
)
solver.register_callback(callback)
solver.max_iters = 1000
solver.use_gauss_newton = False

xu_init = pb_space.neutral()
lams0 = [np.zeros(cs.nr) for cs in constraints]

try:
    flag = solver.solve(workspace, results, xu_init, lams0)
except KeyboardInterrupt:
    pass

dxus_opt = results.xopt
dxs_opt_flat = dxus_opt[: xspace.nx]
dqs_opt = dxs_opt_flat[: model.nv]
dvs_opt = dxs_opt_flat[model.nv :]

f_opt_proxnlp = dxus_opt[xspace.ndx :]
f_opt_proxnlp = np.split(f_opt_proxnlp, 4)

pr_proxnlp = callback.storage.prim_infeas
du_proxnlp = callback.storage.dual_infeas

qopt_proxnlp = integrate(q0, dqs_opt).full()

viz.display(qopt_proxnlp)


print("Difference between the solutions")
for i, (qipopt, qproxnlp) in enumerate(zip(qopt_ipopt, qopt_proxnlp)):
    print("q_" + str(i) + " difference:\t", qipopt - qproxnlp)

print("\nDifference between the solutions")
for i, (fipopt, fproxnlp) in enumerate(zip(f_opt_ipopt, f_opt_proxnlp)):
    print("F_" + str(i) + " difference:\t", fipopt - fproxnlp)

plt.figure(figsize=(12, 6), dpi=90)
plt.subplot(1, 2, 1)
plt.title("IPOPT Residuals")
plt.semilogy(du_ipopt, marker=".")
plt.semilogy(pr_ipopt, marker=".")
ylims = plt.ylim()
plt.legend(["dual", "primal"])

plt.subplot(1, 2, 2)
plt.title("proxnlp Residuals")
plt.semilogy(du_proxnlp, marker=".")
plt.semilogy(pr_proxnlp, marker=".")
ylims = (min(ylims[0], plt.ylim()[0]), max(ylims[1], plt.ylim()[1]))
plt.ylim(*ylims)
plt.legend(["dual", "primal"])

plt.show()
