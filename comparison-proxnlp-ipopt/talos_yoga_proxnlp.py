"""
minimization with constraints

Simple example with regularization cost, desired position and position constraints.

min Dq    sum  || q-q0 ||**2  + || diff(position_hand_r(q), position_hand_l(q)) ||**2 +
               || orientation_hand_r(q) - rot_ref_hand_r ||**2 + || orientation_hand_l(q) - rot_ref_hand_l ||**2 +
               || rot_body_y(q) ||**2 + || dist(elbows_(q)) - elboow_ref ||**2
s.t
        q  = integrate(q, Dq)

        lb_com < position_com(q) < ub_com
        pos_and_rot_foot_r(q) == ref_rfeet
        lb_foot_l < foot_l_pos(q) < ub_foot_left
        lb_hand_l < hand_l_pos(q) < ub_hand_l

So the robot should reach a yoga position

"""
import pinocchio as pin
from pinocchio import casadi as cpin
import casadi
import numpy as np
import example_robot_data as robex

# import matplotlib.pyplot as plt; plt.ion()
from meshcat_utils import ForceDraw  # ManifoldFR's pin-meshcat-utils
import time

import proxnlp
from proxnlp.manifolds import MultibodyPhaseSpace, VectorSpace
from proxnlp.utils import CasadiFunction, plot_pd_errs

# Load the model both in pinocchio and pinocchio casadi
robot = robex.load("talos")
cmodel = cpin.Model(robot.model)
cdata = cmodel.createData()

model = robot.model
data = model.createData()

q0 = robot.q0
cq0 = casadi.SX(q0)
nq = cmodel.nq
nv = robot.nv
nDq = cmodel.nv
nu = robot.nv - 6

nsteps = 0

xspace = MultibodyPhaseSpace(model)
pb_space = xspace.tangent_space()
# pb_space = VectorSpace(nsteps * nu + (nsteps + 1) * (xspace.nx))

# viz = pin.visualize.GepettoVisualizer(robot.model,robot.collision_model,robot.visual_model)
viz = pin.visualize.MeshcatVisualizer(
    robot.model, robot.collision_model, robot.visual_model
)
viz.initViewer()
viz.loadViewerModel()
viz.display(robot.q0)
viz.viewer.open()


# reference configuration

cq = casadi.SX.sym("cq", nq, 1)
cDq = casadi.SX.sym("cx", nDq, 1)
R = casadi.SX.sym("R", 3, 3)
R_ref = casadi.SX.sym("R_ref", 3, 3)

# Get the index of the frames which are going to be used
IDX_BASE = cmodel.getFrameId("torso_2_link")
IDX_LF = cmodel.getFrameId("leg_left_6_link")
IDX_RF = cmodel.getFrameId("leg_right_6_link")
IDX_LG = cmodel.getFrameId("gripper_left_base_link")
IDX_RG = cmodel.getFrameId("gripper_right_base_link")
IDX_LE = cmodel.getFrameId("arm_left_4_joint")
IDX_RE = cmodel.getFrameId("arm_right_4_joint")

# This is used in order to go from a configuration and the displacement to the final configuration.
# Why pinocchio.integrate and not simply q = q0 + v*dt?
# q and v have different dimensions because q contains quaterniions and this can't be done
# So pinocchio.integrate(config, Dq)
integrate = casadi.Function("integrate", [cq, cDq], [cpin.integrate(cmodel, cq, cDq)])

# Casadi function to map joints configuration to COM position
com_position = casadi.Function("com", [cq], [cpin.centerOfMass(cmodel, cdata, cq)])

# Compute the forward kinematics and store the data in 'cdata'
# Note that now cdata is filled with symbols, so there is no need to compute the forward kinematics at every variation of q
# Since everything is a symbol, a substituition (which is what casadi functions do) is enough
cpin.framesForwardKinematics(cmodel, cdata, cq)

base_rotation = casadi.Function("com", [cq], [cdata.oMf[IDX_BASE].rotation])

# Casadi functions can't output a SE3 element, so the oMf matrices are split in rotational and translational components

lf_position = casadi.Function("lf_pos", [cq], [cdata.oMf[IDX_LF].translation])
lf_rotation = casadi.Function("lf_rot", [cq], [cdata.oMf[IDX_LF].rotation])
rf_position = casadi.Function("rf_pos", [cq], [cdata.oMf[IDX_RF].translation])
rf_rotation = casadi.Function("rf_rot", [cq], [cdata.oMf[IDX_RF].rotation])

lg_position = casadi.Function("lg_pos", [cq], [cdata.oMf[IDX_LG].translation])
lg_rotation = casadi.Function("lg_rot", [cq], [cdata.oMf[IDX_LG].rotation])
le_rotation = casadi.Function("le_rot", [cq], [cdata.oMf[IDX_LE].rotation])
le_translation = casadi.Function("le_pos", [cq], [cdata.oMf[IDX_LE].translation])

rg_position = casadi.Function("rg_pos", [cq], [cdata.oMf[IDX_RG].translation])
rg_rotation = casadi.Function("rg_rot", [cq], [cdata.oMf[IDX_RG].rotation])
re_rotation = casadi.Function("re_rot", [cq], [cdata.oMf[IDX_RE].rotation])
re_translation = casadi.Function("re_pos", [cq], [cdata.oMf[IDX_RE].translation])

log = casadi.Function("log", [R, R_ref], [cpin.log3(R.T @ R_ref)])


### ----------------------------------------------------------------------------- ###
### OPTIMIZATION PROBLEM

# Defining weights
parallel_cost = 1e3
distance_cost = 1e3
straightness_body_cost = 1e3
elbow_distance_cost = 1e1
distance_btw_hands = 0.3

assert xspace.nx == model.nq + model.nv
assert xspace.ndx == model.nv * 2
assert pb_space.nx == model.nv * 2

# Note that here the optimization variables are Dq, not q, and q is obtained by integrating.
# q = q + Dq, where the plus sign is intended as an integrator (because nq is different from nv)
# It is also possible to optimize directly in q, but in that case a constraint must be added in order to have
# the norm squared of the quaternions = 1
Dxs = casadi.SX.sym("x_opt_0", xspace.ndx)
Dqs = Dxs[:nv]
cq_int_ = cpin.integrate(cmodel, cq0, Dqs)


# Cost
cost = 0
cost += casadi.sumsqr(cpin.difference(cmodel, cq_int_, cq0)) * 0.5
cost += casadi.sumsqr(com_position(cq_int_)[0] - 0.5) * 1.
cost_fun = CasadiFunction(pb_space.nx, pb_space.ndx, cost, Dxs)

""" # Distance between the hands
cost += distance_cost * casadi.sumsqr(lg_position(qs) - rg_position(qs) 
                                     - np.array([0, distance_btw_hands, 0]))   """

""" # Cost on parallelism of the two hands
r_ref = pin.utils.rotate('x', 3.14 / 2) # orientation target
cost += parallel_cost * casadi.sumsqr(log(rg_rotation(qs), r_ref))
r_ref = pin.utils.rotate('x', -3.14 / 2) # orientation target
cost += parallel_cost * casadi.sumsqr(log(lg_rotation(qs), r_ref))

# Body in a straight position
cost += straightness_body_cost * casadi.sumsqr(log(base_rotation(qs), base_rotation(q0)))

cost +=  elbow_distance_cost *casadi.sumsqr(le_translation(qs)[1] - 2) \
        + elbow_distance_cost *casadi.sumsqr(re_translation(qs)[1] + 2)
"""

# Standing foot
eq_fun_exprs = []
eq_fun_exprs.append(rf_position(cq_int_) - rf_position(q0))
eq_fun_exprs.append(log(rf_rotation(cq_int_), rf_rotation(q0)))

eq_expr = casadi.vertcat(*eq_fun_exprs)
eq_constr_fun = CasadiFunction(
    pb_space.nx, pb_space.ndx, eq_expr, Dxs, use_hessian=False
)


# Free foot
"""opti.subject_to(lf_position(qs)[2] >= 0.4)
opti.subject_to(opti.bounded(0.05, lf_position(qs)[0:2], 0.1))

r_ref = pin.utils.rotate('z', 3.14 / 2) @ pin.utils.rotate('y', 3.14 / 2) # orientation target
opti.subject_to(opti.bounded(-0.0, lf_rotation(qs) - r_ref, 0.0))

# Left hand constraint to be at a certain height
opti.subject_to(opti.bounded(1.1, rg_position(qs)[2], 1.2))
opti.subject_to(opti.bounded(-distance_btw_hands/2, rg_position(qs)[1], 0)) """


### ----------------------------------------------------------------------------- ###
# Solver Setup

cost_fun_ = proxnlp.costs.CostFromFunction(cost_fun)
eq_constr_ = proxnlp.constraints.create_equality_constraint(eq_constr_fun)

constraints = []
constraints.append(eq_constr_)

prob = proxnlp.Problem(cost_fun_, constraints)

print("No. of variables  :", pb_space.nx)
print("No. of constraints:", prob.total_constraint_dim)
workspace = proxnlp.Workspace(pb_space.nx, pb_space.ndx, prob)
results = proxnlp.Results(pb_space.nx, prob)

callback = proxnlp.helpers.HistoryCallback()
tol = 1e-4
rho_init = 1e-7
mu_init = 0.1

solver = proxnlp.Solver(
    pb_space, prob, mu_init=mu_init, rho_init=rho_init, tol=tol, verbose=proxnlp.VERBOSE
)
solver.register_callback(callback)
solver.maxiters = 3000
solver.use_gauss_newton = True

xu_init = pb_space.neutral()
lams0 = [np.zeros(cs.nr) for cs in constraints]

flag = solver.solve(workspace, results, xu_init, lams0)

### -------------------------------------------------------------- ###
# Get results

print("Results struct:\n{}".format(results))
prim_errs = callback.storage.prim_infeas
dual_errs = callback.storage.dual_infeas

dxus_opt = results.xopt
dxs_opt_flat = dxus_opt[: (nsteps + 1) * xspace.nx]
dxs_opt = dxs_opt_flat.reshape(nsteps + 1, -1)
dqs_opt = dxs_opt[:, : model.nv]
dvs_opt = dxs_opt[:, model.nv :]

qs_opt = integrate(q0, dqs_opt).full()


input("(ref) Press enter to see")

viz.display(q0)

input("(opt) Press enter")
### VISUALIZATION

viz.display(qs_opt)

