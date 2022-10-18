import proxnlp
from proxnlp import manifolds, costs
from proxnlp import C2Function

import numpy as np
import pinocchio as pin
import example_robot_data as erd


class FramePlacement(C2Function):
    def __init__(self, model: pin.Model, fid: int, M_ref: pin.SE3):
        nx = model.nq
        ndx = model.nv
        nr = 6
        super().__init__(nx, ndx, nr)
        self.model = model
        self.data: pin.Data = model.createData()
        self.fid = fid
        self.M_ref = M_ref

    def __call__(self, x):
        pin.forwardKinematics(self.model, self.data, x)
        M = self.data.oMi[self.fid]
        return pin.log6(self.M_ref.actInv(M)).vector

    def computeJacobian(self, x, Jout):
        err = self(x)
        pin.computeJointJacobians(self.model, self.data, x)
        Jout[:] = pin.Jlog6(pin.exp(err)) @ pin.getJointJacobian(
            self.model, self.data, self.fid, pin.LOCAL
        )


robot = erd.load("ur5")
model = robot.model
rdata = robot.data

vizer = pin.visualize.MeshcatVisualizer(
    model, robot.collision_model, robot.visual_model, data=rdata
)

vizer.initViewer(loadModel=True)

pin.seed(10)
np.random.seed(1)
q_rand = pin.randomConfiguration(model)
vizer.display(q_rand)

space = manifolds.MultibodyConfiguration(model)

weights = np.eye(space.ndx)

q_ref = pin.neutral(model)
q_ref = pin.integrate(model, q_rand, 4.0 * np.random.rand(model.nv))
q_ref = pin.integrate(model, q_rand, 4.0 * np.ones(model.nv))
cost = costs.QuadraticDistanceCost(space, q_ref, weights)

model.lowerPositionLimit[:] = -1e2
model.upperPositionLimit[:] = +1e2

ee_name = model.names[-1]
ee_id = model.getJointId(ee_name)
pin.forwardKinematics(model, rdata, q_rand)
M_ref = rdata.oMi[ee_id].copy()

frame_pl_fun = FramePlacement(model, ee_id, M_ref)

frame_pl_autodiff = proxnlp.autodiff.FiniteDifferenceHelperC2(space, frame_pl_fun, 1e-6)


def test_fd():
    x0 = space.rand()
    Jref = np.zeros((6, space.ndx))
    Jfd = Jref.copy()

    frame_pl_fun.computeJacobian(x0, Jref)
    frame_pl_autodiff.computeJacobian(x0, Jfd)
    assert np.allclose(Jref, Jfd)


for _ in range(10):
    test_fd()


constraint = proxnlp.constraints.createEqualityConstraint(frame_pl_autodiff)

problem = proxnlp.Problem(space, cost, [constraint])

tol = 1e-3
mu_init = 0.01
rho_init = 1e-6
solver = proxnlp.Solver(problem, tol, mu_init, rho_init, verbose=proxnlp.VERBOSE)


rs = proxnlp.Results(problem)
ws = proxnlp.Workspace(problem)
solver.register_callback(proxnlp.helpers.HistoryCallback())
solver.solve(ws, rs, q_ref)

print(rs)
