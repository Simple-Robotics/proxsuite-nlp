"""
Copyright (C) 2022 LAAS-CNRS, INRIA
"""
from proxnlp import manifolds
import numpy as np
import pinocchio as pin

vs = manifolds.EuclideanSpace(2)

print(vs.neutral())
print(vs.rand())

tso2_space = manifolds.TSO2()


model = pin.buildSampleModelHumanoid()
config_space = manifolds.MultibodyConfiguration(model)
assert model.nq == config_space.nx
assert model.nv == config_space.ndx

q0 = config_space.rand()
q1 = config_space.rand()
for q in [q0, q1]:
    q[:7] = np.clip(-10, 10, q[:7])
print(q0)
v0 = np.random.randn(config_space.ndx)
q2 = config_space.integrate(q0, v0)

J0_ref = np.eye(model.nv)
J1_ref = np.eye(model.nv)
config_space.Jintegrate(q0, v0, J0_ref, 0)
config_space.Jintegrate(q0, v0, J1_ref, 1)
assert np.allclose(J0_ref, config_space.Jintegrate(q0, v0, 0))
assert np.allclose(J1_ref, config_space.Jintegrate(q0, v0, 1))


statemultibody = manifolds.MultibodyPhaseSpace(model)
print("nx:", statemultibody.nx, " | ndx:", statemultibody.ndx)
x0 = statemultibody.neutral()
x1 = statemultibody.rand()
x1[:7] = np.clip(-10, 10, x1[:7])
print("x0:", x0)
print("x1:", x1)
