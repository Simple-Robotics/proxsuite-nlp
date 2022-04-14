from lienlp import manifolds

vs = manifolds.EuclideanSpace(2)

print(vs.neutral())
print(vs.rand())


tso2_space = manifolds.TSO2()


import pinocchio as pin

model = pin.buildSampleModelManipulator()
config_space = manifolds.MultibodyConfiguration(model)
print("nq:", model.nq, " / space nx :", config_space.nx)
print("nv:", model.nv, " / space ndx:", config_space.ndx)

print(config_space.neutral())
print(config_space.rand())


statemultibody = manifolds.MultibodyPhaseSpace(model)
print("nx:", statemultibody.nx, " | ndx:", statemultibody.ndx)
x0 = statemultibody.neutral()
x1 = statemultibody.rand()
print("x0:", x0)
print("x1:", x1)
