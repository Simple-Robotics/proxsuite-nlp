"""
Copyright (C) 2022 LAAS-CNRS, INRIA
"""

# from crocoddyl source code
import numpy as np

import pinocchio as pin
import hppfcl as fcl


def make_cartpole(ub=True):
    """Define a Cartpole Pinocchio model."""
    model = pin.Model()
    model.name = "Cartpole"

    m1 = 1.0
    m2 = 0.1
    length = 0.5
    base_sizes = (0.4, 0.2, 0.05)

    base = pin.JointModelPX()
    max_x = np.array([30])
    min_x = -max_x
    max_float = np.finfo(float).max
    maxEff = np.array([max_float])
    maxVel = np.array([max_float])
    base_id = model.addJoint(
        0, base, pin.SE3.Identity(), "base", maxEff, maxVel, min_x, max_x
    )

    if ub:
        pole = pin.JointModelRUBY()
    else:
        pole = pin.JointModelRY()
    pole_id = model.addJoint(1, pole, pin.SE3.Identity(), "pole")
    model.lowerPositionLimit[:] = -4.0
    model.upperPositionLimit[:] = 4.0

    base_inertia = pin.Inertia.FromBox(m1, *base_sizes)
    pole_inertia = pin.Inertia.FromEllipsoid(m2, *[1e-2, length, 1e-2])

    base_body_pl = pin.SE3.Identity()
    pole_body_pl = pin.SE3.Identity()
    pole_body_pl.translation = np.array([0.0, 0.0, length / 2])

    model.appendBodyToJoint(base_id, base_inertia, base_body_pl)
    model.appendBodyToJoint(pole_id, pole_inertia, pole_body_pl)

    # make visual/collision models
    collision_model = pin.GeometryModel()
    shape_base = fcl.Box(*base_sizes)
    radius = 0.01
    shape_pole = fcl.Capsule(radius, length)
    RED_COLOR = np.array([1, 0.0, 0.0, 1.0])
    WHITE_COLOR = np.array([1, 1.0, 1.0, 1.0])
    geom_base = pin.GeometryObject("link_base", base_id, shape_base, base_body_pl)
    geom_base.meshColor = WHITE_COLOR
    geom_pole = pin.GeometryObject("link_pole", pole_id, shape_pole, pole_body_pl)
    geom_pole.meshColor = RED_COLOR

    collision_model.addGeometryObject(geom_base)
    collision_model.addGeometryObject(geom_pole)
    visual_model = collision_model
    return model, collision_model, visual_model
