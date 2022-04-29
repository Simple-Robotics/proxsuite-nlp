"""
Copyright (C) 2022 LAAS-CNRS, INRIA
"""
# from crocoddyl source code
from math import cos, sin

import numpy as np
from matplotlib import animation
from matplotlib import pyplot as plt

import pinocchio as pin
import hppfcl as fcl


def make_npendulum(N, ub=True, lengths=None):
    model = pin.Model()
    geom_model = pin.GeometryModel()

    parent_id = 0

    base_radius = 0.08
    shape_base = fcl.Sphere(base_radius)
    geom_base = pin.GeometryObject("base", 0, shape_base, pin.SE3.Identity())
    geom_base.meshColor = np.array([1., 0.1, 0.1, 1.])
    geom_model.addGeometryObject(geom_base)

    joint_placement = pin.SE3.Identity()
    body_mass = 1.
    body_radius = 0.06
    if lengths is None:
        lengths = [1. for _ in range(N)]

    for k in range(N):
        joint_name = "joint_" + str(k + 1)
        if ub:
            jmodel = pin.JointModelRUBX()
        else:
            jmodel = pin.JointModelRX()
        joint_id = model.addJoint(parent_id, jmodel, joint_placement, joint_name)

        body_inertia = pin.Inertia.FromSphere(body_mass, body_radius)
        body_placement = joint_placement.copy()
        body_placement.translation[2] = lengths[k]
        model.appendBodyToJoint(joint_id, body_inertia, body_placement)

        geom1_name = "ball_" + str(k + 1)
        shape1 = fcl.Sphere(body_radius)
        geom1_obj = pin.GeometryObject(geom1_name, joint_id, shape1, body_placement)
        geom1_obj.meshColor = np.ones((4))
        geom_model.addGeometryObject(geom1_obj)

        geom2_name = "bar_" + str(k + 1)
        shape2 = fcl.Cylinder(body_radius / 4, body_placement.translation[2])
        shape2_placement = body_placement.copy()
        shape2_placement.translation[2] /= 2.

        geom2_obj = pin.GeometryObject(geom2_name, joint_id, shape2, shape2_placement)
        geom2_obj.meshColor = np.array([0., 0., 0., 1.])
        geom_model.addGeometryObject(geom2_obj)

        parent_id = joint_id
        joint_placement = body_placement.copy()

    return model, geom_model, geom_model


def make_cartpole(ub=True):
    """Define a Cartpole Pinocchio model."""
    model = pin.Model()
    model.name = "Cartpole"

    m1 = 1.
    m2 = .1
    length = .5
    base_sizes = (.4, .2, .05)

    base = pin.JointModelPX()
    max_x = np.array([30])
    min_x = -max_x
    max_float = np.finfo(float).max
    maxEff = np.array([max_float])
    maxVel = np.array([max_float])
    base_id = model.addJoint(0, base, pin.SE3.Identity(), "base", maxEff, maxVel, min_x, max_x)

    if ub:
        pole = pin.JointModelRUBY()
    else:
        pole = pin.JointModelRY()
    pole_id = model.addJoint(1, pole, pin.SE3.Identity(), "pole")
    model.lowerPositionLimit[:] = -4.
    model.upperPositionLimit[:] = 4.

    base_inertia = pin.Inertia.FromBox(m1, *base_sizes)
    pole_inertia = pin.Inertia.FromEllipsoid(m2, *[1e-2, length, 1e-2])

    base_body_pl = pin.SE3.Identity()
    pole_body_pl = pin.SE3.Identity()
    pole_body_pl.translation = np.array([0., 0., length / 2])

    model.appendBodyToJoint(base_id, base_inertia, base_body_pl)
    model.appendBodyToJoint(pole_id, pole_inertia, pole_body_pl)

    # make visual/collision models
    collision_model = pin.GeometryModel()
    shape_base = fcl.Box(*base_sizes)
    radius = 0.01
    shape_pole = fcl.Capsule(radius, length)
    RED_COLOR = np.array([1, 0., 0., 1.])
    WHITE_COLOR = np.array([1, 1., 1., 1.])
    geom_base = pin.GeometryObject("link_base", base_id, shape_base,
                                   base_body_pl)
    geom_base.meshColor = WHITE_COLOR
    geom_pole = pin.GeometryObject("link_pole", pole_id, shape_pole,
                                   pole_body_pl)
    geom_pole.meshColor = RED_COLOR

    collision_model.addGeometryObject(geom_base)
    collision_model.addGeometryObject(geom_pole)
    visual_model = collision_model
    return model, collision_model, visual_model


def animateCartpole(xs, sleep=50):
    print("processing the animation ... ")
    cart_size = 1.
    pole_length = 5.
    fig = plt.figure()
    ax = plt.axes(xlim=(-8, 8), ylim=(-6, 6))
    patch = plt.Rectangle((0., 0.), cart_size, cart_size, fc='b')
    line, = ax.plot([], [], 'k-', lw=2)
    time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes)

    def init():
        ax.add_patch(patch)
        line.set_data([], [])
        time_text.set_text('')
        return patch, line, time_text

    def animate(i):
        x_cart = xs[i][0]
        y_cart = 0.
        theta = xs[i][1]
        patch.set_xy((x_cart - cart_size / 2, y_cart - cart_size / 2))
        x_pole = np.cumsum([x_cart, -pole_length * sin(theta)])
        y_pole = np.cumsum([y_cart, pole_length * cos(theta)])
        line.set_data(x_pole, y_pole)
        time = i * sleep / 1000.
        time_text.set_text('time = %.1f sec' % time)
        return patch, line, time_text

    anim = animation.FuncAnimation(fig, animate, init_func=init, frames=len(xs), interval=sleep, blit=True)
    print("... processing done")
    return anim
