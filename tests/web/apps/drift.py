"""Component queries in systems: a ground without Velocity and 12 boxes that drift, spin and
fall. drift queues its writes; watch, after it, reads the rows from before the flush; low
counts the entities below y = 0 through a one-column field."""

import numpy as np

import manifoldx as mx
from manifoldx.components import Component, Material, Mesh, Transform
from manifoldx.resources import PhongMaterial, cube
from manifoldx.types import Vector3

GRAVITY = np.array([0.0, -9.8, 0.0], np.float32)
SPIN = np.array([0.0, 0.0998334, 0.0, 0.9950042], np.float32)


class Velocity(Component):
    v: Vector3


engine = mx.Engine("Drift", width=320, height=200)
CUBE = cube(1.0, 1.0, 1.0)
GREY = PhongMaterial(color="#888888")
engine.spawn(Mesh(CUBE), Material(GREY), Transform(pos=(0.0, -1.0, 0.0), scale=(20.0, 0.1, 20.0)))
xs = np.linspace(-5.5, 5.5, 12)
for i, x in enumerate(xs):
    v = (float(np.linspace(-1.0, 1.0, 12)[i]), float(np.linspace(2.0, 6.0, 12)[i]), 0.5)
    engine.spawn(Mesh(CUBE), Material(GREY), Transform(pos=(float(x), 1.0, 0.0)), Velocity(v=v))


def drift(q: mx.Query[Transform, Velocity], dt: float):
    q[Velocity].v += GRAVITY * dt
    q[Transform].pos += q[Velocity].v * dt
    q[Transform].rot += SPIN


def watch(q: mx.Query[Transform, Velocity], dt: float):
    engine.state["sum"] = float(np.sum(q[Transform].pos.data, axis=None))
    engine.state["n"] = len(q)


def low(q: mx.Query[Transform], dt: float):
    engine.state["low"] = len(q[Transform].pos[:, 1] < 0.0)


engine.system(drift)
engine.system(watch)
engine.system(low)
engine.run()
