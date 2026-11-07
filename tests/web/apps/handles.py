"""Entity handles in frame code: 8 crates held by app objects circle, spin and grow through
their handles' transforms, take a material on a timer, and crate 3 is destroyed at frame 50.
`--far` moves the floating origin far away first, so a handle's pos is world coordinates;
`--dead` sets a material on crate 3 after it is gone, which raises."""

import math
import sys

import numpy as np

import manifoldx as mx
from manifoldx.components import Material, Mesh, Transform
from manifoldx.resources import PhongMaterial, cube

FAR = "--far" in sys.argv
DEAD = "--dead" in sys.argv
CENTER = (5000.0, -3000.0) if FAR else (0.0, 0.0)


class Crate:
    def __init__(self, entity, phase: float):
        self.entity = entity
        self.phase = phase


class Clock:
    def __init__(self):
        self.n = 0


engine = mx.Engine("Handles", width=320, height=200)
if FAR:
    engine.set_origin(*CENTER)
CUBE = cube(1.0, 1.0, 1.0)
GREY = PhongMaterial(color="#888888")
RED = PhongMaterial(color="#ff0000")
BLUE = PhongMaterial(color="#0000ff")
crates = [
    Crate(
        engine.spawn(Mesh(CUBE), Material(GREY), Transform(pos=(CENTER[0] + i, 1.0, CENTER[1]))),
        0.7 * i,
    )
    for i in range(8)
]
clock = Clock()


def move(query, dt: float):
    clock.n += 1
    for c in crates:
        angle = c.phase + 0.05 * clock.n
        x = CENTER[0] + 4.0 * math.cos(angle)
        z = CENTER[1] + 4.0 * math.sin(angle)
        c.entity.transform.pos = (x, 1.0 + 0.1 * math.sin(angle), z)
        c.entity.transform.rot = Transform.rotation(y=angle)
        s = c.entity.transform.scale
        c.entity.transform.scale = s * 1.002
    if clock.n % 10 == 0:
        c = crates[(clock.n // 10) % 8]
        c.entity.set_material(RED if clock.n % 20 == 0 else BLUE)
    if clock.n == 50:
        crates[3].entity.destroy()
    if DEAD and clock.n > 51:
        engine.set_material(crates[3].entity, RED)
    engine.camera.up = np.array([0.01 * math.sin(0.1 * clock.n), 1.0, 0.0], np.float32)


def watch(query, dt: float):
    p = crates[0].entity.transform.pos
    engine.state["first"] = [float(p[0]), float(p[1]), float(p[2])]
    engine.state["seen"] = len(query)  # crate 3 is still here in frame 50: destroy queues


engine.system(move)
engine.system(watch)
engine.run()
