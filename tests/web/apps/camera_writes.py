"""A system that writes each camera field the page applies (position, target, fov, near,
far), by assignment and in place: the replay and the three.js camera follow
(gia-uh/manifoldx#151)."""

import numpy as np

import manifoldx as mx

engine = mx.Engine("Camera writes", width=320, height=200)
engine.camera.position = np.array([0.0, 2.0, 8.0], np.float32)
engine.camera.target = np.array([0.0, 0.0, 0.0], np.float32)


class Dolly:
    def __init__(self):
        self.n = 0
        self.ahead = 0.0


dolly = Dolly()


def tick(query, dt: float):
    dolly.n += 1
    cam = engine.camera
    cam.position += np.array([0.1, 0.0, -0.2], np.float32)
    engine.camera.target = cam.position + np.array([0.3, -0.5, -3.0], np.float32)
    cam.fov = 50 + dolly.n
    engine.camera.near = 0.1 + 0.01 * dolly.n
    cam.far += 10.0
    dolly.ahead = float(cam.get_forward()[2])


engine.system(tick)
engine.run()
