"""Camera.project in frame code: the camera orbits the origin and projects five world points
each frame, the fifth mirrored behind the camera, so its w is negative. The (5, 3) float64
array is built in frame code: a 2-d array does not cross the snapshot."""

import math

import numpy as np

import manifoldx as mx

POINTS = [[0.0, 0.0, 0.0], [1.5, 0.5, -2.0], [-3.0, 1.0, 0.25], [0.1, 2.7, 4.4]]


class Orbit:
    def __init__(self):
        self.n = 0


engine = mx.Engine("Projecting", width=320, height=200)
engine.camera.set_pose(position=(8.0, 3.0, 0.0), target=(0.0, 0.0, 0.0))
orbit = Orbit()


def look(query, dt: float):
    orbit.n += 1
    a = 0.037 * orbit.n
    engine.camera.position = np.array([8.0 * math.cos(a), 3.0, 8.0 * math.sin(a)], np.float32)
    p = engine.camera.position
    behind = [2.0 * float(p[0]), 2.0 * float(p[1]), 2.0 * float(p[2])]  # the target is the origin
    pts = np.array(POINTS + [behind])
    xy, w = engine.camera.project(pts, engine.w, engine.h)
    engine.state["xy"] = xy.tolist()
    engine.state["w"] = w.tolist()


engine.system(look)
engine.run()
