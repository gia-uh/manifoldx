"""People on a street: 50 sampled figures walk, run, stand and cross, posed at three levels by distance.

Each figure is a `HumanParams.sample` dressed in a seeded outfit and built at the level its row's
distance from the camera picks: level 0 nearer than 8 m, level 1 nearer than 20 m, level 2 beyond.
Rows run across the view; a figure walks its row along x and wraps at the ends. Each figure's
cycle is baked once into a 32-frame clip the GPU keeps (`bake_cycle`, `Engine.bake_poses`); every
frame a system advances the phases and points each figure at its frame (`set_pose_frames`), and
the GPU blends the two frames around it, so no matrices are computed or uploaded while they walk.

    uv run python examples/humans.py
    uv run python examples/humans.py --render --duration 4 --output humans.mp4
"""

import math

import numpy as np

import manifoldx as mx
from manifoldx.components import Material, Mesh, Transform
from manifoldx.humanoid.cycles import bake_cycle, cadence
from manifoldx.modeling import Mesh as Shape
from manifoldx.modeling.assets.human import HumanParams, build_human
from manifoldx.modeling.assets.human_dress import Outfit
from manifoldx.resources import DirectionalLight, PointLight, StandardMaterial

PEOPLE = 50
ROWS = (4.0, 6.0, 9.0, 12.0, 15.0, 19.0, 24.0, 30.0, 38.0)  # each row's distance from the camera, m
HALF = 14.0  # a row runs from x = -HALF to +HALF and wraps
BANDS = (8.0, 20.0)  # level 0 nearer than the first, level 1 nearer than the second, else 2
KINDS = (("walk", 0.6, (1.1, 1.6)), ("run", 0.15, (2.8, 3.8)), ("cross", 0.15, (1.0, 1.3)), ("idle", 0.1, (0.0, 0.0)))
COLOURS = ("#e8e2d0", "#3b4a5c", "#8c2f39", "#2e5e4e", "#c9a227", "#4a4e69", "#d9d4c7", "#1d3557", "#a26769")
HAIR = ("#2a1d14", "#5a3a22", "#151515", "#a67b5b", "#cfcfcf")


def level_at(distance: float) -> int:
    return 0 if distance < BANDS[0] else 1 if distance < BANDS[1] else 2


def outfit(g: np.random.Generator) -> Outfit:
    def pick(xs):
        return xs[int(g.integers(len(xs)))]

    return Outfit(
        top=pick(["tshirt", "shirt", "blouse", "tank"]),
        sleeves=round(float(g.uniform(0, 1)), 2),
        bottom=pick(["trousers", "trousers", "shorts", "skirt"]),
        bottom_length=round(float(g.uniform(0.5, 1)), 2),
        looseness=round(float(g.uniform(0.006, 0.03)), 3),
        shoes=pick(["shoes", "sandals", "boots"]),
        hair=pick([None, "short", "long", "bun", "curly"]),
        hat=pick([None, None, None, "cap", "brim"]),
        top_color=pick(COLOURS),
        bottom_color=pick(COLOURS),
        hair_color=pick(HAIR),
    )


class Person:
    """One figure on the street: its build, its gait, where it is and its entity."""

    def __init__(self, i: int, g: np.random.Generator):
        self.params = HumanParams.sample(1000 + i)
        r, kinds = g.random(), 0.0
        for kind, share, (lo, hi) in KINDS:
            kinds += share
            if r < kinds:
                break
        self.kind, self.speed = kind, float(g.uniform(lo, hi))
        self.z = ROWS[i % len(ROWS)] + float(g.uniform(-0.6, 0.6))
        self.x = float(g.uniform(-HALF, HALF))
        self.heading = 1.0 if g.random() < 0.5 else -1.0
        self.phase = float(g.random())
        self.rate = cadence(self.kind, self.speed, self.params)
        self.level = level_at(self.z)
        self.figure = build_human(self.params, outfit=outfit(g), level=self.level)
        self.handle = None


engine = mx.Engine("People on a street", width=960, height=540)
engine.set_lights([PointLight(color="#fff2e0", intensity=40.0, position=(0.0, 8.0, 2.0))])
engine.set_sun(DirectionalLight(color="#fff6e8", intensity=2.5, direction=(-0.4, -0.8, 0.45)))
engine.camera.set_pose(position=(0.0, 2.2, -1.5), target=(0.0, 1.0, 14.0))
RNG = np.random.default_rng(19)
people = [Person(i, RNG) for i in range(PEOPLE)]
geometries = {id(p): p.figure.mesh.to_geometry() for p in people}
FRAMES = 32  # poses baked per cycle


@engine.on("startup")
def place(_payload):
    ground = Shape.plane(2 * HALF + 10, 60.0).translate((0.0, 0.0, 25.0)).to_geometry()
    engine.spawn(Mesh(ground), Material(mx.material.standard(color="#9a9488", roughness=0.95)), Transform(pos=(0, 0, 0)))
    skin = StandardMaterial("#ffffff", roughness=0.7, vertex_colors=True)  # the figures' painted skin and clothes
    for p in people:
        p.handle = engine.spawn(Mesh(geometries[id(p)]), Material(skin),
                                Transform(pos=(p.x, 0.0, p.z), rot=Transform.rotation(y=p.heading * math.pi / 2)))
        p.clip = engine.bake_poses(bake_cycle(p.figure, p.kind, p.speed, FRAMES))


@engine.system
def step(q: mx.Query[Transform], dt: float):
    for p in people:
        p.phase = (p.phase + p.rate * dt) % 1.0
        if p.kind != "idle":
            p.x += p.heading * p.speed * dt
            if abs(p.x) > HALF:
                p.x -= p.heading * 2 * HALF
            p.handle.transform.pos = (p.x, 0.0, p.z)
    engine.set_pose_frames([p.handle.index for p in people], [p.clip for p in people],
                           [p.phase * FRAMES for p in people])


if __name__ == "__main__":
    engine.cli(duration=4)
