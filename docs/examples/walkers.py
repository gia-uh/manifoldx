import numpy as np
from PIL import Image

import manifoldx as mx
from manifoldx.components import Colors, Component, Layer, Material, Mesh, Transform
from manifoldx.physics import all_pairs
from manifoldx.resources import DirectionalLight, StandardMaterial, cube, plane, sphere
from manifoldx.types import Color, Float, Vector3

N = 200
FLOOR = (-0.70710678, 0.0, 0.0, 0.70710678)  # a quarter turn about x: lay the plane flat
COATS = [Colors.CYAN, Colors.YELLOW, Colors.MAGENTA, Colors.WHITE]


class Walker(Component):
    """What one person knows: where they are going and how fast."""

    velocity: Vector3
    destination: Vector3
    pace: Float = 1.2


engine = mx.Engine("plaza", width=480, height=270)
engine.spawn(
    Mesh(plane(40, 40)),
    Material(StandardMaterial(color="#c8b89a", roughness=0.9)),
    Transform(rot=FLOOR),
)
engine.spawn(
    Mesh(cube(4, 1, 4)),
    Material(StandardMaterial(color="#8a9bb0", roughness=0.6)),
    Transform(pos=(0, 0.5, 0)),
)
engine.camera.set_pose(position=(0, 10, 15), target=(0, 0.8, -6))
engine.set_sun(DirectionalLight(color="#ffffff", intensity=3.0, direction=(-0.4, -1, -0.3)))


@engine.background
def arcade(seed: int, n: int) -> np.ndarray:
    """n destinations along the east arcade, as far apart as we can make them."""
    rng = np.random.default_rng(seed)
    candidates = np.column_stack(
        [
            np.full(n * 4, 18.0),
            np.full(n * 4, 0.85),
            rng.uniform(-16.0, -4.0, n * 4),
        ]
    ).astype(np.float32)
    pairs = all_pairs(candidates)
    chosen = [0]
    nearest = pairs.dist[0].copy()
    for _ in range(n - 1):
        k = int(np.argmax(nearest))
        chosen.append(k)
        nearest = np.minimum(nearest, pairs.dist[k])
    return candidates[chosen]


destinations = arcade(7, N).wait()

rng = np.random.default_rng(11)
start = np.column_stack(
    [
        rng.uniform(-19.0, 8.0, N),
        np.full(N, 0.85),
        rng.uniform(-16.0, -4.0, N),
    ]
).astype(np.float32)
pace = rng.uniform(0.9, 1.6, N).astype(np.float32)

body = Mesh(cube(0.4, 1.7, 0.25))
walkers = []
for i, coat in enumerate(COATS):
    group = slice(i * 50, (i + 1) * 50)
    walkers.append(
        engine.spawn(
            body,
            Material(StandardMaterial(color=coat, roughness=0.8)),
            Transform(pos=start[group]),
            Walker(
                velocity=np.zeros((50, 3), np.float32),
                destination=destinations[group],
                pace=pace[group],
            ),
            Layer(layer=1),
            n=50,
        )
    )

view = engine.store.get_component_view(["Transform", "Walker"])
print(len(view), "walkers,", len(engine.store.get_component_view([])), "entities alive")
Image.fromarray(engine.render_frame(supersample=1))


@engine.system
def walk(query: mx.Query[Transform, Walker], dt: float):
    pos = query[Transform].pos.data
    vel = query[Walker].velocity.data
    dest = query[Walker].destination.data
    pace = query[Walker].pace.data[:, 0]

    to = dest - pos
    dist = np.linalg.norm(to, axis=1, keepdims=True)
    wanted = to / np.maximum(dist, 1e-6) * pace[:, None]
    wanted[dist[:, 0] < 0.6] = 0.0  # close enough: stop where you are

    pairs = all_pairs(pos, softening=0.01)
    overlap = np.maximum(0.9 - pairs.dist, 0.0)
    np.fill_diagonal(overlap, 0.0)
    push = (-pairs.diff / pairs.dist_safe[..., None] * overlap[..., None]).sum(axis=1)
    wanted += push * 1.5

    new_vel = vel + (wanted - vel) * np.minimum(1.0, 4.0 * dt)
    query[Walker].velocity += new_vel - vel
    query[Transform].pos += new_vel * dt


@engine.system
def face(query: mx.Query[Transform, Walker], dt: float):
    to = query[Walker].destination.data - query[Transform].pos.data
    query[Transform].rot = Transform.rotation(y=np.arctan2(to[:, 0], to[:, 2]))


engine.set_fixed_timestep(0.05)
for _ in range(40):
    frame = engine.render_frame(supersample=1)
print("mean x after 2 s:", round(float(view[Transform].pos.data[:, 0].mean()), 2))
Image.fromarray(frame)
leader = walkers[0]
leader.set_material(StandardMaterial(color="#0b7285", roughness=0.4))
leader.transform.pos = np.array([-5.0, 0.85, 1.8], np.float32)
print("the guide is entity", leader.index)

for _ in range(20):
    frame = engine.render_frame(supersample=1)
Image.fromarray(frame)
marker = Mesh(sphere(0.3, 8))
for i, coat in enumerate(COATS):
    c = Color(coat)
    group = slice(i * 50, (i + 1) * 50)
    engine.spawn(
        marker,
        Material(StandardMaterial(color=(0.8 * c.r, 0.8 * c.g, 0.8 * c.b), roughness=0.9)),
        Transform(pos=destinations[group]),
        n=50,
    )
engine.camera.layers = {0, 1}
print(len(engine.store.get_component_view(["Layer"])), "people on layer 1")
for _ in range(120):
    frame = engine.render_frame(supersample=1)

pos = view[Transform].pos.data
left = np.linalg.norm(view[Walker].destination.data - pos, axis=1)
pairs = all_pairs(pos)
off = ~np.eye(N, dtype=bool)
print(
    "walking:",
    int((left > 0.7).sum()),
    "closest pair:",
    round(float(pairs.dist[off].min()), 2),
)
Image.fromarray(frame)
