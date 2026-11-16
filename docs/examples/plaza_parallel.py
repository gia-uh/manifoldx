import os

import numpy as np
from PIL import Image

import manifoldx as mx
from manifoldx.components import Component, Material, Mesh, Transform
from manifoldx.resources import DirectionalLight, StandardMaterial, cube


class Northbound(Component):
    pass


class Southbound(Component):
    pass


class Eastbound(Component):
    pass


class Westbound(Component):
    pass


WALKERS = 60


def build():
    """The plaza with four sets of walkers, each set with its own system."""
    rng = np.random.default_rng(11)
    engine = mx.Engine("parallel plaza", width=480, height=270)
    engine.background_color = (0.12, 0.14, 0.2)
    engine.camera.fit(24.0, center=(0.0, 1.0, 0.0), azimuth=45, elevation=34)
    engine.set_sun(DirectionalLight(color="#fff1d6", intensity=2.6, direction=(-0.4, -1.0, -0.5)))

    engine.spawn(
        Mesh(cube(40, 0.2, 40)),
        Material(StandardMaterial("#6b6a63", roughness=0.95)),
        Transform(pos=(0.0, -0.1, 0.0)),
    )
    engine.spawn(
        Mesh(cube(3, 1.2, 3)),
        Material(StandardMaterial("#8fa3b8", roughness=0.4)),
        Transform(pos=(0.0, 0.6, 0.0)),
    )

    for marker, color in [
        (Northbound, "#c8a06a"),
        (Southbound, "#a6735a"),
        (Eastbound, "#7f9d6a"),
        (Westbound, "#6e7f9d"),
    ]:
        positions = np.column_stack([
            rng.uniform(-16.0, 16.0, WALKERS),
            np.full(WALKERS, 0.55),
            rng.uniform(-16.0, 16.0, WALKERS),
        ]).astype(np.float32)
        engine.spawn(
            marker(),
            Mesh(cube(0.35, 1.1, 0.35)),
            Material(StandardMaterial(color)),
            Transform(pos=positions),
            n=WALKERS,
        )

    @engine.system(parallel=True)
    def north(q: mx.Query[Transform, Northbound], dt: float):
        q[Transform].position += np.array([0.0, 0.0, -3.0 * dt], np.float32)

    @engine.system(parallel=True)
    def south(q: mx.Query[Transform, Southbound], dt: float):
        q[Transform].position += np.array([0.0, 0.0, 3.0 * dt], np.float32)

    @engine.system(parallel=True)
    def east(q: mx.Query[Transform, Eastbound], dt: float):
        q[Transform].position += np.array([3.0 * dt, 0.0, 0.0], np.float32)

    @engine.system(parallel=True)
    def west(q: mx.Query[Transform, Westbound], dt: float):
        q[Transform].position += np.array([-3.0 * dt, 0.0, 0.0], np.float32)

    @engine.system
    def count_the_edge(q: mx.Query[Transform], dt: float):
        position = q[Transform].position.data
        engine.edge = int((np.abs(position[:, [0, 2]]) > 15.0).any(axis=1).sum())

    return engine


engine = build()
for _ in range(6):
    frame = engine.render_frame(supersample=1)

print("at the edge:", engine.edge)
print("per system, ms:", {name.rsplit(".", 1)[-1]: round(1000 * seconds, 2)
                          for name, seconds in engine.system_split.items()})
print("group wall, ms:", [round(1000 * seconds, 2) for seconds in engine.group_split])
print("systems total, ms:", round(1000 * sum(engine.system_split.values()), 2))
parallel, serial = build(), build()
for _ in range(12):
    parallel.render_frame(supersample=1)

os.environ["MANIFOLDX_PARALLEL"] = "serial"
for _ in range(12):
    serial.render_frame(supersample=1)
del os.environ["MANIFOLDX_PARALLEL"]

same = parallel.store._components["Transform"].tobytes() == serial.store._components["Transform"].tobytes()
print("same store after 12 frames:", same)
parallel.close()
serial.close()
Image.fromarray(frame)
