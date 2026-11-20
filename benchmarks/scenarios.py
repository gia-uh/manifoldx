"""manifoldx's own scenarios for `manifoldx bench run`, over its examples and benchmark scenes.

    uv run manifoldx bench run quad_zoom --size 960x540
    uv run manifoldx bench run meshes_spin --loop pipelined
"""

import time

import numpy as np

import manifoldx as mx
from manifoldx import bench
from manifoldx.components import Material, Mesh, Transform
from manifoldx.resources import StandardMaterial, cube

AT = (51_200.0, 51_200.0)  # the middle of the quad streaming plain


@bench.scenario("quad_zoom", sizes={"smoke": 20, "full": None})
def quad_zoom(engine, p):
    """The quad streaming plain from 20 km up down to 50 m, each height held until it has settled."""
    from examples.quad_streaming import build

    streamer = build(engine, fly=False)

    @engine.system
    def follow(query, dt):  # the streamer follows the camera the steps move
        ox, oz = engine.origin
        x, y, z = (float(v) for v in engine.camera.position)
        streamer.focus(x + ox, z + oz, y)

    ready = bench.settled(streamer)
    return streamer, [
        bench.camera((AT[0], 20_000.0, AT[1] - 6000.0), (AT[0], 0.0, AT[1])),
        bench.hold(until=ready, max_s=60),
        bench.mark("zoom"),
        bench.zoom(20_000.0, 50.0, at=AT, steps=6, hold=ready, max_s=60),
        bench.shot("street"),
    ]


@bench.scenario("meshes_spin", params={"n": 22_000, "system_ms": 4.0}, sizes={"smoke": 20, "full": None})
def meshes_spin(engine, p):
    """n cubes and a system that busies the CPU for system_ms: what --loop pipelined overlaps."""
    side = int(np.ceil(np.sqrt(p.n)))
    grid = np.stack(np.meshgrid(np.arange(side), np.arange(side)), -1).reshape(-1, 2)[: p.n] - side / 2
    pos = np.zeros((p.n, 3), np.float32)
    pos[:, 0], pos[:, 2] = grid[:, 0], grid[:, 1]
    engine.spawn(Mesh(cube(0.5, 0.5, 0.5)), Material(StandardMaterial("#88aaff")), Transform(pos=pos), n=p.n)
    engine.camera.fit(side)

    @engine.system
    def spin(q: mx.Query[Transform], dt: float):
        data = q[Transform].position.data
        deadline = time.perf_counter() + p.system_ms / 1000.0
        while time.perf_counter() < deadline:
            data = np.sin(data) * 0.5 + 0.5

    return None, [bench.hold(frames=5), bench.mark("spin"), bench.hold(frames=60)]
