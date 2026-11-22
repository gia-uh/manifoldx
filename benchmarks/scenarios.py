"""manifoldx's own scenarios for `manifoldx bench run`, over its examples and benchmark scenes.

    uv run manifoldx bench run quad_zoom --size 960x540
    uv run manifoldx bench run meshes_spin --loop pipelined
    uv run manifoldx bench run stream_walk
    uv run manifoldx bench check                # the scenarios with checks, at smoke size
"""

import time

import numpy as np

import manifoldx as mx
from manifoldx import bench
from manifoldx.components import Component, Material, Mesh, Transform
from manifoldx.types import Vector3
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


_BUILD_S = [0.0]  # what quad_arrive's node build sleeps, as a real build takes that long


def _slow_node(L, a, b):
    from examples.quad_streaming import build_node

    time.sleep(_BUILD_S[0])
    return build_node(L, a, b)


@bench.scenario("quad_arrive", params={"build_ms": 20.0, "builders": 2}, sizes={"smoke": 20, "full": None})
def quad_arrive(engine, p):
    """The quad streaming plain at street height, looking along it, each node costing build_ms to build:
    `near_s` is how long after the camera arrives every node it draws within two cells of it is drawn."""
    from manifoldx.streaming import QuadStreamer

    from examples.quad_streaming import CELL, LEVELS

    _BUILD_S[0] = p.build_ms / 1000.0
    engine.camera.near, engine.camera.far = 0.5, 30000.0
    streamer = QuadStreamer(engine, CELL, (0, 0), LEVELS, lambda L, a, b: True, _slow_node,
                            error=lambda L: CELL * 2**L / 16, builders=int(p.builders))
    radius = 2 * CELL
    at = {"t0": None, "near_s": None}

    @engine.system
    def follow(query, dt):
        ox, oz = engine.origin
        x, y, z = (float(v) for v in engine.camera.position)
        streamer.focus(x + ox, z + oz, y)
        if at["t0"] is None:
            if abs(x + ox - AT[0]) + abs(z + oz - AT[1]) > 1.0:
                return  # the frames before the camera step put it there
            at["t0"] = time.perf_counter()
        near = [n for n in streamer._selected if streamer._distance(n, x + ox, z + oz, 0.0) <= radius]
        if at["near_s"] is None and near and all(n in streamer.drawn() for n in near):
            at["near_s"] = time.perf_counter() - at["t0"]
        if at["near_s"] is not None:
            bench.metric("near_s", round(at["near_s"], 3))

    return streamer, [
        bench.camera((AT[0], 2.0, AT[1]), (AT[0] + 400.0, 1.0, AT[1])),
        bench.mark("arrive"),
        bench.hold(until=bench.settled(streamer), max_s=120),
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
        with bench.span("busy"):
            while time.perf_counter() < deadline:
                data = np.sin(data) * 0.5 + 0.5
        bench.metric("cubes", p.n)

    return None, [bench.hold(frames=5), bench.mark("spin"), bench.hold(frames=60)]


meshes_spin.check(counters_draws=1, phase="spin")  # every cube in one instanced draw


def _walk(engine, workers: str, z0: float):
    """The stream walk terrain from z0, the floating origin following the camera; (streamer, place, walk)."""
    from examples.stream_walk import CAM_UP, CELL, LOOK_AHEAD, STEP, height, make

    streamer = make(engine, workers)
    at = {"z": z0}

    def place():
        x, z = CELL / 2, at["z"]
        if abs(z - engine.origin[1]) > CELL * 1.5:  # rebase with half a cell of slack, as the example does
            engine.set_origin(0.0, float(np.floor(z / CELL) * CELL))
        ox, oz = engine.origin
        streamer.focus(x, z)
        eye = (x - ox, float(height(x, z)) + CAM_UP, z - oz)
        engine.camera.set_pose(eye, (x - ox, float(height(x, z - LOOK_AHEAD)), z - oz - LOOK_AHEAD))

    def walk(app):
        at["z"] -= STEP
        place()

    place()
    return streamer, place, walk


# A spawned worker process takes about a second to start: at smoke size the first hold must last that long.
@bench.scenario("stream_walk", params={"workers": "process", "metres": 2000.0}, sizes={"smoke": 200, "full": None})
def stream_walk(engine, p):
    """The stream walk terrain, built in a worker process, walked north 4 m a frame for `metres`."""
    from examples.stream_walk import STEP

    streamer, place, walk = _walk(engine, p.workers, 0.0)
    return streamer, [
        bench.each_frame(lambda app: place(), until=bench.settled(streamer), max_s=60),
        bench.mark("walk"),
        bench.each_frame(walk, frames=int(p.metres / STEP)),
    ]


@bench.scenario("far_walk", params={"workers": "process"}, sizes={"smoke": 200, "full": None})
def far_walk(engine, p):
    """The stream walk 900 km from the world's origin: the floating origin keeps what is drawn near."""
    streamer, place, walk = _walk(engine, p.workers, -900_000.0)
    return streamer, [
        bench.each_frame(lambda app: place(), until=bench.settled(streamer), max_s=60),
        bench.mark("far"),
        bench.each_frame(walk, frames=30),
    ]


far_walk.check(origin_max_m=250, phase="far")  # 101 m measured at smoke size; 900 km without the origin


class Stroll(Component):
    v: Vector3  # m/s along the ground


@bench.scenario("humans", params={"n": 500, "levels": "on", "pose": "gpu"}, sizes={"smoke": 20, "full": None})
def humans(engine, p):
    """n walking figures in rows 3 to 60 m away, 24 bodies at three walking speeds shared among them.

    pose "gpu": each body and speed is a baked 32-frame clip, every walker plays its own phase of it and the
    GPU blends two frames; "cpu": every figure is posed in Python each frame it is due (levels "on" picks the
    level and pose rate by distance, "off" poses every figure at level 0 each frame).
    """
    from manifoldx.humanoid.cycles import (bake_cycle, cadence, cycle_angles, pose_matrices, root_offset,
                                           snap_phase, update_every)
    from manifoldx.modeling.assets.human import HumanParams, build_human
    from manifoldx.modeling.assets.human_dress import Outfit

    g = np.random.default_rng(5)
    bodies = [HumanParams.sample(i) for i in range(24)]
    speeds = (1.2, 1.4, 1.6)
    levels = (0, 1, 2) if p.levels == "on" else (0,)
    figures = {(b, lv): build_human(bodies[b], outfit=Outfit(), level=lv) for b in range(24) for lv in levels}
    geometry = {k: f.mesh.to_geometry() for k, f in figures.items()}
    skin = StandardMaterial("#ffffff", roughness=0.7, vertex_colors=True)
    dist = np.linspace(3.0, 60.0, p.n)
    level = np.where(dist < 8, 0, np.where(dist < 20, 1, 2)) if p.levels == "on" else np.zeros(p.n, int)
    body = np.arange(p.n) % 24
    speed_k = g.integers(len(speeds), size=p.n)
    speed = np.array(speeds)[speed_k]
    rate = np.array([cadence("walk", speed[i], bodies[body[i]]) for i in range(p.n)])
    phase = g.random(p.n)
    index = np.empty(p.n, np.int64)
    for i in range(p.n):
        h = engine.spawn(Mesh(geometry[(int(body[i]), int(level[i]))]), Material(skin),
                         Transform(pos=(float(g.uniform(-15, 15)), 0.0, float(dist[i]))),
                         Stroll(v=(float(speed[i]), 0.0, 0.0)))
        index[i] = h.index
    engine.camera.set_pose(position=(0.0, 2.0, -2.0), target=(0.0, 1.0, 20.0))
    frames = 32
    if p.pose == "gpu":
        with bench.span("bake"):
            clips = {(b, k): engine.bake_poses(bake_cycle(figures[(b, 0)], "walk", speeds[k], frames))
                     for b in range(24) for k in range(len(speeds))}
        clip = np.array([clips[(int(body[i]), int(speed_k[i]))] for i in range(p.n)])
    cache, clock = {}, {"frame": 0}

    @engine.system
    def walk(q: mx.Query[Transform, Stroll], dt: float):
        q[Transform].pos += q[Stroll].v * dt
        phase[:] = (phase + rate * dt) % 1.0
        frame = clock["frame"]
        clock["frame"] = frame + 1
        with bench.span("pose"):
            if p.pose == "gpu":
                engine.set_pose_frames(index, clip, phase * frames)
                bench.metric("posed", p.n)
                return
            due, poses = [], []
            for i in range(p.n):
                lv = int(level[i])
                if frame % update_every(lv):
                    continue
                f = figures[(int(body[i]), lv)]
                ph = snap_phase(float(phase[i]), lv)
                key = (id(f), ph, int(speed_k[i])) if lv == 2 else None
                m = cache.get(key) if key else None
                if m is None:
                    m = pose_matrices(f, cycle_angles("walk", ph, speed[i], f.params), root_offset("walk", ph, speed[i], f.params))
                    if key:
                        cache[key] = m
                due.append(index[i])
                poses.append(m)
        with bench.span("upload"):
            engine.set_joints(due, np.stack(poses))
        bench.metric("posed", len(due))

    return None, [bench.hold(frames=5), bench.mark("walk"), bench.hold(frames=60)]
