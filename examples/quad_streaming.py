"""Quadtree streaming: a plain of boxes 100 km wide, seen from a camera that rises to 6 km and comes back down.

A QuadStreamer picks, every frame, the nodes of a quadtree whose error on screen (a sixteenth of the node's
side, seen from the camera) is under 6 pixels: single 100 m cells under the camera, nodes of many kilometres
at the horizon. Each node is one box per square of a 4 x 4 grid, so a far node costs as little as a near one.
The engine's origin follows the camera.

    uv run python examples/quad_streaming.py
    uv run python examples/quad_streaming.py --render --duration 2 --output /tmp/quad_streaming.mp4
"""

import argparse
import math

import numpy as np

import manifoldx as mx
from manifoldx.resources import DirectionalLight
from manifoldx.streaming import MeshData, QuadStreamer

CELL, LEVELS, GRID = 100.0, 10, 4  # 2^10 cells of 100 m: 102.4 km a side


def build_node(L, a, b):
    """GRID x GRID boxes on node (L, a, b), each as tall as a hash of its square says."""
    side = CELL * 2**L / GRID
    pos, nrm, col, idx = [], [], [], []
    for u in range(GRID):
        for v in range(GRID):
            h = ((a * GRID + u) * 73856093 ^ (b * GRID + v) * 19349663 ^ L * 83492791) % 1000 / 1000
            x0, z0, w, top = u * side + side * 0.2, v * side + side * 0.2, side * 0.6, 5 + 40 * h * h
            c = (0.55 + 0.4 * h, 0.55, 0.6 - 0.3 * h)
            quads = [((x0, top, z0), (x0 + w, top, z0), (x0 + w, top, z0 + w), (x0, top, z0 + w), (0, 1, 0)),
                     ((x0, 0, z0 + w), (x0 + w, 0, z0 + w), (x0 + w, top, z0 + w), (x0, top, z0 + w), (0, 0, 1)),
                     ((x0 + w, 0, z0), (x0, 0, z0), (x0, top, z0), (x0 + w, top, z0), (0, 0, -1)),
                     ((x0, 0, z0), (x0, 0, z0 + w), (x0, top, z0 + w), (x0, top, z0), (-1, 0, 0)),
                     ((x0 + w, 0, z0 + w), (x0 + w, 0, z0), (x0 + w, top, z0), (x0 + w, top, z0 + w), (1, 0, 0))]
            for *corners, n in quads:
                k = len(pos)
                pos += corners
                nrm += [n] * 4
                col += [c] * 4
                idx += [k, k + 2, k + 1, k, k + 3, k + 2]
    return MeshData(np.array(pos, np.float32), np.array(nrm, np.float32), np.array(col, np.float32),
                    np.array(idx, np.uint32))


def build(engine, *, fly: bool = True) -> QuadStreamer:
    """The plain and its streamer on `engine`; with fly, a system that flies the camera up to 6 km and back."""
    engine.background_color = (0.70, 0.80, 0.92)
    engine.camera.near, engine.camera.far = 0.5, 30000.0
    engine.set_sun(DirectionalLight(color="#fff4e0", intensity=3.0, direction=(-0.4, -0.7, -0.5)))
    streamer = QuadStreamer(engine, CELL, (0, 0), LEVELS, lambda L, a, b: True, build_node,
                            error=lambda L: CELL * 2**L / 16)
    if not fly:
        return streamer
    st = {"t": 0.0}

    @engine.system
    def fly_camera(query, dt):
        st["t"] += dt
        t = st["t"]
        x, z = 51_200.0 + 300.0 * t, 51_200.0
        y = 20.0 + 3000.0 * (1 - math.cos(t * 0.3))  # up to 6 km and back
        if math.hypot(x - engine.origin[0], z - engine.origin[1]) > 2000:
            engine.set_origin(x, z)
        ox, oz = engine.origin
        streamer.focus(x, z, y)
        engine.camera.set_pose((x - ox, y, z - oz), (x - ox + 400.0, y * 0.6, z - oz - 200.0))

    return streamer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--render", action="store_true")
    ap.add_argument("--output", default="/tmp/quad_streaming.mp4")
    ap.add_argument("--duration", type=float, default=20.0)
    args = ap.parse_args()

    engine = mx.Engine("Quad streaming", width=960, height=540)
    streamer = build(engine)
    if args.render:
        engine.render(args.output, fps=24, duration=args.duration)
    else:
        engine.run()
    print(streamer.stats())
    streamer.close()


if __name__ == "__main__":
    main()
