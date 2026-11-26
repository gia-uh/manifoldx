"""Streaming and the floating origin: walk a camera across 50 km of noise terrain.

A Streamer builds 32 m terrain cells in a worker around the camera and draws the
3x3 ring under it. The engine's origin follows the camera, so positions 50 km out
draw as precisely as at the start. Time is compressed: the camera covers 4 m a
frame. Every frame's wall and CPU time is logged; the walk fails if any frame
takes more than twice the median (CPU time by default, --clock wall for wall time).

    uv run python examples/stream_walk.py --km 50 --headless
    uv run python examples/stream_walk.py --km 50 --headless --workers thread
    uv run python examples/stream_walk.py --render --duration 30 --output /tmp/stream_walk.mp4
"""

import argparse
import sys
from time import perf_counter, process_time

import numpy as np

import manifoldx as mx
from manifoldx.resources import DirectionalLight
from manifoldx.streaming import MeshData, Streamer

CELL, SEGMENTS, HEIGHT = 32.0, 32, 26.0
STEP, CAM_UP, LOOK_AHEAD = 4.0, 40.0, 24.0      # metres a frame, above the ground, ahead
WARMUP = 30                                      # frames before timing starts
LOW, HIGH = np.array([0.06, 0.22, 0.04]), np.array([0.62, 0.52, 0.38])


def _hash(ix, iz):
    h = (ix * 374761393 + iz * 668265263) & 0xFFFFFFFF
    h = ((h ^ (h >> 13)) * 1274126177) & 0xFFFFFFFF
    return (h ^ (h >> 16)) / 0xFFFFFFFF


def height(x, z):
    """Value-noise fbm in world metres (float64), in [0, HEIGHT]."""
    x, z = np.asarray(x, np.float64), np.asarray(z, np.float64)
    total, amp, freq = 0.0, 0.5, 1 / 70
    for _ in range(4):
        fx, fz = x * freq, z * freq
        ix, iz = np.floor(fx).astype(np.int64), np.floor(fz).astype(np.int64)
        tx, tz = fx - ix, fz - iz
        tx, tz = tx * tx * (3 - 2 * tx), tz * tz * (3 - 2 * tz)
        a = _hash(ix, iz) * (1 - tx) + _hash(ix + 1, iz) * tx
        b = _hash(ix, iz + 1) * (1 - tx) + _hash(ix + 1, iz + 1) * tx
        total = total + amp * (a * (1 - tz) + b * tz)
        amp, freq = amp * 0.5, freq * 2.0
    return total / 0.9375 * HEIGHT


def build_cell(i, j):
    """Cell (i, j) as a mesh relative to its corner. Pure, so a process can run it."""
    n, d = SEGMENTS + 1, CELL / SEGMENTS
    g = np.arange(-1, n + 1) * d                         # one sample past each border
    gx, gz = np.meshgrid(i * CELL + g, j * CELL + g)
    h = height(gx, gz)
    nx = h[1:-1, :-2] - h[1:-1, 2:]                      # central differences: seams agree
    nz = h[:-2, 1:-1] - h[2:, 1:-1]
    normals = np.stack([nx, np.full_like(nx, 2 * d), nz], -1)
    normals /= np.linalg.norm(normals, axis=-1, keepdims=True)
    lx, lz = np.meshgrid(g[1:-1], g[1:-1])
    hh = h[1:-1, 1:-1]
    pos = np.stack([lx, hh, lz], -1).reshape(-1, 3)
    t = (hh / HEIGHT).reshape(-1, 1)
    q = np.arange(SEGMENTS)
    a = (q[:, None] * n + q[None, :]).ravel()            # top-left of each quad
    tris = np.stack([a, a + n, a + 1, a + 1, a + n, a + n + 1], -1).ravel()
    return MeshData(pos.astype(np.float32), normals.reshape(-1, 3).astype(np.float32),
                    (LOW * (1 - t) + HIGH * t).astype(np.float32), tris.astype(np.uint32))


def make(engine, workers: str = "process") -> Streamer:
    """The sky, the sun and the terrain's streamer on `engine`."""
    engine.background_color = (0.70, 0.80, 0.92)
    engine.set_sun(DirectionalLight(color="#fff4e0", intensity=3.0, direction=(-0.4, -0.7, -0.5)))
    return Streamer(engine, CELL, build_cell, workers=workers)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--km", type=float, default=50.0)
    ap.add_argument("--workers", choices=["thread", "process"], default="process")
    ap.add_argument("--headless", action="store_true", help="walk offscreen and report frame times")
    # On a shared host with a software renderer, wall time mostly measures the
    # neighbours; the CPU this process spends per frame is the engine's own work,
    # a thread worker's builds included. Both are printed.
    ap.add_argument("--clock", choices=["wall", "cpu"], default="cpu",
                    help="what the twice-the-median check reads (default: this process's CPU time)")
    ap.add_argument("--log", help="write one line per frame: frame, wall ms, cpu ms, drawn cells")
    ap.add_argument("--render", action="store_true", help="record a clip with the frame time in a corner")
    ap.add_argument("--output", default="/tmp/stream_walk.mp4")
    ap.add_argument("--duration", type=float, default=30.0)
    ap.add_argument("--fps", type=int, default=24)
    args = ap.parse_args()

    engine = mx.Engine("Stream walk", width=960, height=540)
    streamer = make(engine, args.workers)
    if not args.headless:
        from manifoldx.gui import Panel, Text

        ms_text, km_text = Text(""), Text("")  # the atlas keeps each string: round them
        engine.gui.append(Panel(children=[ms_text, km_text], anchor="top-left", offset=(12, 12),
                                       style_overrides={"width": 150, "height": 44, "padding": 6, "gap": 2}))
    st = {"frame": 0, "last": None, "wall": [], "cpu": []}

    @engine.system
    def walk(query, dt):
        now = (perf_counter(), process_time())
        if st["last"] is not None:
            ms = (now[0] - st["last"][0]) * 1000
            if st["frame"] > WARMUP:
                st["wall"].append(ms)
                st["cpu"].append((now[1] - st["last"][1]) * 1000)
            if not args.headless:
                ms_text.text = f"frame {ms:.0f} ms"
                km_text.text = f"{max(0, st['frame'] - WARMUP) * STEP / 1000:.1f} km"
        st["last"] = now
        z = -max(0, st["frame"] - WARMUP) * STEP         # walk north once the ring is warm
        st["frame"] += 1
        ox, oz = engine.origin
        if abs(z - oz) > CELL * 1.5:                     # rebase with half a cell of slack
            engine.set_origin(0.0, np.floor(z / CELL) * CELL)
            ox, oz = engine.origin
        x = CELL / 2                                     # down the middle of a column of cells
        streamer.focus(x, z)
        eye = np.array([x - ox, float(height(x, z)) + CAM_UP, z - oz])
        look = np.array([x - ox, float(height(x, z - LOOK_AHEAD)), z - oz - LOOK_AHEAD])
        engine.camera.set_pose(tuple(eye), tuple(look))

    frames = WARMUP + int(args.km * 1000 / STEP)
    if args.render:
        engine.render(args.output, fps=args.fps, duration=args.duration)
        streamer.close()
        return 0
    if not args.headless:
        engine.run()
        streamer.close()
        return 0

    log = open(args.log, "w") if args.log else None
    for f in range(frames + 1):
        engine.render_frame(supersample=1)
        if log and st["wall"]:
            log.write(f"{f},{st['wall'][-1]:.3f},{st['cpu'][-1]:.3f},{len(streamer.drawn())}\n")
    streamer.close()
    for clock in ("wall", "cpu"):
        t = np.array(st[clock])
        print(f"{args.workers} {clock}: {len(t)} frames over {args.km:g} km, median {np.median(t):.2f} ms, "
              f"p99 {np.percentile(t, 99):.2f} ms, worst {t.max():.2f} ms at frame {int(t.argmax()) + WARMUP + 1}")
    print(f"{args.workers}: {streamer.stats()}")
    t = np.array(st[args.clock])
    med, worst = float(np.median(t)), float(t.max())
    if worst > 2 * med:
        print(f"FAIL: worst {args.clock} frame {worst:.2f} ms is over twice the median ({2 * med:.2f} ms)")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
