"""What an app waits for when it starts: its first frame and its first streamed cell, after it loaded for a while.

Each run is a fresh Python process, because what is measured happens once per process: cffi resolving wgpu's C
types and a streamer's workers starting. The size is the app's own loading time in ms, spent between making the
Engine (or the streamer) and asking for the frame (or the cell); work the engine does in the background during it
does not show in the time.
"""

import json
import os
import subprocess
import sys

from benchmarks.harness import REPEAT, case

FIRST_FRAME = """
import json, sys, time
import manifoldx as mx
engine = mx.Engine("startup", width=64, height=64)
from manifoldx.components import Material, Mesh, Transform
from manifoldx.modeling.mesh import Mesh as M
from manifoldx.resources import StandardMaterial
engine.spawn(Mesh(M.box(1, 1, 1).to_geometry()), Material(StandardMaterial("#808080")), Transform(pos=(0, 0, -3)))
time.sleep({gap})
t = time.perf_counter()
engine.render_frame(supersample=1)
print(json.dumps({{"ms": 1e3 * (time.perf_counter() - t)}}))
"""

FIRST_CELL = """
import json, sys, time
import manifoldx as mx
from manifoldx.streaming import Streamer
from benchmarks.suites.startup import one_triangle
engine = mx.Engine("startup", width=64, height=64)
s = Streamer(engine, 100.0, one_triangle, radius=0, ready_radius=0, workers="process")
time.sleep({gap})
t = time.perf_counter()
while (0, 0) not in s.drawn() and time.perf_counter() - t < 60:
    s.focus(50.0, 50.0)
    engine.render_frame(supersample=1)
print(json.dumps({{"ms": 1e3 * (time.perf_counter() - t)}}))
s.close()
"""


def one_triangle(i, j):
    """A cell's mesh: one triangle; the worker's cost is starting, not building."""
    import numpy as np

    from manifoldx.streaming import MeshData

    p = np.array([[0, 0, 0], [8, 0, 0], [0, 0, 8]], np.float32)
    return MeshData(p, np.tile([0, 1, 0], (3, 1)).astype(np.float32), np.full((3, 3), 0.5, np.float32),
                    np.array([0, 2, 1], np.uint32))


def _fresh(script: str, gap_ms: int) -> float:
    """Best over REPEAT fresh processes (at most 3) of the ms the script prints."""
    best = None
    for _ in range(min(REPEAT, 3)):
        out = subprocess.run([sys.executable, "-c", script.format(gap=gap_ms / 1000)], capture_output=True,
                             text=True, check=True, env={**os.environ, "PYTHONPATH": os.pathsep.join(sys.path)})
        ms = json.loads(out.stdout.strip().splitlines()[-1])["ms"]
        best = ms if best is None else min(best, ms)
    return best


@case("startup", "first_frame", sizes=[0, 1000, 3000], smoke=0)
def first_frame(gap_ms):
    """The first render_frame of a one-box scene, after the app loads for `n` ms."""
    return _fresh(FIRST_FRAME, gap_ms)


@case("startup", "first_cell", sizes=[0, 1000, 3000], smoke=0)
def first_cell(gap_ms):
    """From the first focus() to the focus cell drawn, process workers, after the app loads for `n` ms."""
    return _fresh(FIRST_CELL, gap_ms)
