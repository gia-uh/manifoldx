"""Streaming: how soon a QuadStreamer shows detail and settles while the camera zooms in, on a slow build."""

from __future__ import annotations

import time

from benchmarks.harness import case


def _tri():
    import numpy as np

    from manifoldx.streaming import MeshData

    p = np.array([[0, 0, 0], [8, 0, 0], [0, 0, 8]], np.float32)
    return MeshData(p, np.tile([0, 1, 0], (3, 1)).astype(np.float32), np.full((3, 3), 0.5, np.float32),
                    np.array([0, 2, 1], np.uint32))


@case("streaming", "quad_zoom", sizes=[6, 8, 10], smoke=3)
def quad_zoom(levels):
    """A zoom from the whole tree of `levels` levels to 20 m in 1 s, 3 builders sleeping 20 ms a node; ms to settle."""
    import manifoldx as mx
    from manifoldx.streaming import QuadStreamer

    ms = 20.0 if levels > 3 else 1.0  # the smoke size runs in a blink
    frame = 16.0 if levels > 3 else 1.0

    def build(L, a, b):
        time.sleep(ms / 1000.0)
        return _tri()

    engine = mx.Engine("quad_zoom", width=64, height=600)
    engine.camera.far = 1e6
    cell = 100.0
    s = QuadStreamer(engine, cell, (0, 0), levels, lambda L, a, b: True, build,
                     error=lambda L: cell * 2**L / 16, builders=3)
    mid = cell * 2 ** (levels - 1)
    top, low = cell * 2**levels * 4, 20.0

    def settled():
        return s._quiet and s.drawn() == set(s._selected)

    for _ in range(int(20_000 / frame)):  # the opening view, not timed
        s.focus(mid, mid, top)
        if settled():
            break
        time.sleep(frame / 1000.0)
    coarsest = min(L for L, _a, _b in s.drawn())
    built0 = s.stats()["built"]
    steps = int(1000 / frame)
    t0 = time.perf_counter()
    first = None
    k = 0
    while True:
        h = top * (low / top) ** min(1.0, k / steps)
        s.focus(mid, mid, h)
        now = (time.perf_counter() - t0) * 1000.0
        if first is None and any(L < coarsest for L, _a, _b in s.drawn()):
            first = now
        if k >= steps and settled():
            break
        if now > 120_000:
            raise RuntimeError("never settled")
        k += 1
        time.sleep(frame / 1000.0)
    out = {"ms": now, "first_ms": round(first, 1), "builds": s.stats()["built"] - built0, "nodes": len(s.drawn())}
    s.close()
    return out
