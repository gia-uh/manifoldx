"""manifoldx.streaming.QuadStreamer: nodes picked by screen error, drawn once, parents kept until children are."""

import threading
import time

import numpy as np

import manifoldx as mx
from manifoldx.streaming import MeshData, QuadStreamer

CELL = 200.0


def tri(*_):
    pos = np.array([[0, 0, 0], [8, 0, 0], [0, 0, 8]], np.float32)
    return MeshData(pos, np.tile([0, 1, 0], (3, 1)).astype(np.float32), np.full((3, 3), 0.5, np.float32),
                    np.array([0, 2, 1], np.uint32))


def full(levels):
    return lambda L, a, b: 0 <= a < (1 << (levels - L)) and 0 <= b < (1 << (levels - L))


def streamer(levels, build=tri, origin=(0, 0)):
    engine = mx.Engine("quad", width=64, height=600)
    engine.camera.far = 15000.0
    return QuadStreamer(engine, CELL, origin, levels, full(levels), build, lambda L: CELL * 2**L / 16)


def test_selection_tiles_the_content():
    s = streamer(6)
    s.engine.camera.far = 1e6  # the far corner, 18 km away, stays in view
    picked = s._select(1.0, 1.0, 2.0)
    covered = np.zeros((64, 64), int)
    for L, a, b in picked:
        covered[a << L:(a + 1) << L, b << L:(b + 1) << L] += 1
    assert (covered == 1).all()  # disjoint, and every cell drawn
    assert (0, 0, 0) in picked
    far = next(n for n in picked if n[1] << n[0] <= 63 < (n[1] + 1) << n[0] and n[2] << n[0] <= 63 < (n[2] + 1) << n[0])
    assert far[0] >= 3
    s.close()


def test_merge_uses_hysteresis():
    s = streamer(6)
    node = (3, 2, 2)
    focal = s.engine.h / (2 * np.tan(np.radians(s.engine.camera.fov) / 2))
    x0, z0, side = s._box(node)
    # A height where the node's error is 5 px: under tau (6), over tau / 1.5 (4).
    h = s.error(3) * focal / 5.0
    s._selected = set()
    assert node not in s._select(x0 + side / 2, z0 + side / 2, h)  # refined: not merged until under 4 px
    s._selected = {node}
    assert node in s._select(x0 + side / 2, z0 + side / 2, h)  # once picked, it stays until over 6 px
    s.close()


def test_high_camera_is_bounded():
    s = streamer(13)
    assert len(s._select(800_000.0, 800_000.0, 5000.0)) < 2000
    s.close()


def test_negative_cells():
    s = streamer(13, origin=(-4096, -4096))
    x0, z0, side = s._box((3, (-1 + 4096) >> 3, (-1 + 4096) >> 3))
    assert x0 <= -CELL < x0 + side and z0 <= -CELL < z0 + side
    s.close()


def test_parent_stays_until_children_ready():
    gate = threading.Event()

    def build(L, a, b):
        if L == 0:
            gate.wait(5)
        return tri()

    s = streamer(1, build)
    for _ in range(200):  # high up: the root alone, built and drawn
        s.focus(200.0, 200.0, 10_000.0)
        if (1, 0, 0) in s.drawn():
            break
        time.sleep(0.005)
    assert s.drawn() == {(1, 0, 0)}
    for _ in range(20):  # down low: the children are picked but not built, so the root stays
        s.focus(200.0, 200.0, 1.0)
        time.sleep(0.005)
    assert s.drawn() == {(1, 0, 0)}
    gate.set()
    for _ in range(400):
        s.focus(200.0, 200.0, 1.0)
        if (1, 0, 0) not in s.drawn():
            break
        time.sleep(0.005)
    assert s.drawn() == {(0, 0, 0), (0, 0, 1), (0, 1, 0), (0, 1, 1)}  # all four at once, the root gone
    s.close()
