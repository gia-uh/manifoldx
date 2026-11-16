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


def streamer(levels, build=tri, origin=(0, 0), refine_ms=None):
    engine = mx.Engine("quad", width=64, height=600)
    engine.camera.far = 15000.0
    return QuadStreamer(engine, CELL, origin, levels, full(levels), build, lambda L: CELL * 2**L / 16,
                        refine_ms=refine_ms)


def focal(s):
    return s.engine.h / (2 * np.tan(np.radians(s.engine.camera.fov) / 2))


def tiles(s, levels):
    """The cut covers every cell exactly once."""
    covered = np.zeros((1 << levels, 1 << levels), int)
    for L, a, b in s._cut:
        covered[a << L:(a + 1) << L, b << L:(b + 1) << L] += 1
    return (covered == 1).all()


def settled(s, x, z, h):
    """Every node of the cut is one the rules keep at this camera: none splits, no four siblings merge."""
    f = focal(s)
    for L, a, b in s._cut:
        d = s._distance((L, a, b), x, z, h)
        if L > 0 and s.error(L) * f / d > s.tau:
            return False
        if L < s.levels:
            p = (L + 1, a >> 1, b >> 1)
            if all(k in s._cut for k in s._children(p)) and s.error(L + 1) * f / s._distance(p, x, z, h) <= s.tau / s.merge:
                return False
    return True


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
    x0, z0, side = s._box(node)
    cx, cz = x0 + side / 2, z0 + side / 2
    # A height where the node's error is 5 px: under tau (6), over tau / 1.5 (4).
    h = s.error(3) * focal(s) / 5.0
    s._select(cx, cz, 1.0)  # low: refined below the node
    assert node not in s._select(cx, cz, h)  # its children stay until it is under 4 px
    s._select(cx, cz, 1e6)  # high: coarser than the node
    assert node in s._select(cx, cz, h)  # split down to it, and it stays while under 6 px
    s.close()


def test_refined_cut_tiles_and_settles():
    s = streamer(7)
    s.engine.camera.far = 1e6
    rng = np.random.default_rng(1)
    for _ in range(20):
        x, z = rng.uniform(-2000, 27600, 2)
        h = float(rng.choice([1.0, 50.0, 400.0, 3000.0]))
        s._select(x, z, h)
        assert tiles(s, 7) and settled(s, x, z, h)
    s.close()


def test_budget_spreads_the_refinement():
    s = streamer(7, refine_ms=0.0)  # 16 checks a frame, the floor
    s.engine.camera.far = 1e6
    s._select(15000.0, 15000.0, 1.0)  # the first focus refines to the end, wherever it is
    assert settled(s, 15000.0, 15000.0, 1.0)
    frames = 0
    while True:
        before = set(s._cut or ())
        s._select(100.0, 100.0, 1.0)
        frames += 1
        assert tiles(s, 7)  # every frame draws a whole tiling, refined or not
        if s._cut == before and not s._queue:
            break
    assert frames > 5 and settled(s, 100.0, 100.0, 1.0)  # reached over several frames
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


def _settle(s, x, z, h, tries=600):
    for _ in range(tries):
        s.focus(x, z, h)
        if not s._pending and s._quiet:
            return
        time.sleep(0.002)
    raise AssertionError("never settled")


def test_still_camera_skips_selection(monkeypatch):
    s = streamer(6)
    _settle(s, 500.0, 500.0, 2.0)
    calls = {"n": 0}
    real = QuadStreamer._distance

    def counting(self, *a):
        calls["n"] += 1
        return real(self, *a)

    monkeypatch.setattr(QuadStreamer, "_distance", counting)
    drawn = s.drawn()
    s.focus(500.0, 500.0, 2.0)
    assert calls["n"] == 0 and s.drawn() == drawn
    s.close()


def test_build_finishing_while_still_is_drawn():
    gate = threading.Event()

    def build(L, a, b):
        if (L, a, b) == (0, 0, 0):
            gate.wait(5)
        return tri()

    s = streamer(2, build)
    for _ in range(50):
        s.focus(100.0, 100.0, 1.0)
        time.sleep(0.002)
    assert (0, 0, 0) not in s.drawn()
    gate.set()
    for _ in range(400):
        s.focus(100.0, 100.0, 1.0)
        if (0, 0, 0) in s.drawn():
            break
        time.sleep(0.005)
    assert (0, 0, 0) in s.drawn()
    s.close()


def test_fov_change_while_still_reselects():
    s = streamer(6)
    _settle(s, 500.0, 500.0, 2.0)
    before = set(s._cut)
    s.engine.camera.fov /= 4  # four times the zoom: far nodes now show more error and split
    s.focus(500.0, 500.0, 2.0)
    for _ in range(50):
        s.focus(500.0, 500.0, 2.0)
    assert len(s._cut) > len(before)
    s.close()


def test_distances_match_scalar():
    s = streamer(8)
    rng = np.random.default_rng(0)
    nodes = [(int(L), int(rng.integers(0, 1 << (8 - L))), int(rng.integers(0, 1 << (8 - L))))
             for L in rng.integers(0, 9, 500)]
    want = [s._distance(n, 1234.5, -300.0, 7.0) for n in nodes]
    assert np.allclose(s._distances(nodes, 1234.5, -300.0, 7.0), want)
    s.close()
