"""manifoldx.streaming.QuadStreamer: nodes picked by screen error, drawn once, parents kept until children are."""

import threading
import time

import numpy as np

import manifoldx as mx
from manifoldx.commands import CommandType
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


def test_small_moves_keep_the_settled_cut_and_order(monkeypatch):
    s = streamer(6)
    _settle(s, 500.0, 500.0, 2.0)
    calls = {"n": 0}
    real = QuadStreamer._check

    def counting(self, *a):
        calls["n"] += 1
        return real(self, *a)

    monkeypatch.setattr(QuadStreamer, "_check", counting)
    picked = s._select(500.0, 500.0, 2.0)
    for k in range(1, 20):  # 1 m a frame, under move (25 m)
        assert s._select(500.0 + k, 500.0, 2.0) is picked
    assert calls["n"] == 0
    s._select(500.0 + s.move, 500.0, 2.0)  # a full move: a new pass checks the cut again
    assert calls["n"] > 0
    s.close()


def test_moving_camera_with_nothing_new_skips_the_draw_set(monkeypatch):
    s = streamer(6)
    _settle(s, 500.0, 500.0, 2.0)
    drawn = s.drawn()
    calls = {"n": 0}
    real = QuadStreamer._on_gpu

    def counting(self, *a):
        calls["n"] += 1
        return real(self, *a)

    monkeypatch.setattr(QuadStreamer, "_on_gpu", counting)
    for k in range(1, 20):
        s.focus(500.0 + k, 500.0, 2.0)
    assert calls["n"] == 0 and s.drawn() == drawn
    s.close()


def test_build_finishing_while_moving_is_drawn():
    gate = threading.Event()

    def build(L, a, b):
        if (L, a, b) == (0, 0, 0):
            gate.wait(5)
        return tri()

    s = streamer(2, build)
    for k in range(50):
        s.focus(100.0 + k * 0.01, 100.0, 1.0)
        time.sleep(0.002)
    assert (0, 0, 0) not in s.drawn()
    gate.set()
    for k in range(400):
        s.focus(100.5 + k * 0.01, 100.0, 1.0)
        if (0, 0, 0) in s.drawn():
            break
        time.sleep(0.005)
    assert (0, 0, 0) in s.drawn()
    s.close()


def test_tau_change_while_still_reselects():
    s = streamer(6)
    _settle(s, 500.0, 500.0, 2.0)
    before = len(s._cut)
    s.tau *= 4  # four times the error allowed: near nodes merge
    for _ in range(50):
        s.focus(500.0, 500.0, 2.0)
    assert len(s._cut) < before
    s.close()


def test_crossfade_keeps_the_parent_till_its_children_fade_in():
    gate = threading.Event()

    def build(L, a, b):
        if L == 0:
            gate.wait(5)
        return tri()

    engine = mx.Engine("quad", width=64, height=600)
    engine.camera.far = 15000.0
    s = QuadStreamer(engine, CELL, (0, 0), 1, full(1), build, lambda L: CELL * 2**L / 16, refine_ms=None,
                     crossfade=0.2)
    assert s.material.faded

    def fade(n):
        handle = s._fading[n][0] if n in s._fading else s._drawn[n]
        return engine.store._components["Fade"][handle.index][0]

    for _ in range(200):
        s.focus(200.0, 200.0, 10_000.0)
        if (1, 0, 0) in s.drawn():
            break
        time.sleep(0.005)
    time.sleep(0.25)
    s.focus(200.0, 200.0, 10_000.0)
    assert fade((1, 0, 0)) == 1.0 and not s._fading  # faded in, whole
    gate.set()
    kids = {(0, 0, 0), (0, 0, 1), (0, 1, 0), (0, 1, 1)}
    for _ in range(400):
        s.focus(200.0, 200.0, 1.0)
        if s.drawn() == kids:
            break
        time.sleep(0.005)
    assert s.drawn() == kids and (1, 0, 0) in s._fading  # the root still drawn, fading out
    time.sleep(0.08)
    s.focus(200.0, 200.0, 1.0)
    out, ins = fade((1, 0, 0)), [fade(k) for k in kids]
    assert -1.0 < out < 0.0 and all(0.0 < v < 1.0 for v in ins)
    assert all(abs(-out - (1.0 - v)) < 0.05 for v in ins)  # complementary: the root's pixels are the kids' gaps
    root = s._fading[(1, 0, 0)][0]
    time.sleep(0.2)
    s.focus(200.0, 200.0, 1.0)
    assert not s._fading and all(fade(k) == 1.0 for k in kids)
    assert (1, 0, 0) in s._ready and (1, 0, 0) not in s.drawn()  # faded out, kept built in the cache
    destroyed = [i for c in engine.commands._commands if c.type == CommandType.DESTROY for i in c.data["indices"]]
    assert root.index in destroyed  # gone at the frame's flush
    s.close()


def test_crossfade_needs_a_faded_material():
    import pytest
    from manifoldx.resources import StandardMaterial

    engine = mx.Engine("quad", width=64, height=600)
    with pytest.raises(ValueError):
        QuadStreamer(engine, CELL, (0, 0), 1, full(1), tri, lambda L: CELL, crossfade=0.3,
                     material=StandardMaterial("#fff", vertex_colors=True))


class FakePool:
    """Stands in for the build pool: jobs wait until the test finishes them, in the order it likes."""

    def __init__(self):
        self.jobs = {}  # node -> Future, in the order submitted
        self.submitted = []

    def submit(self, fn, *args):
        from concurrent.futures import Future

        f = Future()
        f.job = (fn, args)
        node = tuple(args[1:])
        self.jobs[node] = f
        self.submitted.append(node)
        return f

    def finish(self, keep=lambda n: False):
        """Run every waiting job except those `keep` holds back."""
        for node, f in list(self.jobs.items()):
            if f.cancelled():
                del self.jobs[node]
            elif not keep(node):
                fn, args = f.job
                f.set_result(fn(*args))
                del self.jobs[node]

    def shutdown(self, **_):
        pass


def faked(levels, **kw):
    engine = mx.Engine("quad", width=64, height=600)
    engine.camera.far = 15000.0
    s = QuadStreamer(engine, CELL, (0, 0), levels, full(levels), tri, lambda L: CELL * 2**L / 16,
                     refine_ms=None, **kw)
    s._pool.shutdown()
    s._pool = FakePool()
    return s


def covering(s, cell):
    """The level of the drawn node over cell (i, j)."""
    return [L for L, a, b in s.drawn() if a == cell[0] >> L and b == cell[1] >> L]


def test_each_level_shows_as_soon_as_it_is_whole():
    s = faked(3)
    s.in_flight = 1000  # the level-0 builds held back below take no room from the coarser ones
    for _ in range(12):  # from high up the root is the view
        s.focus(100.0, 100.0, 1e4)
        s._pool.finish()
    assert s.drawn() == {(3, 0, 0)}
    seen = []
    for _ in range(12):  # zoom in: every build but the cells lands
        s.focus(100.0, 100.0, 1.0)
        s._pool.finish(keep=lambda n: n[0] == 0)
        lv = covering(s, (0, 0))
        if lv and (not seen or seen[-1] != lv[0]):
            seen.append(lv[0])
    assert seen == [3, 2, 1]  # the root, then each level under the camera as soon as it is whole
    assert all(L > 0 for L, _a, _b in s.drawn()) and (3, 0, 0) not in s.drawn()
    for _ in range(12):
        s.focus(100.0, 100.0, 1.0)
        s._pool.finish()
    assert covering(s, (0, 0)) == [0] and s.drawn() == set(s._selected) and s._quiet
    s.close()


def test_a_bare_spot_asks_for_the_level_above_its_nodes_not_the_root():
    s = faked(3)
    s.in_flight = 1000
    s.focus(100.0, 100.0, 1.0)  # nothing drawn yet, as after a teleport
    asked = set(s._pool.jobs)
    assert (3, 0, 0) not in asked and (1, 0, 0) in asked and (0, 0, 0) in asked
    s._pool.finish(keep=lambda n: n[0] == 0)
    s.focus(100.0, 100.0, 1.0)
    assert covering(s, (0, 0)) == [1]  # the first thing under the camera is one level above its cells
    s.close()


def test_one_missing_node_keeps_only_its_parent():
    s = faked(3)
    s.in_flight = 1000
    late = (0, 0, 0)
    for _ in range(20):
        s.focus(100.0, 100.0, 1.0)
        s._pool.finish(keep=lambda n: n == late)
    parent = (1, 0, 0)
    assert late in s._selected and parent not in s._selected
    under = {n for n in s._selected if n[0] < 1 and (n[1] >> 1, n[2] >> 1) == parent[1:]}
    assert s.drawn() == (set(s._selected) - under) | {parent}  # everything else drawn at its own level
    s.close()


def test_builds_go_by_error_on_screen_a_few_at_a_time():
    s = faked(4, builders=2)
    x, z, h = 100.0, 100.0, 1.0
    for _ in range(400):
        s.focus(x, z, h)
        assert len(s._pool.jobs) <= 4  # twice the builders in flight, never a backlog
        if s._quiet:
            break
        first = next(iter(s._pool.jobs), None)  # one job finishes a frame, the oldest
        if first is not None:
            s._pool.finish(keep=lambda n: n != first)
    assert s._quiet and s.drawn() == set(s._selected)
    f = focal(s)
    err = [s.error(n[0]) * f / s._distance(n, x, z, h) for n in s._pool.submitted if n in s._selected]
    assert err == sorted(err, reverse=True)  # the selected nodes went most error on screen first
    assert len(s._pool.submitted) == len(set(s._pool.submitted))  # none built twice
    s.close()


def test_zooming_back_out_finds_the_coarse_nodes_built():
    counts = {}

    def build(L, a, b):
        counts[(L, a, b)] = counts.get((L, a, b), 0) + 1
        return tri()

    for cache, root_builds in ((256, 1), (0, 2)):
        counts.clear()
        engine = mx.Engine("quad", width=64, height=600)
        engine.camera.far = 1e6
        s = QuadStreamer(engine, CELL, (0, 0), 3, full(3), build, lambda L: CELL * 2**L / 16, cache=cache)
        _settle(s, 800.0, 800.0, 50_000.0)
        assert s.drawn() == {(3, 0, 0)}
        _settle(s, 100.0, 100.0, 1.0)
        assert (3, 0, 0) not in s.drawn()
        _settle(s, 800.0, 800.0, 50_000.0)
        assert s.drawn() == {(3, 0, 0)} and counts[(3, 0, 0)] == root_builds
        s.close()


def test_with_a_device_every_picked_node_is_uploaded_and_drawn():
    import pytest

    try:
        from manifoldx.backends import get_offscreen_canvas

        canvas = get_offscreen_canvas(width=32, height=32)
    except Exception as e:
        pytest.skip(f"offscreen canvas unavailable: {e}")
    engine = mx.Engine("quad", width=32, height=32)
    engine._init_canvas(canvas)
    engine._running = True
    engine.camera.far = 15000.0
    s = QuadStreamer(engine, CELL, (0, 0), 4, full(4), tri, lambda L: CELL * 2**L / 16)
    for _ in range(600):
        s.focus(100.0, 100.0, 1.0)
        engine._draw_frame()
        if s._quiet:
            break
        time.sleep(0.002)
    assert s._quiet and s.drawn() == set(s._selected)
    assert all(s._gpu[n]["done"] for n in s.drawn())
    s.close()


def test_settled_only_when_every_picked_node_is_drawn():
    s = faked(3)
    s.in_flight = 1000
    s.focus(100.0, 100.0, 1.0)
    assert not s.settled  # builds in flight
    for _ in range(20):
        s._pool.finish()
        s.focus(100.0, 100.0, 1.0)
    assert s.settled and s.drawn() == set(s._selected)
    s.focus(100.0, 100.0, 1e4)  # zoomed out: the root is wanted, not built yet
    assert not s.settled
    s.close()


def test_nodes_report_their_life():
    from manifoldx import counters

    counters.start(None)
    try:
        s = faked(3)
        for _ in range(20):
            s.focus(100.0, 100.0, 1.0)
            s._pool.finish()
        evs = {e["ev"] for e in counters.events() if e["kind"] == "node"}
        assert {"submitted", "built", "landed", "drawn"} <= evs
        for _ in range(20):
            s.focus(5000.0, 5000.0, 1.0)
            s._pool.finish()
        nodes = [e for e in counters.events() if e["kind"] == "node"]
        assert "undrawn" in {e["ev"] for e in nodes}
        assert all(e["level"] == e["key"][0] and e["streamer"] == s._serial for e in nodes)
        s.close()
    finally:
        counters.stop()


def test_live_lists_open_streamers_only():
    from manifoldx import streaming

    s = faked(2)
    assert s in streaming.live()
    s.close()
    assert s not in streaming.live()
