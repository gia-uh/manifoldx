"""Static entities cost no per-frame routing work, and every change still reaches the next frame."""
import numpy as np
import pytest

import manifoldx as mx


def _engine(w=64, h=48):
    try:
        from manifoldx.backends import get_offscreen_canvas
        get_offscreen_canvas(width=8, height=8)
    except Exception as e:
        pytest.skip(f"offscreen canvas unavailable: {e}")
    e = mx.Engine("test", width=w, height=h, max_entities=4096)
    e.background_color = (0.0, 0.0, 0.0)
    e.camera.set_pose(position=(0, 0, 10), target=(0, 0, 0))
    return e


def _sprites(e, pos, value=1.0, radius=0.3):
    from manifoldx.components import Material, Transform
    from manifoldx.viz import ColormapMaterial, PointCloud, Radius, ScalarValue

    n = len(pos)
    return e.spawn(PointCloud(), Material(ColormapMaterial(cmap="viridis", vmin=0, vmax=1)),
                   Transform(pos=np.asarray(pos, np.float32)), ScalarValue(value=np.full(n, value, np.float32)),
                   Radius(radius=np.full(n, radius, np.float32)), n=n)


def _cube(e, colour, at=(0, 0, 0)):
    from manifoldx.components import Material, Mesh, Transform
    from manifoldx.resources import BasicMaterial, cube

    return e.spawn(Mesh(cube(2, 2, 2)), Material(BasicMaterial(colour)), Transform(pos=at))


def test_spawn_after_frames_is_drawn():
    e = _engine()
    _sprites(e, [(-3, 0, 0)])
    for _ in range(3):
        before = e.render_frame(supersample=1)
    _sprites(e, [(0, 0, 0)])
    after = e.render_frame(supersample=1)
    assert before[24, 32].sum() == 0 and after[24, 32].sum() > 0


def test_despawn_after_frames_is_gone():
    e = _engine()
    _sprites(e, [(-3, 0, 0)])
    h = _sprites(e, [(0, 0, 0)])
    for _ in range(3):
        before = e.render_frame(supersample=1)
    h.destroy()
    after = e.render_frame(supersample=1)
    assert before[24, 32].sum() > 0 and after[24, 32].sum() == 0


def test_material_written_in_array_reroutes():
    e = _engine()
    red = _cube(e, "#ff0000")
    blue = _cube(e, "#0000ff", at=(100, 0, 0))  # out of view: only its material id is borrowed
    for _ in range(3):
        before = e.render_frame(supersample=1)
    mats = e.store._components["Material"]
    mats[red.index] = mats[blue.index]
    after = e.render_frame(supersample=1)
    assert before[24, 32, 0] > 200 and after[24, 32, 2] > 200 and after[24, 32, 0] < 50


def test_moved_sprite_matches_fresh_engine():
    rng = np.random.default_rng(0)
    still = rng.uniform(-4, 4, (1000, 3)) * (1, 1, 0)
    moving = rng.uniform(-4, 4, (10, 3)) * (1, 1, 0)
    e = _engine()
    _sprites(e, still, radius=0.05)
    first = _sprites(e, moving, radius=0.2)
    rows = np.arange(first.index, first.index + 10)
    for k in range(5):
        e.render_frame(supersample=1)
        e.store._components["Transform"][rows, 1] -= 0.3
    final = e.store._components["Transform"][rows, :3].copy()
    moved = e.render_frame(supersample=1)

    f = _engine()
    _sprites(f, still, radius=0.05)
    _sprites(f, final, radius=0.2)
    assert np.array_equal(moved, f.render_frame(supersample=1))


def test_routing_rebuilt_only_on_change():
    from manifoldx import renderer

    e = _engine()
    _sprites(e, np.zeros((50, 3)))
    _cube(e, "#ff0000")
    e.render_frame(supersample=1)
    calls = {"n": 0}
    real = renderer._groups

    def counting(*a, **k):
        calls["n"] += 1
        return real(*a, **k)

    renderer._groups = counting
    try:
        e.render_frame(supersample=1)
        e.render_frame(supersample=1)
    finally:
        renderer._groups = real
    assert calls["n"] == 0


def test_moving_few_sprites_uploads_few_rows():
    e = _engine()
    _sprites(e, np.random.default_rng(0).uniform(-4, 4, (2000, 3)), radius=0.05)
    first = _sprites(e, np.zeros((10, 3)), radius=0.2)
    for _ in range(3):
        e.render_frame(supersample=1)
    e.store._components["Transform"][first.index:first.index + 10, 1] += 0.1
    q = e._device.queue
    real, sent = q.write_buffer, {"bytes": 0}

    def counting(buf, offset, data, *a, **k):
        sent["bytes"] += memoryview(data).nbytes
        return real(buf, offset, data, *a, **k)

    q.write_buffer = counting
    try:
        e.render_frame(supersample=1)
    finally:
        q.write_buffer = real
    assert sent["bytes"] < 2010 * 64 // 4  # a full re-upload of the transforms alone is 2010 * 64
