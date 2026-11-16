"""The main pass keeps each batch's resolved state between frames and sets no state twice in a row."""
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
    e.camera.set_pose(position=(0, 0, 30), target=(0, 0, 0))
    return e


def _unique_cubes(e, n, mats, size=0.4):
    from manifoldx.components import Material, Mesh, Transform
    from manifoldx.resources import cube

    rng = np.random.default_rng(0)
    out = []
    for k, p in enumerate(rng.uniform(-8, 8, (n, 3)) * (1, 1, 0)):
        out.append(e.spawn(Mesh(cube(size, size, size)), Material(mats[k % len(mats)]), Transform(pos=tuple(p))))
    return out


def _count(cls, name, calls):
    real = getattr(cls, name)

    def spy(*a, **k):
        calls[name] = calls.get(name, 0) + 1
        return real(*a, **k)

    return real, spy


def test_still_scene_resolves_no_pipeline(monkeypatch):
    from manifoldx.renderer import RenderPipeline
    from manifoldx.resources import StandardMaterial

    e = _engine()
    _unique_cubes(e, 200, [StandardMaterial("#88aaff")])
    for _ in range(2):
        e.render_frame(supersample=1)
    calls = {}
    real, spy = _count(RenderPipeline, "_get_or_create_pipeline", calls)
    monkeypatch.setattr(RenderPipeline, "_get_or_create_pipeline", spy)
    for _ in range(3):
        e.render_frame(supersample=1)
    assert calls.get("_get_or_create_pipeline", 0) == 0


def test_one_material_sets_its_state_once(monkeypatch):
    import wgpu.backends.wgpu_native as native
    from manifoldx.resources import StandardMaterial

    e = _engine()
    _unique_cubes(e, 200, [StandardMaterial("#88aaff")])
    for _ in range(2):
        e.render_frame(supersample=1)
    calls = {}
    for name in ("set_pipeline", "draw_indexed"):
        real, spy = _count(native.GPURenderPassEncoder, name, calls)
        monkeypatch.setattr(native.GPURenderPassEncoder, name, spy)
    e.render_frame(supersample=1)
    assert calls["draw_indexed"] >= 200
    assert calls["set_pipeline"] <= 5  # skybox, sprites and the one mesh pipeline at most


def test_colour_change_shows():
    from manifoldx.resources import BasicMaterial

    e = _engine()
    mat = BasicMaterial("#ff0000")
    _unique_cubes(e, 1, [mat], size=20)
    for _ in range(3):
        before = e.render_frame(supersample=1)
    mat.color = "#0000ff"
    after = e.render_frame(supersample=1)
    assert before[24, 32, 0] > 200 and after[24, 32, 2] > 200 and after[24, 32, 0] < 50


def test_replaced_geometry_shows():
    from manifoldx.resources import BasicMaterial, cube

    e = _engine()
    (h,) = _unique_cubes(e, 1, [BasicMaterial("#ff0000")], size=1)
    e.store._components["Transform"][h.index, :3] = 0
    for _ in range(3):
        before = e.render_frame(supersample=1)
    h.set_geometry(cube(20, 20, 20))
    after = e.render_frame(supersample=1)
    assert (before.sum(axis=2) > 0).sum() < (after.sum(axis=2) > 0).sum()


def test_two_materials_render_as_before():
    from manifoldx.resources import StandardMaterial

    e = _engine()
    _unique_cubes(e, 300, [StandardMaterial("#88aaff"), StandardMaterial("#ffaa44")])
    frames = [e.render_frame(supersample=1) for _ in range(3)]
    assert np.array_equal(frames[1], frames[2]) and frames[2].sum() > 0
