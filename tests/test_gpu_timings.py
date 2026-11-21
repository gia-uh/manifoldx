"""GPU time per pass, from timestamp queries: per pass in milliseconds, or None where the adapter has none."""

import numpy as np

from manifoldx.components import Material, Mesh, Transform
from manifoldx.resources import DirectionalLight, StandardMaterial, plane, sphere
from tests.test_shadow_render import _FLOOR_ROT, _make_offscreen_engine, _render_array


def _lit_scene(name):
    eng = _make_offscreen_engine(name, w=96, h=96)
    eng.set_sun(DirectionalLight(color="#ffffff", intensity=3.0, direction=(-0.3, -1.0, -0.3)))
    eng.enable_shadows(target=(0, 0, 0), extent=6.0, resolution=512)
    eng.spawn(Mesh(plane(12, 12)), Material(StandardMaterial(color="#ffffff")), Transform(pos=(0, 0, 0), rot=_FLOOR_ROT))
    eng.spawn(Mesh(sphere(1.0, 16)), Material(StandardMaterial(color="#ffffff")), Transform(pos=(0, 2, 0)))
    eng.camera.fit(radius=6.0, center=(0, 0, 0), azimuth=25, elevation=55)
    return eng


def test_gpu_timings_per_pass():
    eng = _lit_scene("gpu-t")
    for _ in range(12):  # one frame in eight is measured, read two frames later
        _render_array(eng)
    t = eng.gpu_timings()
    if "timestamp-query" not in eng._adapter.features:
        assert t is None
        return
    assert set(t) >= {"shadow", "main", "total"} and "views" not in t  # no second camera this frame
    assert all(v >= 0 for v in t.values()) and t["total"] >= t["main"]
    assert t["total"] < 1000.0  # milliseconds, not raw ticks


def test_gpu_timings_off(monkeypatch):
    monkeypatch.setenv("MANIFOLDX_GPU_TIMINGS", "0")
    eng = _lit_scene("gpu-off")
    for _ in range(4):
        _render_array(eng)
    assert eng.gpu_timings() is None


def test_gpu_timings_count_second_cameras_as_views():
    import manifoldx as mx

    eng = _lit_scene("gpu-views")
    if "timestamp-query" not in eng._adapter.features:
        return
    cam = mx.Camera() if hasattr(mx, "Camera") else None
    from manifoldx.camera import Camera

    cam = Camera()
    cam.fit(radius=6.0, center=(0, 0, 0), azimuth=60, elevation=40)
    eng.render_to_texture(cam, (32, 32), live=True)
    for _ in range(12):  # one frame in eight is measured, read two frames later
        _render_array(eng)
    t = eng.gpu_timings()
    assert "views" in t and t["views"] >= 0 and t["total"] >= t["main"]
