"""Facades: a per-vertex facade coordinate draws windows, doors and wear on a vertex-coloured wall."""

import numpy as np
import pytest

import manifoldx as mx
from manifoldx.components import Material, Mesh, Transform
from manifoldx.resources import DirectionalLight, StandardMaterial, material_rgb, pack_vertices


def _wall(kind, with_key=True, bays=4.0, floors=3.0):
    """A 12 x 9 m wall facing +z, with `bays` window bays across and `floors` floors up."""
    geo = {
        "positions": np.array([[-6, 0, 0], [6, 0, 0], [6, 9, 0], [-6, 9, 0]], np.float32),
        "normals": np.tile([0, 0, 1], (4, 1)).astype(np.float32),
        "colors": np.tile(material_rgb("#e7c3a1"), (4, 1)).astype(np.float32),
        "indices": np.array([0, 1, 2, 0, 2, 3], np.uint32),
    }
    if with_key:
        geo["facade"] = np.array([[0, 0, kind, 0.3], [bays, 0, kind, 0.3], [bays, floors, kind, 0.3],
                                  [0, floors, kind, 0.3]], np.float32)
    return geo


def _render(geo, eye=(0, 4.5, 14)):
    try:
        from manifoldx.backends import get_offscreen_canvas
        canvas = get_offscreen_canvas(width=128, height=128)
    except Exception as e:
        pytest.skip(f"offscreen canvas unavailable: {e}")
    eng = mx.Engine("t", width=128, height=128)
    eng._init_canvas(canvas)
    eng._running = True
    eng.set_sun(DirectionalLight(color="#ffffff", intensity=3.0, direction=(-0.3, -0.6, -1.0)))
    eng.spawn(Mesh(geo), Material(StandardMaterial(color="#ffffff", vertex_colors=True)), Transform(pos=(0, 0, 0)))
    eng.camera.set_pose(position=eye, target=(0, 4.5, 0))
    eng._draw_frame()
    return np.asarray(canvas.draw())[..., :3].astype(int)


def test_the_facade_key_adds_four_floats_to_each_vertex():
    _, plain = pack_vertices(_wall(2, with_key=False))
    _, faced = pack_vertices(_wall(2))
    assert plain["stride"] == 40 and faced["stride"] == 56 and faced["has_facade"] and not plain["has_facade"]


def test_windows_darken_a_wall_and_type_zero_stays_plain():
    plain = _render(_wall(0))[30:100, 20:108]
    faced = _render(_wall(2))[30:100, 20:108]
    assert plain.std(axis=(0, 1)).max() < 6  # one flat colour
    assert faced.std(axis=(0, 1)).max() > 25  # windows, frames and doors
    assert (faced.sum(axis=2) < plain.sum(axis=2).mean() * 0.4).mean() > 0.08  # dark glass and doors


def test_every_type_draws():
    for kind in range(1, 8):
        img = _render(_wall(kind))[30:100, 20:108]
        assert img.std(axis=(0, 1)).max() > 12, kind
