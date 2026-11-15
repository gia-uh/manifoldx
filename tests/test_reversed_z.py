"""Reversed-Z float depth: far geometry keeps its order (near maps to 1, far to 0)."""
import numpy as np
import pytest

import manifoldx as mx
from manifoldx.components import Material, Mesh, Transform
from manifoldx.resources import BasicMaterial, cube


def _offscreen():
    try:
        from manifoldx.backends import get_offscreen_canvas
        get_offscreen_canvas(width=8, height=8)
    except Exception as e:
        pytest.skip(f"offscreen canvas unavailable: {e}")


def test_projection_maps_near_to_one_and_far_to_zero():
    cam = mx.Engine("rz", width=8, height=8).camera
    p = cam.get_projection_matrix(1.0, near=0.1, far=20000.0).astype(np.float64)
    for z, want in ((-0.1, 1.0), (-20000.0, 0.0)):
        clip = p @ np.array([0.0, 0.0, z, 1.0])
        assert abs(clip[2] / clip[3] - want) < 1e-5


def test_far_quads_keep_their_order():
    # With a 24-bit depth buffer cleared to 1, two surfaces at 8 km land on the
    # same depth value unless they are tens of metres apart.
    _offscreen()
    e = mx.Engine("rz", width=64, height=64)
    e.camera.near, e.camera.far = 0.1, 20000.0
    # Green is half a metre behind red and spawned first, so a tie keeps green.
    e.spawn(Mesh(cube(4000, 4000, 1)), Material(BasicMaterial("#00ff00")), Transform(pos=(0, 0, -8001.0)))
    e.spawn(Mesh(cube(4000, 4000, 1)), Material(BasicMaterial("#ff0000")), Transform(pos=(0, 0, -8000.5)))
    e.camera.set_pose(position=(0, 0, 0), target=(0, 0, -1))
    img = e.render_frame(supersample=1).astype(int)
    r, g, _ = img[32, 32]
    assert r > 200 and g < 50, img[32, 32]
