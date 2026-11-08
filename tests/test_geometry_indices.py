"""Indices given as (M, 3) triangles draw every triangle, as a flat (3M,) array does."""
import numpy as np
import pytest

import manifoldx as mx
from manifoldx.components import Material, Mesh, Transform
from manifoldx.resources import FlatMaterial, plane


def _engine():
    try:
        from manifoldx.backends import get_offscreen_canvas
        get_offscreen_canvas(width=8, height=8)
    except Exception as e:
        pytest.skip(f"offscreen canvas unavailable: {e}")
    e = mx.Engine("indices", width=32, height=32)
    e.camera.set_pose(position=(0, 0, 3), target=(0, 0, 0))
    return e


@pytest.mark.parametrize("shape", [(-1,), (-1, 3)])
def test_triangle_indices_draw_every_triangle(shape):
    quad = plane(10, 10)
    geo = {**quad, "indices": np.asarray(quad["indices"]).reshape(shape)}
    e = _engine()
    e.spawn(Mesh(geo), Material(FlatMaterial("#cc2222")), Transform(pos=(0, 0, 0)))
    frame = e.render_frame(supersample=1)
    # The quad fills the view: both of its triangles must cover their corners.
    for y, x in [(2, 2), (2, 29), (29, 2), (29, 29)]:
        assert frame[y, x, 0] > 150 and frame[y, x, 1] < 80, (y, x, frame[y, x])
