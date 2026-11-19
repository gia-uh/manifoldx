"""Tint: a per-instance colour that multiplies the vertex colours of a tinted StandardMaterial's mesh."""

import numpy as np
import pytest

import manifoldx as mx
from manifoldx.components import Material, Mesh, Tint, Transform
from manifoldx.resources import StandardMaterial


def _offscreen(w=96, h=48):
    try:
        from manifoldx.backends import get_offscreen_canvas
        canvas = get_offscreen_canvas(width=w, height=h)
    except Exception as e:
        pytest.skip(f"offscreen canvas unavailable: {e}")
    eng = mx.Engine("t", width=w, height=h)
    eng._init_canvas(canvas)
    eng._running = True
    eng.camera.set_pose(position=(0, 0, 6), target=(0, 0, 0))
    return eng, canvas


def _quad():
    p = np.array([(-0.9, -0.9, 0), (0.9, -0.9, 0), (0.9, 0.9, 0), (-0.9, 0.9, 0)], np.float32)
    return {"positions": p, "normals": np.tile((0, 0, 1), (4, 1)).astype(np.float32),
            "colors": np.ones((4, 3), np.float32), "indices": np.array([0, 1, 2, 0, 2, 3], np.uint32)}


def _halves(frame):
    """The middle row's pixels left and right of the centre, without the background's."""
    row = frame[frame.shape[0] // 2, :, :3]
    fg = ~np.all(row == row[0], axis=1)
    mid = frame.shape[1] // 2
    return row[:mid][fg[:mid]], row[mid:][fg[mid:]]


def _white():
    return StandardMaterial("#ffffff", roughness=1.0, vertex_colors=True, tinted=True)


def test_each_instance_wears_its_tint_in_one_batch():
    eng, canvas = _offscreen()
    h = eng.spawn(Mesh(_quad()), Material(_white()), Transform(pos=[(-1.2, 0, 0), (1.2, 0, 0)]),
                  Tint(rgb=[(1, 0, 0), (0, 0, 1)]), n=2)
    assert len(h.indices) == 2
    eng._draw_frame()
    f = np.asarray(canvas.draw()).astype(int)
    left, right = _halves(f)
    assert (left[:, 0] > left[:, 2] + 20).any() and (right[:, 2] > right[:, 0] + 20).any()
    assert not (left[:, 2] > left[:, 0] + 20).any() and not (right[:, 0] > right[:, 2] + 20).any()
    assert eng._render_pipeline.drawn == 2


def test_an_entity_without_tint_keeps_its_colours():
    eng, canvas = _offscreen()
    eng.spawn(Mesh(_quad()), Material(_white()), Transform(pos=(-1.2, 0, 0)), Tint(rgb=(0, 1, 0)))
    eng.spawn(Mesh(_quad()), Material(_white()), Transform(pos=(1.2, 0, 0)))
    eng._draw_frame()
    f = np.asarray(canvas.draw()).astype(int)
    left, right = _halves(f)
    assert (left[:, 1] > left[:, 0] + 20).any()  # green
    lit = right[right.sum(axis=1) > 20]  # the white quad: lit, grey, never green
    assert len(lit) and not (lit[:, 1] > lit[:, 0] + 20).any()


def test_tinted_needs_vertex_colours():
    with pytest.raises(ValueError):
        StandardMaterial("#ffffff", tinted=True)


def test_a_tinted_batch_after_an_untinted_one_reads_its_own_rows():
    eng, canvas = _offscreen()
    plain = StandardMaterial("#ffffff", roughness=1.0, vertex_colors=True)
    eng.spawn(Mesh(_quad()), Material(plain), Transform(pos=[(-1.2, 0, 0), (-1.2, 0, -5)]), n=2)
    eng.spawn(Mesh(_quad()), Material(_white()), Transform(pos=(1.2, 0, 0)), Tint(rgb=(1, 0, 0)))
    eng._draw_frame()
    left, right = _halves(np.asarray(canvas.draw()).astype(int))
    assert (right[:, 0] > right[:, 1] + 20).any()  # red
    assert not (left[:, 0] > left[:, 1] + 20).any()
