"""Fade: a per-instance share of a faded StandardMaterial's mesh that draws, dropped against a 4x4 pattern."""

import numpy as np
import pytest

import manifoldx as mx
from manifoldx.components import Fade, Material, Mesh, Tint, Transform
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


def _mask(canvas):
    """The pixels that are not the background (the corner's colour)."""
    f = np.asarray(canvas.draw()).astype(int)[:, :, :3]
    return np.any(f != f[0, 0], axis=2)


def _faded(**kw):
    return StandardMaterial("#ffffff", roughness=1.0, faded=True, **kw)


@pytest.mark.parametrize("kw", [{}, {"vertex_colors": True}, {"vertex_colors": True, "tinted": True}])
def test_a_fade_draws_its_share_of_the_pixels(kw):
    eng, canvas = _offscreen()
    h = eng.spawn(Mesh(_quad()), Material(_faded(**kw)), Transform(), Fade(value=1.0))
    eng._draw_frame()
    whole = _mask(canvas)
    assert whole.sum() > 100
    shares = {}
    for v in (0.5, 0.25, 0.0):
        eng.store._components["Fade"][h.index] = v
        eng._draw_frame()
        m = _mask(canvas)
        assert not (m & ~whole).any()
        shares[v] = m.sum() / whole.sum()
    assert shares[0.5] == pytest.approx(0.5, abs=0.06)
    assert shares[0.25] == pytest.approx(0.25, abs=0.06)
    assert shares[0.0] == 0


def test_an_entity_without_a_fade_draws_whole():
    eng, canvas = _offscreen()
    plain = StandardMaterial("#ffffff", roughness=1.0, vertex_colors=True)
    eng.spawn(Mesh(_quad()), Material(plain), Transform())
    eng._draw_frame()
    whole = _mask(canvas)
    eng2, canvas2 = _offscreen()
    eng2.spawn(Mesh(_quad()), Material(_faded(vertex_colors=True)), Transform())
    eng2._draw_frame()
    assert np.array_equal(_mask(canvas2), whole)


def test_a_negative_fade_draws_the_complementary_pixels():
    eng, canvas = _offscreen()
    h = eng.spawn(Mesh(_quad()), Material(_faded(vertex_colors=True)), Transform(), Fade(value=1.0))
    eng._draw_frame()
    whole = _mask(canvas)
    masks = []
    for v in (0.3, -0.7):  # one level fading in at t, the other out at -(1 - t)
        eng.store._components["Fade"][h.index] = v
        eng._draw_frame()
        masks.append(_mask(canvas))
    assert not (masks[0] & masks[1]).any()
    assert np.array_equal(masks[0] | masks[1], whole)


def test_fade_and_tint_share_one_batch():
    eng, canvas = _offscreen()
    mat = _faded(vertex_colors=True, tinted=True)
    eng.spawn(Mesh(_quad()), Material(mat), Transform(pos=[(-1.2, 0, 0), (1.2, 0, 0)]),
              Tint(rgb=[(1, 0, 0), (0, 0, 1)]), Fade(value=[1.0, 0.5]), n=2)
    eng._draw_frame()
    f = np.asarray(canvas.draw()).astype(int)[:, :, :3]
    fg = np.any(f != f[0, 0], axis=2)
    left, right = f[:, :48][fg[:, :48]], f[:, 48:][fg[:, 48:]]
    assert (left[:, 0] > left[:, 2] + 20).all() and (right[:, 2] > right[:, 0] + 20).all()
    assert len(right) == pytest.approx(len(left) / 2, rel=0.15)
    assert eng._render_pipeline.drawn == 2


def test_faded_is_its_own_pipeline_and_unfaded_keeps_its_subtype():
    assert StandardMaterial("#fff").pipeline_subtype is None
    assert StandardMaterial("#fff", vertex_colors=True).pipeline_subtype == "vcolor"
    assert StandardMaterial("#fff", faded=True).pipeline_subtype == "fade"
    assert StandardMaterial("#fff", vertex_colors=True, tinted=True, faded=True).pipeline_subtype == "vcolor-tint-fade"

