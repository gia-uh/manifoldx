"""Palette: per-instance colours by part for a paletted StandardMaterial's mesh."""

import numpy as np
import pytest

from manifoldx.components import Fade, Material, Mesh, Palette, Transform
from manifoldx.resources import PALETTE_SURFACE, StandardMaterial
from tests.test_tint import _halves, _offscreen, _quad


def _slotted(slot):
    q = _quad()
    q["surfaces"] = np.full(4, PALETTE_SURFACE + slot, np.float32)
    return q


def _paletted(**kw):
    return StandardMaterial("#ffffff", roughness=1.0, vertex_colors=True, paletted=True, **kw)


def test_each_instance_wears_its_palette_in_one_batch():
    eng, canvas = _offscreen()
    h = eng.spawn(Mesh(_slotted(2)), Material(_paletted()), Transform(pos=[(-1.2, 0, 0), (1.2, 0, 0)]),
                  Palette(c2=[(1, 0, 0), (0, 0, 1)], c0=(0, 1, 0)), n=2)
    assert len(h.indices) == 2
    eng._draw_frame()
    left, right = _halves(np.asarray(canvas.draw()).astype(int))
    assert (left[:, 0] > left[:, 2] + 20).any() and (right[:, 2] > right[:, 0] + 20).any()
    assert not (left[:, 1] > left[:, 0] + 20).any()  # slot 0's green is not slot 2's
    assert eng._render_pipeline.drawn == 2


def test_vertices_outside_the_slots_and_entities_without_a_palette_keep_their_colours():
    eng, canvas = _offscreen()
    eng.spawn(Mesh(_quad()), Material(_paletted()), Transform(pos=(-1.2, 0, 0)), Palette(c0=(0, 1, 0)))
    eng.spawn(Mesh(_slotted(0)), Material(_paletted()), Transform(pos=(1.2, 0, 0)))
    eng._draw_frame()
    left, right = _halves(np.asarray(canvas.draw()).astype(int))
    for side in (left, right):  # both white quads: lit grey, never green
        lit = side[side.sum(axis=1) > 20]
        assert len(lit) and not (lit[:, 1] > lit[:, 0] + 20).any()


def test_the_palette_is_srgb_and_scales_the_vertex_colour():
    eng, canvas = _offscreen()
    q = _slotted(1)
    q["colors"][:] = 0.5  # a shade: half the slot's colour, in sRGB
    eng.spawn(Mesh(q), Material(_paletted()), Transform(pos=(-1.2, 0, 0)), Palette(c1=(1, 0, 0)))
    eng.spawn(Mesh(_quad()), Material(StandardMaterial("#800000", roughness=1.0)), Transform(pos=(1.2, 0, 0)))
    eng._draw_frame()
    left, right = _halves(np.asarray(canvas.draw()).astype(int))
    assert abs(int(np.median(left[:, 0])) - int(np.median(right[:, 0]))) <= 6


def test_paletted_needs_vertex_colours_and_no_tint():
    with pytest.raises(ValueError):
        StandardMaterial("#ffffff", paletted=True)
    with pytest.raises(ValueError):
        StandardMaterial("#ffffff", vertex_colors=True, tinted=True, paletted=True)


def test_paletted_and_faded_share_one_batch():
    eng, canvas = _offscreen()
    mat = _paletted(faded=True)
    assert mat.pipeline_subtype == "vcolor-palette-fade"
    eng.spawn(Mesh(_slotted(1)), Material(mat), Transform(pos=[(-1.2, 0, 0), (1.2, 0, 0)]),
              Palette(c1=[(1, 0, 0), (0, 0, 1)]), Fade(value=[1.0, 0.0]), n=2)
    eng._draw_frame()
    left, right = _halves(np.asarray(canvas.draw()).astype(int))
    assert (left[:, 0] > left[:, 2] + 20).any() and len(right) == 0  # the faded-out one draws nothing


def test_a_changed_palette_shows_through_the_recorded_bundle():
    eng, canvas = _offscreen()
    h = eng.spawn(Mesh(_slotted(3)), Material(_paletted()), Transform(pos=(1.2, 0, 0)), Palette(c3=(1, 0, 0)))
    eng._draw_frame()
    canvas.draw()
    chunks = [c.bundle for c in eng._render_pipeline._mesh_draw.chunks]
    eng.store._components["Palette"][h.index, 9:12] = (0, 0, 1)
    eng._draw_frame()
    _, right = _halves(np.asarray(canvas.draw()).astype(int))
    assert (right[:, 2] > right[:, 0] + 20).any()
    assert [c.bundle for c in eng._render_pipeline._mesh_draw.chunks] == chunks


def _figure_frame(paletted: bool, palette=None):
    from manifoldx.humanoid.cycles import bake_cycle
    from manifoldx.modeling.assets.human import HumanParams, build_human
    from manifoldx.modeling.assets.human_dress import Outfit

    fig = build_human(HumanParams(), Outfit(top="tshirt", top_color="#c03030", bottom="trousers",
                                            bottom_color="#2040a0", hair="short", hat="cap"), level=1)
    eng, canvas = _offscreen(64, 96)
    eng.camera.set_pose(position=(0, 0.9, 3.2), target=(0, 0.9, 0))
    mat = StandardMaterial("#ffffff", roughness=1.0, vertex_colors=True, faded=True, paletted=paletted)
    geo = fig.paletted_geometry() if paletted else fig.mesh.to_geometry()
    extra = [Palette(**{f"c{k}": c for k, c in enumerate(palette if palette is not None else fig.palette())})]
    h = eng.spawn(Mesh(geo), Material(mat), Transform(pos=(0, 0, 0)), Fade(value=1.0), *(extra if paletted else []))
    clip = eng.bake_poses(bake_cycle(fig, "walk", 1.3, 8))
    eng.set_pose_frames([h.index], clip, [2.5])
    eng._draw_frame()
    return np.asarray(canvas.draw()).astype(int)[..., :3], fig


def test_a_skinned_figure_playing_a_clip_in_its_own_palette_looks_as_its_vertex_colours():
    want, fig = _figure_frame(False)
    got, _ = _figure_frame(True)
    assert np.abs(want - got).mean() < 1.0
    shirt = fig.palette().copy()
    shirt[1] = (0.1, 0.8, 0.1)
    green, _ = _figure_frame(True, shirt)
    assert int((green[..., 1] > green[..., 0] + 8).sum()) > 20  # a green shirt now (the scene is unlit, dark)
    assert int((want[..., 1] > want[..., 0] + 8).sum()) == 0
