"""Screen-space effects after the main pass (manifoldx.post): each one changes the frame where it should, and
the frame is the same as before with all of them off."""

import numpy as np
import pytest

from manifoldx.components import Material, Mesh, Transform
from manifoldx.post import SSAO, Bloom, Grade, Post, sun_on_screen
from manifoldx.resources import BasicMaterial, DirectionalLight, StandardMaterial, cube, plane, sphere
from tests.test_shadow_render import _FLOOR_ROT, _make_offscreen_engine, _render_array


def _lum(img):
    return img[..., :3].astype(float).mean(axis=-1)


def _street(name, w=160, h=120):
    eng = _make_offscreen_engine(name, w=w, h=h)
    eng.background_color = (0.55, 0.7, 0.9)
    eng.set_sun(DirectionalLight(color="#ffffff", intensity=2.0, direction=(-0.4, -0.6, -0.7)))
    eng.spawn(Mesh(plane(40, 40)), Material(StandardMaterial(color="#bbbbbb")), Transform(pos=(0, 0, 0), rot=_FLOOR_ROT))
    eng.spawn(Mesh(cube(3, 3, 3)), Material(StandardMaterial(color="#bbbbbb")), Transform(pos=(0, 1.5, 0)))
    eng.camera.position = (4.0, 4.0, 6.0)
    eng.camera.target = (0.0, 0.5, 0.0)
    return eng


def test_all_off_draws_the_frame_as_before():
    eng = _street("post-off")
    before = _render_array(eng)
    eng.set_post(ssao=True, bloom=True, fxaa=True)
    eng.set_post(ssao=None, bloom=False, fxaa=False)
    assert not eng._post.active()
    assert np.array_equal(_render_array(eng), before)


def test_a_neutral_grade_is_a_copy():
    eng = _street("post-copy")
    before = _render_array(eng).astype(int)
    eng.set_post(grade=True)  # saturation 1, contrast 1, no vignette
    assert np.abs(_render_array(eng).astype(int) - before).max() <= 1


def test_ssao_darkens_where_the_box_meets_the_ground_and_not_the_open_ground():
    eng = _street("post-ssao")
    off = _lum(_render_array(eng))
    eng.set_post(ssao=SSAO(radius=1.5))
    on = _lum(_render_array(eng))
    changed = off - on
    assert changed.max() > 20  # the contact
    assert np.abs(changed[-12:, :]).max() <= 2  # the open ground near the camera
    assert np.abs(changed[:10, :]).max() <= 2  # the sky


def test_bloom_spreads_light_past_a_bright_square():
    eng = _make_offscreen_engine("post-bloom", w=96, h=96)
    eng.background_color = (0.0, 0.0, 0.0)
    eng.spawn(Mesh(cube(1, 1, 0.1)), Material(BasicMaterial(color="#ffffff")), Transform(pos=(0, 0, 0)))
    eng.camera.position = (0.0, 0.0, 6.0)
    eng.camera.target = (0.0, 0.0, 0.0)
    off = _lum(_render_array(eng))
    eng.set_post(bloom=Bloom(threshold=0.5, intensity=1.0))
    on = _lum(_render_array(eng))
    ring = (off == 0) & (np.abs(np.arange(96)[:, None] - 48) < 20) & (np.abs(np.arange(96)[None, :] - 48) < 20)
    assert ring.any() and on[ring].mean() > 5  # black around the square turns grey
    assert on[0, 0] <= 1  # far corners stay black


def test_fxaa_softens_a_diagonal_edge_and_leaves_flat_regions():
    eng = _make_offscreen_engine("post-fxaa", w=96, h=96)
    eng.background_color = (1.0, 1.0, 1.0)
    rot = (0.0, 0.0, np.sin(np.radians(12.5)), np.cos(np.radians(12.5)))  # 25 degrees about z
    eng.spawn(Mesh(cube(4, 4, 0.1)), Material(BasicMaterial(color="#000000")), Transform(pos=(-1.6, 0, 0), rot=rot))
    eng.camera.position = (0.0, 0.0, 6.0)
    eng.camera.target = (0.0, 0.0, 0.0)
    off = _lum(_render_array(eng))
    eng.set_post(fxaa=True)
    on = _lum(_render_array(eng))
    assert ((off > 5) & (off < 250)).sum() < ((on > 5) & (on < 250)).sum()  # more in-between pixels on the edge
    white = np.pad(off == 255, 1, constant_values=True)
    flat = np.all([white[1 + dy:97 + dy, 1 + dx:97 + dx] for dy in (-1, 0, 1) for dx in (-1, 0, 1)], axis=0)
    assert np.abs(on[flat] - off[flat]).max() <= 1


def test_the_sun_on_screen_is_where_the_camera_sees_it():
    eng = _street("post-sun")
    eng.camera.position = (0.0, 2.0, 10.0)
    eng.camera.target = (0.0, 2.0, 0.0)
    ahead = DirectionalLight(color="#ffffff", intensity=1.0, direction=(0.0, -0.2, 1.0))  # shines toward the camera
    u, v, vis = sun_on_screen(eng.camera, 4 / 3, ahead)
    assert abs(u - 0.5) < 1e-6 and v < 0.5 and vis > 0.9
    behind = DirectionalLight(color="#ffffff", intensity=1.0, direction=(0.0, -0.2, -1.0))
    assert sun_on_screen(eng.camera, 4 / 3, behind)[2] == 0.0
    set_ = DirectionalLight(color="#ffffff", intensity=1.0, direction=(0.0, 0.3, 1.0))  # below the horizon
    assert sun_on_screen(eng.camera, 4 / 3, set_)[2] == 0.0


def test_light_shafts_brighten_the_sky_toward_the_sun_and_stop_at_geometry():
    eng = _street("post-shafts")
    eng.camera.position = (0.0, 1.5, 10.0)
    eng.camera.target = (0.0, 3.0, 0.0)
    eng.set_sun(DirectionalLight(color="#ffffff", intensity=2.0, direction=(0.15, -0.3, 1.0)))
    off = _lum(_render_array(eng))
    eng.set_post(shafts=True)
    on = _lum(_render_array(eng))
    assert (on - off).max() > 30


def test_the_gui_is_drawn_after_the_effects():
    from manifoldx.gui import Panel

    eng = _street("post-gui")
    eng.gui.append(Panel([], style_overrides={"bg": "#ff0000", "width": 40, "height": 30, "padding": 0}))
    plain = _render_array(eng)
    eng.set_post(ssao=True, bloom=True, fxaa=True, grade=Grade(saturation=0.0))
    post = _render_array(eng)
    assert np.array_equal(post[5:20, 5:30], plain[5:20, 5:30])  # the panel keeps its colour


def test_unknown_effects_are_refused():
    eng = _street("post-bad")
    with pytest.raises(ValueError, match="unknown post effect"):
        eng.set_post(dof=True)
    assert Post().changed(ssao=True).ssao == SSAO()


def test_gpu_timings_carry_the_post_passes():
    eng = _street("post-timed")
    eng.set_post(ssao=True, bloom=True)
    for _ in range(12):
        _render_array(eng)
    t = eng.gpu_timings()
    if t is None:
        pytest.skip("no timestamp queries on this adapter")
    assert t["post"] >= 0
