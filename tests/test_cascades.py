"""Cascaded sun shadows: the split and fit math, and what the cascades change on screen.

The render tests gate on the offscreen wgpu backend like tests/test_shadow_render.py.
"""

import numpy as np
import pytest

from manifoldx.components import Material, Mesh, Transform
from manifoldx.resources import DirectionalLight, StandardMaterial, cube, plane
from manifoldx.shadow import cascade_splits, fit_cascade


def test_splits_grow_and_end_at_the_shadow_distance():
    s = cascade_splits(0.1, 300.0, 3)
    assert len(s) == 3 and np.all(np.diff(s) > 0) and s[-1] == pytest.approx(300.0)
    assert s[0] < 300.0 / 3  # nearer than a uniform split: the near cascade is the sharp one


def _view(position, target):
    from manifoldx.camera import Camera

    return Camera(position=position, target=target).get_view_matrix()


def test_a_cascade_box_holds_its_whole_view_slice():
    view = _view((0, 2, 0), (0, 1, -10))
    inv = np.linalg.inv(view)
    t, aspect, d0, d1 = np.tan(np.radians(30)), 16 / 9, 5.0, 40.0
    lvp, _, _ = fit_cascade(inv, t, aspect, d0, d1, (-0.4, -1.0, -0.3), 2048, 100.0)
    corners = np.array([[sx * t * aspect * d, sy * t * d, -d, 1.0]
                        for d in (d0, d1) for sx in (-1, 1) for sy in (-1, 1)])
    clip = (lvp.astype(np.float64) @ (inv @ corners.T)).T
    assert np.all(np.abs(clip[:, :2]) <= 1.0) and np.all((clip[:, 2] >= 0) & (clip[:, 2] <= 1))


def test_a_cascade_box_keeps_its_size_when_the_camera_turns():
    inv_a = np.linalg.inv(_view((0, 2, 0), (0, 2, -10)))
    inv_b = np.linalg.inv(_view((0, 2, 0), (7, 0, 3)))
    a = fit_cascade(inv_a, 0.5, 1.5, 1.0, 30.0, (0.2, -1, 0.1), 1024, 50.0)
    b = fit_cascade(inv_b, 0.5, 1.5, 1.0, 30.0, (0.2, -1, 0.1), 1024, 50.0)
    assert a[1] == pytest.approx(b[1]) and a[0][0, 0] == pytest.approx(b[0][0, 0])


def test_moving_the_camera_moves_the_box_by_whole_texels():
    """Sub-texel camera steps move a fixed world point by a whole number of texels in the map, or not at all,
    so shadow edges do not crawl."""
    res, frac = 1024, []
    for dx in np.linspace(0.0, 0.2, 9):
        inv = np.linalg.inv(_view((dx, 2, 0), (dx, 0, -10)))
        lvp, _, _ = fit_cascade(inv, 0.5, 1.5, 1.0, 30.0, (0.3, -1, 0.2), res, 50.0)
        p = lvp.astype(np.float64) @ np.array([3.0, 0.0, -8.0, 1.0])
        frac.append((p[:2] * 0.5 + 0.5) * res % 1.0)
    frac = np.array(frac)
    d = np.abs(frac - frac[0])
    assert np.minimum(d, 1.0 - d).max() < 1e-3


def test_enable_shadows_takes_one_to_four_cascades():
    import manifoldx as mx

    eng = mx.Engine("cascades-args", width=64, height=64)
    eng.enable_shadows(cascades=3, shadow_distance=200.0)
    assert eng._shadow_config["cascades"] == 3 and eng._shadow_config["shadow_distance"] == 200.0
    for bad in (0, 5):
        with pytest.raises(ValueError):
            eng.enable_shadows(cascades=bad)


# --------------------------------------------------------------------------
# On screen
# --------------------------------------------------------------------------


def _engine(w=192, h=192):
    try:
        from manifoldx.backends import get_offscreen_canvas

        canvas = get_offscreen_canvas(width=w, height=h)
    except Exception as e:  # pragma: no cover
        pytest.skip(f"offscreen canvas unavailable: {e}")
    import manifoldx as mx

    eng = mx.Engine("cascades", width=w, height=h)
    eng._init_canvas(canvas)
    eng._running = True
    return eng


_FLOOR = Transform.rotation(x=-np.pi / 2, y=0, z=0)


def _kilometre(cascades, camera, post=(0.0, -10.0)):
    """A 1 km floor, a 1 m post 5 m tall at `post` and a far box that makes the scene 1 km wide, the sun low
    and to the side; the camera at `camera` looking at the post's foot."""
    eng = _engine()
    eng.set_sun(DirectionalLight(color="#ffffff", intensity=3.0, direction=(-0.6, -0.7, 0.2)))
    if cascades == 1:  # one map over a fixed box 150 m in half side around the camera, as games set it
        eng.enable_shadows(target=(camera[0], 0.0, camera[2]), extent=150.0, near=0.01, far=600.0, bias=0.002,
                           resolution=1024, pcf_radius=1, auto_fit=False)
    else:
        eng.enable_shadows(resolution=1024, pcf_radius=1, cascades=cascades, shadow_distance=300.0)
    eng.spawn(Mesh(plane(1000, 1000)), Material(StandardMaterial(color="#ffffff", roughness=1.0)),
              Transform(pos=(0, 0, 0), rot=_FLOOR))
    white = Material(StandardMaterial(color="#ffffff", roughness=1.0))
    eng.spawn(Mesh(cube(1, 5, 1)), white, Transform(pos=(post[0], 2.5, post[1])))
    eng.spawn(Mesh(cube(2, 2, 2)), white, Transform(pos=(480, 1, 480)))
    eng.camera.set_pose(position=camera, target=(post[0], 0.0, post[1]))
    return eng


def _frame(eng):
    eng._draw_frame()
    return eng._render_canvas.draw()[..., :3].mean(axis=2).astype(np.float64)


def _shadowed(eng):
    """The frame and how much each pixel is darker than with shadows off."""
    shadowed = _frame(eng)
    cfg, eng._shadow_config = eng._shadow_config, None
    lit = _frame(eng)
    eng._shadow_config = cfg
    return lit - shadowed


def test_cascades_sharpen_a_shadow_near_the_camera():
    """The post's shadow 10 m away: with 3 cascades its blurred edge is under half as wide as with one map
    over a box 300 m across."""
    def penumbra(cascades):
        dark = _shadowed(_kilometre(cascades, camera=(0.0, 6.0, 0.0)))
        full = dark.max()
        assert full > 20, f"no shadow with cascades={cascades}"
        return int(((dark > 0.15 * full) & (dark < 0.85 * full)).sum()), int((dark >= 0.85 * full).sum())

    single, single_core = penumbra(1)
    cascaded, cascaded_core = penumbra(3)
    assert cascaded_core > 0
    assert cascaded < 0.5 * single


def test_cascades_reach_a_caster_250_m_away():
    eng = _kilometre(3, camera=(0.0, 30.0, 0.0), post=(0.0, -250.0))
    eng.spawn(Mesh(cube(12, 30, 12)), Material(StandardMaterial(color="#ffffff", roughness=1.0)),
              Transform(pos=(0.0, 15.0, -250.0)))
    assert _shadowed(eng).max() > 20


def test_one_cascade_draws_as_before():
    """cascades=1 leaves the Cascades uniform at count 0: the shader takes the one-map path."""
    eng = _kilometre(1, camera=(0.0, 6.0, 0.0))
    _frame(eng)
    count = np.frombuffer(eng._render_pipeline._cascades_bytes, np.uint32)[76]
    assert count == 0 and eng._render_pipeline._shadow_map.size[2] == 1


def test_cascades_never_fit_the_whole_scene():
    """With cascades the auto-fit over every mesh's bounds, costly in a city, never runs."""
    eng = _kilometre(3, camera=(0.0, 6.0, 0.0))

    def boom(engine):
        raise AssertionError("scene bounds computed")

    eng._render_pipeline._scene_bounds = boom
    assert _shadowed(eng).max() > 20
