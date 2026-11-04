"""Engine.render_frame: one still frame, headless."""
import numpy as np
import pytest

import manifoldx as mx


def _engine(w=64, h=48):
    try:
        from manifoldx.backends import get_offscreen_canvas
        get_offscreen_canvas(width=8, height=8)
    except Exception as e:
        pytest.skip(f"offscreen canvas unavailable: {e}")
    return mx.Engine("test", width=w, height=h)


def test_returns_rgb_at_engine_size():
    e = _engine()
    img = e.render_frame(supersample=2)
    assert img.shape == (48, 64, 3) and img.dtype == np.uint8


def test_empty_scene_is_the_background_colour():
    e = _engine()
    e.background_color = (1.0, 0.0, 0.0)
    img = e.render_frame(supersample=1)
    assert tuple(img[24, 32]) == (255, 0, 0)


def test_consecutive_frames_are_identical():
    e = _engine()
    from manifoldx.components import Material, Mesh, Transform
    from manifoldx.resources import StandardMaterial, cube
    e.spawn(Mesh(cube(1, 1, 1)), Material(StandardMaterial(color="#3366cc")), Transform(pos=(0, 0, 0)))
    e.camera.set_pose(position=(2, 2, 3), target=(0, 0, 0))
    assert np.array_equal(e.render_frame(supersample=1), e.render_frame(supersample=1))


def test_supersample_is_fixed_per_engine():
    e = _engine()
    e.render_frame(supersample=2)
    with pytest.raises(ValueError, match="supersample"):
        e.render_frame(supersample=1)


def test_unknown_pass_is_rejected():
    e = _engine()
    with pytest.raises(ValueError, match="pass_"):
        e.render_frame(pass_="albedo")


def test_animated_scene_is_reproducible_whatever_the_wall_clock():
    # Review F3: render_frame simulated with wall-clock dt, so the same scene
    # rendered with pauses between calls drifted.
    import time
    from manifoldx.components import Material, Mesh, Transform
    from manifoldx.resources import FlatMaterial, cube

    shots = []
    for pause in (0.0, 0.2):
        e = _engine()
        e.spawn(Mesh(cube(0.4, 0.4, 0.4)), Material(FlatMaterial("#ffffff")), Transform(pos=(-1.0, 0, 0)))
        e.camera.set_pose(position=(0, 0, 4), target=(0, 0, 0))

        @e.system
        def slide(query: mx.Query[Transform], dt: float):
            pos = query[Transform].pos.data.copy()
            pos[:, 0] += 2.0 * dt
            query[Transform].pos = pos

        for _ in range(3):
            time.sleep(pause)
            img = e.render_frame(supersample=1)
        shots.append(img)
    assert np.array_equal(shots[0], shots[1])


def _cube_scene(far=50.0):
    from manifoldx.components import Material, Mesh, Transform
    from manifoldx.resources import DirectionalLight, StandardMaterial, cube

    e = _engine()
    e.spawn(Mesh(cube(1, 1, 1)), Material(StandardMaterial(color="#3366cc")), Transform(pos=(0, 0, 0)))
    e.set_sun(DirectionalLight(color="#ffffff", intensity=3.0, direction=(-0.3, -1.0, -0.5)))
    e.camera.far = far
    e.camera.set_pose(position=(0, 0, 5), target=(0, 0, 0))
    return e


def test_normals_pass_faces_the_camera_on_the_front_face():
    e = _cube_scene()
    n = e.render_frame(supersample=1, pass_="normals")
    assert n.shape == (48, 64, 3) and n.dtype == np.float32
    assert np.allclose(n[24, 32], (0, 0, 1), atol=0.02)
    assert np.allclose(n[0, 0], 0)


def test_depth_pass_measures_view_distance():
    e = _cube_scene(far=50.0)
    d = e.render_frame(supersample=1, pass_="depth")
    assert d.shape == (48, 64) and d.dtype == np.float32
    assert abs(d[24, 32] - 4.5) < 0.01  # camera at z=5, front face at z=0.5
    assert np.isinf(d[0, 0])


def test_light_pass_is_shading_without_colour():
    e = _cube_scene()
    light = e.render_frame(supersample=1, pass_="light")
    beauty = e.render_frame(supersample=1)
    assert light.shape == (48, 64) and 0 < light[24, 32] <= 1
    assert light[0, 0] == 0
    r, g, b = beauty[24, 32]
    assert b > r  # the beauty frame keeps the blue material; the light pass is grey


def test_passes_do_not_change_the_beauty_frame():
    e = _cube_scene()
    before = e.render_frame(supersample=1)
    for pass_ in ("normals", "depth", "light"):
        e.render_frame(supersample=1, pass_=pass_)
    assert np.array_equal(before, e.render_frame(supersample=1))


def test_passes_downsample_with_supersampling():
    e = _cube_scene()
    e.render_frame(supersample=2)
    assert e.render_frame(supersample=2, pass_="normals").shape == (48, 64, 3)
    assert e.render_frame(supersample=2, pass_="depth").shape == (48, 64)
    assert e.render_frame(supersample=2, pass_="light").shape == (48, 64)
