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
        e.render_frame(pass_="depth")


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


def test_frame_is_contiguous_rgb_uint8():
    e = _engine()
    img = e.render_frame(supersample=1)
    assert img.shape == (48, 64, 3) and img.dtype == np.uint8 and img.flags.c_contiguous


def test_frame_equals_the_canvas_without_alpha():
    e = _engine()
    from manifoldx.components import Material, Mesh, Transform
    from manifoldx.resources import StandardMaterial, cube
    e.spawn(Mesh(cube(1, 1, 1)), Material(StandardMaterial(color="#3366cc")), Transform(pos=(0, 0, 0)))
    e.camera.set_pose(position=(2, 2, 3), target=(0, 0, 0))
    img = e.render_frame(supersample=1)
    raw = np.asarray(e._render_canvas.draw())
    assert np.array_equal(img, raw[:, :, :3])


@pytest.mark.parametrize("shape", [(1, 1), (1, 7), (48, 64), (37, 101), (1080, 1920)])
def test_drop_alpha_matches_a_slice(shape):
    from manifoldx.engine import _drop_alpha

    rgba = np.random.default_rng(0).integers(0, 256, (*shape, 4), dtype=np.uint8)
    out = _drop_alpha(rgba)
    assert out.shape == (*shape, 3) and out.flags.c_contiguous and np.array_equal(out, rgba[:, :, :3])


def test_drop_alpha_falls_back_to_a_slice(monkeypatch):
    import manifoldx.engine as eng

    monkeypatch.setattr(eng, "_OVERLAP_OK", False)  # as on a numpy that writes the overlap out of order
    rgba = np.random.default_rng(1).integers(0, 256, (37, 101, 4), dtype=np.uint8)
    assert np.array_equal(eng._drop_alpha(rgba), rgba[:, :, :3])
