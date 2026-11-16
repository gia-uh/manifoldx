"""Model matrices and shadow bounds each frame draws with, as entities move, stay,
spawn and die. GPU-gated on the offscreen backend, like tests/test_shadow_render.py."""

import numpy as np
import pytest

from manifoldx.components import Material, Mesh, Transform
from manifoldx.resources import DirectionalLight, StandardMaterial, cube


def _engine(shadows=False):
    try:
        from manifoldx.backends import get_offscreen_canvas

        canvas = get_offscreen_canvas(width=32, height=32)
    except Exception as e:  # pragma: no cover
        pytest.skip(f"offscreen canvas unavailable: {e}")
    import manifoldx as mx

    engine = mx.Engine("transform-frame", width=32, height=32)
    engine._init_canvas(canvas)
    engine._running = True
    engine.camera.fit(60)
    if shadows:
        engine.set_sun(DirectionalLight(color="#ffffff", intensity=3.0, direction=(-0.3, -1.0, -0.2)))
        engine.enable_shadows()
    return engine


def _model(pos, rot=(0, 0, 0, 1), scale=(1, 1, 1)):
    """Row-major model matrix, scale then rotate (quaternion x, y, z, w) then translate."""
    x, y, z, w = rot
    r = np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])
    m = np.eye(4)
    m[:3, :3] = r * np.asarray(scale)
    m[:3, 3] = pos
    return m


class _Frames:
    """Draws frames and returns the model matrices the mesh pass's buffer holds, row-major (N, 4, 4): its
    latest upload, since a frame whose instances did not change uploads nothing."""

    def __init__(self, engine):
        self.engine = engine
        engine._draw_frame()  # creates the batch buffers
        self.uploads = []
        buffers = engine._render_pipeline._batch_buffers
        real = buffers.upload_transforms

        def upload(mats):
            self.uploads.append(np.asarray(mats).reshape(-1, 4, 4).transpose(0, 2, 1).copy())
            return real(mats)

        buffers.upload_transforms = upload
        buffers._transforms_key = None  # forget what the buffer holds, so the first draw records it

    def draw(self):
        before = len(self.uploads)
        self.engine._draw_frame()
        assert len(self.uploads) - before <= 1 and self.uploads
        return self.uploads[-1]


def _spawn(engine, pos, rot=(0, 0, 0, 1), scale=(1, 1, 1)):
    return engine.spawn(
        Mesh(cube(1.0, 1.0, 1.0)),
        Material(StandardMaterial(color="#ffffff")),
        Transform(pos=pos, rot=rot, scale=scale),
    )


QUARTER_Z = (0.0, 0.0, np.sin(np.pi / 4), np.cos(np.pi / 4))


def test_static_entities_draw_the_same_matrices_every_frame():
    e = _engine()
    _spawn(e, (1, 2, 3), QUARTER_Z, (2, 1, 1))
    _spawn(e, (-1, 0, 0))
    f = _Frames(e)
    first = f.draw()
    np.testing.assert_allclose(first, [_model((1, 2, 3), QUARTER_Z, (2, 1, 1)), _model((-1, 0, 0))], atol=1e-6)
    np.testing.assert_array_equal(f.draw(), first)


def test_moved_entity_draws_where_it_moved():
    e = _engine()
    h = _spawn(e, (0, 0, 0))
    _spawn(e, (5, 0, 0))
    f = _Frames(e)
    f.draw()
    h.transform.pos = (0, 4, 0)
    h.transform.rot = QUARTER_Z
    h.transform.scale = (3, 3, 3)
    np.testing.assert_allclose(f.draw(), [_model((0, 4, 0), QUARTER_Z, (3, 3, 3)), _model((5, 0, 0))], atol=1e-6)


def test_entity_moved_by_a_system_draws_where_it_moved():
    import manifoldx as mx

    e = _engine()
    _spawn(e, (0, 0, 0))

    @e.system
    def lift(q: mx.Query[Transform], dt: float):
        q[Transform].position += np.float32([0, 1, 0])

    f = _Frames(e)
    y0 = f.draw()[0, 1, 3]
    assert f.draw()[0, 1, 3] == pytest.approx(y0 + 1.0)


def test_direct_store_write_between_frames_is_drawn():
    e = _engine()
    h = _spawn(e, (0, 0, 0))
    f = _Frames(e)
    f.draw()
    e.store._components["Transform"][h.index, 0:3] = (7, 8, 9)
    np.testing.assert_allclose(f.draw()[0], _model((7, 8, 9)), atol=1e-6)


def test_spawned_entity_draws_with_its_own_matrix():
    e = _engine()
    _spawn(e, (1, 0, 0))
    f = _Frames(e)
    f.draw()
    _spawn(e, (0, 0, -6), QUARTER_Z, (1, 2, 1))
    np.testing.assert_allclose(f.draw(), [_model((1, 0, 0)), _model((0, 0, -6), QUARTER_Z, (1, 2, 1))], atol=1e-6)


def test_destroyed_entity_stops_drawing_and_a_later_spawn_draws_its_own_matrix():
    e = _engine()
    a = _spawn(e, (1, 0, 0))
    _spawn(e, (2, 0, 0))
    f = _Frames(e)
    f.draw()
    a.destroy()
    np.testing.assert_allclose(f.draw(), [_model((2, 0, 0))], atol=1e-6)
    _spawn(e, (0, 9, 0))
    drawn = f.draw()
    np.testing.assert_allclose(sorted(drawn[:, :3, 3].tolist()), [[0, 9, 0], [2, 0, 0]], atol=1e-6)


def _bounds_each_frame(e):
    seen = []
    rp = e._render_pipeline
    real = rp._scene_bounds

    def record(engine):
        b = real(engine)
        seen.append(b)
        return b

    rp._scene_bounds = record
    return seen


def _encloses(bounds, points):
    center, radius = bounds
    return bool((np.linalg.norm(np.asarray(points) - center, axis=1) <= radius + 1e-4).all())


def _corners(pos, rot=(0, 0, 0, 1), scale=(1, 1, 1)):
    local = np.array([[x, y, z, 1] for x in (-0.5, 0.5) for y in (-0.5, 0.5) for z in (-0.5, 0.5)])
    return (local @ _model(pos, rot, scale).T)[:, :3]


def test_shadow_bounds_follow_moved_spawned_and_destroyed_entities():
    e = _engine(shadows=True)
    a = _spawn(e, (0, 0, 0))
    seen = _bounds_each_frame(e)
    e._draw_frame()
    assert seen and _encloses(seen[-1], _corners((0, 0, 0)))
    a.transform.pos = (10, 0, 0)
    seen.clear()
    e._draw_frame()
    assert seen and _encloses(seen[-1], _corners((10, 0, 0)))
    _spawn(e, (-10, 3, 0), QUARTER_Z, (4, 1, 1))
    seen.clear()
    e._draw_frame()
    assert seen and _encloses(seen[-1], np.vstack([_corners((10, 0, 0)), _corners((-10, 3, 0), QUARTER_Z, (4, 1, 1))]))
    a.destroy()
    seen.clear()
    e._draw_frame()
    center, radius = seen[-1]
    assert _encloses(seen[-1], _corners((-10, 3, 0), QUARTER_Z, (4, 1, 1)))
    assert radius < 5  # the destroyed cube at x=10 no longer widens it


def test_shadow_bounds_are_the_box_of_the_transformed_corners():
    rng = np.random.default_rng(3)
    e = _engine(shadows=True)
    pts = []
    for _ in range(20):
        q = rng.normal(size=4)
        q /= np.linalg.norm(q)
        pos, scale = rng.uniform(-8, 8, 3), rng.uniform(0.2, 3, 3)
        _spawn(e, pos, q, scale)
        pts.append(_corners(pos, q, scale))
    pts = np.vstack(pts)
    lo, hi = pts.min(axis=0), pts.max(axis=0)
    center, radius = e._render_pipeline._scene_bounds(e)
    np.testing.assert_allclose(center, (lo + hi) / 2, atol=1e-4)
    assert radius == pytest.approx(np.linalg.norm(hi - lo) / 2, abs=1e-4)


def test_scene_bounds_outside_a_frame_see_a_move_since_the_last_frame():
    e = _engine(shadows=True)
    h = _spawn(e, (0, 0, 0))
    e._draw_frame()
    h.transform.pos = (20, 0, 0)
    center, _ = e._render_pipeline._scene_bounds(e)
    np.testing.assert_allclose(center, (20, 0, 0), atol=1e-4)
