"""The renderer recomputes a model matrix only when its Transform row changed, whatever wrote it."""

import numpy as np

import manifoldx as mx
from manifoldx.components import Material, Mesh, Transform
from manifoldx.renderer import TransformCache
from manifoldx.resources import StandardMaterial


def _scene(n=4):
    e = mx.Engine("transform-changes", width=32, height=32)
    h = e.spawn(Mesh(mx.geometry.cube(1.0, 1.0, 1.0)), Material(StandardMaterial("#cc3333")),
                Transform(pos=np.zeros((n, 3), np.float32)), n=n)
    e.render_frame(supersample=1)
    return e, h


def _matrix(e, i):
    rp = e._render_pipeline
    return rp._transform_cache.get_transforms(e.store, np.array([i]))[0]


def _rows_recomputed(e, monkeypatch):
    rows = []
    orig = TransformCache._recompute_matrices

    def spy(self, store, idx):
        rows.append(len(idx))
        return orig(self, store, idx)

    monkeypatch.setattr(TransformCache, "_recompute_matrices", spy)
    e.render_frame(supersample=1)
    monkeypatch.undo()
    return sum(rows)


def test_static_entities_are_not_recomputed(monkeypatch):
    e, _ = _scene()
    assert _rows_recomputed(e, monkeypatch) == 0


def test_a_handle_write_reaches_the_next_frame(monkeypatch):
    e, h = _scene()
    h.transform.pos = (3.0, 0.0, 0.0)
    assert _rows_recomputed(e, monkeypatch) == 1
    assert np.isclose(_matrix(e, h.index)[3], 3.0)


def test_a_query_write_reaches_the_next_frame():
    e, h = _scene()

    @e.system
    def move(q: mx.Query[Transform], dt):
        q[Transform].pos = np.full((len(q), 3), 2.0, np.float32)

    e.render_frame(supersample=1)
    assert np.isclose(_matrix(e, h.index)[7], 2.0)


def test_an_in_place_write_to_the_store_reaches_the_next_frame():
    e, h = _scene()
    e.store._components["Transform"][h.index, 7:10] = 5.0  # scale, written straight into the array
    e.render_frame(supersample=1)
    assert np.isclose(_matrix(e, h.index)[0], 5.0)


def test_a_reused_slot_gets_its_new_transform():
    e, h = _scene(1)
    i = h.index
    e.destroy(np.array([i]))
    e.render_frame(supersample=1)
    h2 = e.spawn(Mesh(mx.geometry.cube(1.0, 1.0, 1.0)), Material(StandardMaterial("#cc3333")),
                 Transform(pos=np.array([[0.0, 4.0, 0.0]], np.float32)))
    e.render_frame(supersample=1)
    assert h2.index == i and np.isclose(_matrix(e, i)[7], 4.0)


def _sprite_scene(n=5):
    from manifoldx.viz import ColormapMaterial, PointCloud, Radius, ScalarValue

    e = mx.Engine("sprite-uploads", width=32, height=32)
    e.camera.fit(10)
    e.spawn(PointCloud(), Material(ColormapMaterial(cmap="viridis", vmin=0, vmax=1)),
            Transform(pos=np.zeros((n, 3), np.float32)), ScalarValue(value=np.full(n, 0.5, np.float32)),
            Radius(radius=np.full(n, 0.1, np.float32)), n=n)
    e.render_frame(supersample=1)
    return e


def _uploads(e, monkeypatch, change=None):
    from manifoldx.renderer import _BatchBuffers

    calls = []
    orig, orig_rows = _BatchBuffers.upload_transforms, _BatchBuffers.write_rows

    def spy(self, data):
        calls.append(len(data))
        return orig(self, data)

    def spy_rows(self, name, rows, values):  # a few changed rows go up alone
        if name == "transforms":
            calls.append(len(rows))
        return orig_rows(self, name, rows, values)

    monkeypatch.setattr(_BatchBuffers, "upload_transforms", spy)
    monkeypatch.setattr(_BatchBuffers, "write_rows", spy_rows)
    if change:
        change(e)
    e.render_frame(supersample=1)
    monkeypatch.undo()
    return len(calls)


def test_a_static_sprite_cloud_is_not_uploaded_again(monkeypatch):
    e = _sprite_scene()
    assert _uploads(e, monkeypatch) == 0


def test_sprite_moves_values_and_radii_are_uploaded(monkeypatch):
    e = _sprite_scene()
    alive = np.flatnonzero(e.store._alive)
    assert _uploads(e, monkeypatch, lambda e: e.store._components["Transform"].__setitem__((alive[0], 0), 1.0)) == 1
    assert _uploads(e, monkeypatch, lambda e: e.store._components["ScalarValue"].__setitem__((alive[1], 0), 0.9)) == 1
    assert _uploads(e, monkeypatch, lambda e: e.store._components["Radius"].__setitem__((alive[2], 0), 0.3)) == 1
    assert _uploads(e, monkeypatch) == 0


def _transform_uploads(e, monkeypatch, change=None):
    from manifoldx.renderer import _BatchBuffers

    calls = []
    orig = _BatchBuffers.upload_transforms

    def spy(self, data):
        calls.append(self)
        return orig(self, data)

    monkeypatch.setattr(_BatchBuffers, "upload_transforms", spy)
    if change:
        change(e)
    e.render_frame(supersample=1)
    monkeypatch.undo()
    rp = e._render_pipeline
    return {"mesh": sum(b is rp._batch_buffers for b in calls), "shadow": sum(b is rp._shadow_batch_buffers for b in calls)}


def _shadowed_scene():
    from manifoldx.resources import DirectionalLight

    e, h = _scene()
    e.set_sun(DirectionalLight(color="#ffffff", intensity=3.0, direction=(-0.5, -1.0, -0.3)))
    e.enable_shadows(target=(0, 0, 0), extent=8.0, resolution=256)
    e.render_frame(supersample=1)
    return e, h


def test_static_meshes_and_shadow_casters_are_not_uploaded_again(monkeypatch):
    e, _ = _shadowed_scene()
    assert _transform_uploads(e, monkeypatch) == {"mesh": 0, "shadow": 0}


def test_a_moved_mesh_is_uploaded_to_both_passes(monkeypatch):
    e, h = _shadowed_scene()

    def move(e):
        h.transform.pos = (0.5, 0.0, 0.0)

    assert _transform_uploads(e, monkeypatch, move) == {"mesh": 1, "shadow": 1}
    assert _transform_uploads(e, monkeypatch) == {"mesh": 0, "shadow": 0}
