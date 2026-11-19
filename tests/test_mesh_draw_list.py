"""The main pass keeps each batch's resolved state between frames and sets no state twice in a row."""
import numpy as np
import pytest

import manifoldx as mx


def _engine(w=64, h=48):
    try:
        from manifoldx.backends import get_offscreen_canvas
        get_offscreen_canvas(width=8, height=8)
    except Exception as e:
        pytest.skip(f"offscreen canvas unavailable: {e}")
    e = mx.Engine("test", width=w, height=h, max_entities=4096)
    e.background_color = (0.0, 0.0, 0.0)
    e.camera.set_pose(position=(0, 0, 30), target=(0, 0, 0))
    return e


def _unique_cubes(e, n, mats, size=0.4):
    from manifoldx.components import Material, Mesh, Transform
    from manifoldx.resources import cube

    rng = np.random.default_rng(0)
    out = []
    for k, p in enumerate(rng.uniform(-8, 8, (n, 3)) * (1, 1, 0)):
        out.append(e.spawn(Mesh(cube(size, size, size)), Material(mats[k % len(mats)]), Transform(pos=tuple(p))))
    return out


def _count(cls, name, calls):
    real = getattr(cls, name)

    def spy(*a, **k):
        calls[name] = calls.get(name, 0) + 1
        return real(*a, **k)

    return real, spy


def test_still_scene_resolves_no_pipeline(monkeypatch):
    from manifoldx.renderer import RenderPipeline
    from manifoldx.resources import StandardMaterial

    e = _engine()
    _unique_cubes(e, 200, [StandardMaterial("#88aaff")])
    for _ in range(2):
        e.render_frame(supersample=1)
    calls = {}
    real, spy = _count(RenderPipeline, "_get_or_create_pipeline", calls)
    monkeypatch.setattr(RenderPipeline, "_get_or_create_pipeline", spy)
    for _ in range(3):
        e.render_frame(supersample=1)
    assert calls.get("_get_or_create_pipeline", 0) == 0


def test_one_material_sets_its_state_once(monkeypatch):
    import wgpu.backends.wgpu_native as native
    from manifoldx.resources import StandardMaterial

    e = _engine()
    _unique_cubes(e, 200, [StandardMaterial("#88aaff")])
    e.render_frame(supersample=1)
    e._render_pipeline._use_bundles = False  # direct draws: a bundle would hold them
    for _ in range(2):
        e.render_frame(supersample=1)
    calls = {}
    for name in ("set_pipeline", "draw_indexed"):
        real, spy = _count(native.GPURenderPassEncoder, name, calls)
        monkeypatch.setattr(native.GPURenderPassEncoder, name, spy)
    e.render_frame(supersample=1)
    assert calls["draw_indexed"] >= 200
    assert calls["set_pipeline"] <= 5  # skybox, sprites and the one mesh pipeline at most


def test_still_and_turning_views_replay_one_bundle(monkeypatch):
    import wgpu.backends.wgpu_native as native
    from manifoldx.resources import StandardMaterial

    e = _engine()
    _unique_cubes(e, 200, [StandardMaterial("#88aaff"), StandardMaterial("#ffaa44")])
    for _ in range(2):
        e.render_frame(supersample=1)
    if not e._render_pipeline._use_bundles:
        pytest.skip("device lacks indirect-first-instance")
    calls = {}
    for cls, name in [(native.GPUDevice, "create_render_bundle_encoder"),
                      (native.GPURenderPassEncoder, "execute_bundles"),
                      (native.GPURenderPassEncoder, "draw_indexed")]:
        real, spy = _count(cls, name, calls)
        monkeypatch.setattr(cls, name, spy)
    for x in (0.0, 0.0, 6.0, 12.0):  # still, then turning: other cubes culled
        e.camera.set_pose(position=(0, 0, 30), target=(x, 0, 0))
        e.render_frame(supersample=1)
    assert calls == {"execute_bundles": 4}



def test_a_spawn_or_destroy_records_one_bundle_again(monkeypatch):
    """A streamed cell that comes or goes re-records the one bundle it lands in, not all of them."""
    import wgpu.backends.wgpu_native as native
    from manifoldx.resources import StandardMaterial

    e = _engine()
    cells = _unique_cubes(e, 200, [StandardMaterial("#88aaff"), StandardMaterial("#ffaa44")])
    e.render_frame(supersample=1)
    if not e._render_pipeline._use_bundles:
        pytest.skip("device lacks indirect-first-instance")
    _unique_cubes(e, 1, [StandardMaterial("#88aaff")])  # grows the transforms buffer: every bundle once
    for _ in range(2):
        e.render_frame(supersample=1)
    calls = {}
    for cls, name in [(native.GPUDevice, "create_render_bundle_encoder"),
                      (native.GPURenderBundleEncoder, "draw_indexed_indirect")]:
        real, spy = _count(cls, name, calls)
        monkeypatch.setattr(cls, name, spy)
    recorded = []
    for change in (lambda: _unique_cubes(e, 1, [StandardMaterial("#88aaff")]), lambda: None,
                   lambda: e.destroy([cells[50].index]), lambda: None):
        change()
        e.render_frame(supersample=1)
        recorded.append((calls.pop("create_render_bundle_encoder", 0), calls.pop("draw_indexed_indirect", 0)))
    assert [n for n, _ in recorded] == [1, 0, 1, 0] and max(d for _, d in recorded) <= 32  # one chunk, not 202

def test_colour_change_shows():
    from manifoldx.resources import BasicMaterial

    e = _engine()
    mat = BasicMaterial("#ff0000")
    _unique_cubes(e, 1, [mat], size=20)
    for _ in range(3):
        before = e.render_frame(supersample=1)
    mat.color = "#0000ff"
    after = e.render_frame(supersample=1)
    assert before[24, 32, 0] > 200 and after[24, 32, 2] > 200 and after[24, 32, 0] < 50


def test_replaced_geometry_shows():
    from manifoldx.resources import BasicMaterial, cube

    e = _engine()
    (h,) = _unique_cubes(e, 1, [BasicMaterial("#ff0000")], size=1)
    e.store._components["Transform"][h.index, :3] = 0
    for _ in range(3):
        before = e.render_frame(supersample=1)
    h.set_geometry(cube(20, 20, 20))
    after = e.render_frame(supersample=1)
    assert (before.sum(axis=2) > 0).sum() < (after.sum(axis=2) > 0).sum()


def test_two_materials_render_as_before():
    from manifoldx.resources import StandardMaterial

    e = _engine()
    _unique_cubes(e, 300, [StandardMaterial("#88aaff"), StandardMaterial("#ffaa44")])
    frames = [e.render_frame(supersample=1) for _ in range(3)]
    assert np.array_equal(frames[1], frames[2]) and frames[2].sum() > 0


@pytest.fixture(params=["bundles", "direct"])
def city(request):
    """A still scene of many batches (unique cubes over two materials), drawn through a render
    bundle or with direct draws; the tests then change one thing and look for it on screen."""
    from manifoldx.resources import BasicMaterial

    e = _engine()
    _unique_cubes(e, 300, [BasicMaterial("#202020"), BasicMaterial("#303030")], size=0.2)
    e.render_frame(supersample=1)
    if request.param == "direct":
        e._render_pipeline._use_bundles = False
    elif not getattr(e._render_pipeline, "_use_bundles", True):
        pytest.skip("device lacks indirect-first-instance")
    for _ in range(2):
        e.render_frame(supersample=1)
    return e


def _red(img):
    return int(((img[..., 0] > 200) & (img[..., 1] < 50)).sum())


def _red_cube(e, at):
    from manifoldx.components import Material, Mesh, Transform
    from manifoldx.resources import BasicMaterial, cube

    return e.spawn(Mesh(cube(3, 3, 3)), Material(BasicMaterial("#ff0000")), Transform(pos=at))


def test_spawn_into_a_still_scene_shows(city):
    assert _red(city.render_frame(supersample=1)) == 0
    _red_cube(city, (0, 0, 5))
    assert _red(city.render_frame(supersample=1)) > 20


def test_destroy_from_a_still_scene_shows(city):
    h = _red_cube(city, (0, 0, 5))
    for _ in range(2):
        assert _red(city.render_frame(supersample=1)) > 20
    city.destroy([h.index])
    assert _red(city.render_frame(supersample=1)) == 0


def test_moved_transform_in_a_still_scene_shows(city):
    h = _red_cube(city, (-6, 0, 5))
    for _ in range(2):
        left = city.render_frame(supersample=1)
    city.store._components["Transform"][h.index, 0] = 6.0
    right = city.render_frame(supersample=1)
    cols = lambda img: np.flatnonzero(((img[..., 0] > 200) & (img[..., 1] < 50)).any(axis=0))  # noqa: E731
    assert cols(left).mean() < 32 < cols(right).mean()


def test_a_culled_mesh_shows_again_when_the_camera_turns_back(city):
    _red_cube(city, (0, 0, 5))
    seen = _red(city.render_frame(supersample=1))
    city.camera.set_pose(position=(0, 0, 30), target=(0, 200, 0))  # looking up: the cube is culled
    city.render_frame(supersample=1)
    city.camera.set_pose(position=(0, 0, 30), target=(0, 0, 0))
    assert seen > 20 and _red(city.render_frame(supersample=1)) == seen


def test_bundles_draw_what_direct_draws():
    """Two engines, one drawing through the bundle and one directly, agree on every frame while the
    camera turns, instances move, and batches come and go."""
    from manifoldx.components import Material, Mesh, Transform
    from manifoldx.resources import BasicMaterial, StandardMaterial, cube

    def scene():
        e = _engine()
        mats = [BasicMaterial(c) for c in ("#ff0000", "#00ff00", "#0000ff", "#ffff00")]
        cells = _unique_cubes(e, 120, mats, size=0.6)
        props = e.spawn(Mesh(cube(0.5, 0.5, 0.5)), Material(mats[3]),
                        Transform(pos=np.random.default_rng(1).uniform(-8, 8, (60, 3)) * (1, 1, 0)), n=60)
        glass = [StandardMaterial(c, opacity=0.5) for c in ("#00ffff", "#ff8800")]  # blended: order shows
        _unique_cubes(e, 6, glass, size=4.0)
        e.render_frame(supersample=1)
        return e, cells, props

    a, cells, props_a = scene()
    b, _, props_b = scene()
    if not a._render_pipeline._use_bundles:
        pytest.skip("device lacks indirect-first-instance")
    b._render_pipeline._use_bundles = False
    rows = np.arange(props_a.index, props_a.index + 20)  # spawned in one block
    assert props_a.index == props_b.index

    def step(fn):
        for e in (a, b):
            fn(e)
        ia, ib = a.render_frame(supersample=1), b.render_frame(supersample=1)
        assert ia.sum() > 0 and np.array_equal(ia, ib)

    for x in (0.0, 4.0, 9.0, 3.0):
        step(lambda e, x=x: e.camera.set_pose(position=(0, 0, 30), target=(x, 0, 0)))
    step(lambda e: e.store._components["Transform"].__setitem__((rows, 1), 40.0))  # props leave the view
    step(lambda e: e.store._components["Transform"].__setitem__((rows, 1), 0.0))
    step(lambda e: e.destroy([cells[k].index for k in (0, 5, 7)]))
    step(lambda e: _unique_cubes(e, 3, [BasicMaterial("#ff00ff")], size=1.0))
    step(lambda e: _unique_cubes(e, 2, [StandardMaterial("#00ff00", opacity=0.5)], size=3.0))
    step(lambda e: e.camera.set_pose(position=(0, 0, 30), target=(-6, 2, 0)))
