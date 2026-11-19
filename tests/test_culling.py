"""Frustum culling: a mesh whose bounding sphere lies outside the camera's view is not drawn,
and one outside the shadow caster's box casts nothing; what is in view renders as before."""
import numpy as np
import pytest

import manifoldx as mx
from manifoldx.components import Material, Mesh, Transform
from manifoldx.resources import DirectionalLight, StandardMaterial


def _engine():
    try:
        from manifoldx.backends import get_offscreen_canvas
        get_offscreen_canvas(width=8, height=8)
    except Exception as e:
        pytest.skip(f"offscreen canvas unavailable: {e}")
    e = mx.Engine("culling", width=64, height=64)
    e.set_sun(DirectionalLight(color="#ffffff", intensity=3.0, direction=(-0.3, -1.0, -0.5)))
    e.camera.set_pose(position=(0.0, 0.0, 6.0), target=(0.0, 0.0, 0.0))
    return e


def _cube(e, at, **mat):
    return e.spawn(Mesh(mx.geometry.cube(1.0, 1.0, 1.0)), Material(StandardMaterial("#cc3333", **mat)),
                   Transform(pos=at))


def test_a_mesh_out_of_view_is_not_drawn():
    e = _engine()
    _cube(e, (0.0, 0.0, 0.0))
    e.render_frame(supersample=1)
    alone = e._render_pipeline.drawn
    _cube(e, (0.0, -1000.0, 0.0))     # parked far below, as a game parks what waits
    _cube(e, (0.0, 0.0, 20.0))        # behind the camera
    _cube(e, (40.0, 0.0, 0.0))        # far to the side
    e.render_frame(supersample=1)
    assert alone == 1 and e._render_pipeline.drawn == 1


def test_what_is_in_view_renders_as_before():
    e = _engine()
    _cube(e, (0.0, 0.0, 0.0))
    before = e.render_frame(supersample=1)
    _cube(e, (0.0, 0.0, 20.0))
    _cube(e, (2.4, 0.0, 0.0))         # half out of the side of the view: still drawn
    e.render_frame(supersample=1)
    assert e._render_pipeline.drawn == 2
    e2 = _engine()
    _cube(e2, (0.0, 0.0, 0.0))
    assert np.array_equal(before, e2.render_frame(supersample=1))


def test_a_mirrored_mesh_is_culled_where_its_reflection_shows():
    e = _engine()
    e.camera.set_pose(position=(0.0, -8.0, 6.0), target=(0.0, -8.0, 0.0))   # looking at y = -8
    _cube(e, (0.0, 8.0, 0.0), mirror=0.0)    # stands at +8, its reflection at -8, in view
    e.render_frame(supersample=1)
    assert e._render_pipeline.drawn == 1


def test_a_mesh_outside_the_shadow_box_casts_nothing():
    e = _engine()
    e.enable_shadows(extent=4.0, auto_fit=False)
    _cube(e, (0.0, 0.0, 0.0))
    _cube(e, (0.0, -1000.0, 0.0))
    e.render_frame(supersample=1)
    assert sum(len(m) for m, _ in e._render_pipeline._collect_mesh_instances(e, with_entities=True, cull=True).values()) == 1


def test_a_mesh_moved_into_and_out_of_view_is_culled_where_it_is_now():
    e = _engine()
    _cube(e, (0.0, 0.0, 0.0))
    far = _cube(e, (40.0, 0.0, 0.0))
    for _ in range(2):
        e.render_frame(supersample=1)
    drawn = [e._render_pipeline.drawn]
    e.store._components["Transform"][far.index, 0] = 1.5   # into view, its sphere follows
    shown = e.render_frame(supersample=1)
    drawn.append(e._render_pipeline.drawn)
    e.store._components["Transform"][far.index, 0] = 40.0
    e.render_frame(supersample=1)
    drawn.append(e._render_pipeline.drawn)
    assert drawn == [1, 2, 1] and (shown[:, 40:, 0] > 100).any()


def test_a_stretched_camera_keeps_what_its_scale_brings_into_view():
    e = _engine()
    e.camera.set_pose(position=(0.0, 4.5, 6.0), target=(0.0, 4.5, 0.0))   # the view starts about 1 m up
    e.camera.vertical_scale = 3.0     # the cube's top, at 0.5 m, shows at 1.5 m
    _cube(e, (0.0, 0.0, 0.0))
    e.render_frame(supersample=1)
    assert e._render_pipeline.drawn == 1


def test_meshes_all_moved_at_once_are_culled_where_they_are_now():
    e = _engine()
    a, b = _cube(e, (0.0, 0.0, 0.0)), _cube(e, (40.0, 0.0, 0.0))
    for _ in range(2):
        e.render_frame(supersample=1)
    drawn = [e._render_pipeline.drawn]
    e.store._components["Transform"][[a.index, b.index], 0] = (40.0, 1.5)   # swap: every sphere is stale
    e.render_frame(supersample=1)
    drawn.append(e._render_pipeline.drawn)
    e.store._components["Transform"][[a.index, b.index], 0] = (0.0, -1.5)
    e.render_frame(supersample=1)
    drawn.append(e._render_pipeline.drawn)
    assert drawn == [1, 1, 2]
