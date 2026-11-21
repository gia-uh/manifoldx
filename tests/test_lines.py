"""engine.lines: segments drawn a fixed width in screen pixels over the scene, never cut by the ground they lie
on, hidden by what stands in front."""

import numpy as np
import pytest

import manifoldx as mx
from manifoldx.components import Material, Mesh, Transform
from manifoldx.resources import FlatMaterial, plane


def _engine(distance: float):
    e = mx.Engine("lines", width=128, height=128)
    e.camera.set_pose(position=(0, 0, distance), target=(0, 0, 0))
    e.spawn(Mesh(plane(400, 400)), Material(FlatMaterial("#22cc22")), Transform(pos=(0, 0, 0)))
    e.lines.set("road", [(-50, 0, 0)], [(50, 0, 0)], color=(1.0, 0.0, 0.0, 1.0), width=6.0)
    return e


def _red_rows(img) -> int:
    red = (img[..., 0] > 200) & (img[..., 1] < 120)
    return int(red[:, 64].sum())


@pytest.mark.parametrize("distance", [10.0, 60.0])
def test_a_line_on_its_ground_is_drawn_its_width_in_pixels_at_any_distance(distance):
    img = _engine(distance).render_frame(supersample=1)
    assert 5 <= _red_rows(img) <= 7


def test_what_stands_in_front_hides_the_line():
    e = _engine(10.0)
    e.spawn(Mesh(plane(2, 2)), Material(FlatMaterial("#2222cc")), Transform(pos=(0, 0, 3)))
    img = e.render_frame(supersample=1)
    assert _red_rows(img) == 0 and img[64, 64, 2] > 150
    assert ((img[..., 0] > 200) & (img[..., 1] < 120))[:, 5].sum() >= 5  # beside the plane it still shows


def test_a_removed_layer_is_not_drawn():
    e = _engine(10.0)
    e.render_frame(supersample=1)
    e.lines.remove("road")
    assert _red_rows(e.render_frame(supersample=1)) == 0


def test_a_layers_look_scales_its_widths_without_setting_it_again():
    e = _engine(10.0)
    e.render_frame(supersample=1)
    version = e.lines.versions["road"]
    e.lines.look("road", scale=2.0)
    img = e.render_frame(supersample=1)
    assert 11 <= _red_rows(img) <= 13 and e.lines.versions["road"] == version
    e.lines.look("road", alpha=0.0)
    assert _red_rows(e.render_frame(supersample=1)) == 0
