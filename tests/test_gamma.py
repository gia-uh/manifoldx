"""Pixels are gamma-encoded exactly once."""
import pytest

import manifoldx as mx
from manifoldx.components import Material, Mesh, Transform
from manifoldx.resources import DirectionalLight, FlatMaterial, StandardMaterial, plane


def _engine():
    try:
        from manifoldx.backends import get_offscreen_canvas
        get_offscreen_canvas(width=8, height=8)
    except Exception as e:
        pytest.skip(f"offscreen canvas unavailable: {e}")
    e = mx.Engine("gamma", width=32, height=32)
    e.camera.set_pose(position=(0, 0, 3), target=(0, 0, 0))
    return e


def test_flat_colour_reads_back_as_written():
    e = _engine()
    e.spawn(Mesh(plane(10, 10)), Material(FlatMaterial("#cc2222")), Transform(pos=(0, 0, 0)))
    px = e.render_frame(supersample=1)[16, 16]
    assert all(abs(int(a) - b) <= 1 for a, b in zip(px, (0xCC, 0x22, 0x22))), px


def test_target_encodes_and_no_shader_encodes_again():
    from manifoldx import resources
    from manifoldx.render.passes import skybox
    e = _engine()
    e.render_frame(supersample=1)
    assert str(e._texture_format).endswith("srgb")
    sources = [v for k, v in vars(resources).items() if k.endswith("_SHADER") and isinstance(v, str)]
    sources += [v for k, v in vars(skybox).items() if isinstance(v, str) and "fn " in v]
    assert sources and not any("1.0 / 2.2" in s for s in sources)


def test_lit_colour_keeps_its_saturation():
    # #cc2222 under a sun, measured on zion: (216, 167, 167) with double gamma
    # (g/r 0.77); about 0.5 with gamma encoded once but the colour taken as
    # linear; (164, 47, 47) with the colour decoded from sRGB (g/r 0.29).
    e = _engine()
    e.set_sun(DirectionalLight(color="#ffffff", intensity=3.0, direction=(0, 0, -1)))
    e.spawn(Mesh(plane(10, 10)), Material(StandardMaterial(color="#cc2222", roughness=0.9)), Transform(pos=(0, 0, 0)))
    r, g, b = (int(v) for v in e.render_frame(supersample=1)[16, 16])
    assert g / r < 0.40, (r, g, b)


def test_a_color_object_is_a_material_colour():
    """StandardMaterial(color=Color(...)) failed at spawn: 'Color' object is not subscriptable."""
    import numpy as np

    from manifoldx.resources import material_rgb
    from manifoldx.types import Color

    np.testing.assert_allclose(material_rgb(Color("#cc2222")), material_rgb("#cc2222"), atol=1e-6)
    np.testing.assert_allclose(material_rgb(Color(linear_r=0.2, linear_g=0.1, linear_b=0.0)), [0.2, 0.1, 0.0], atol=1e-6)
