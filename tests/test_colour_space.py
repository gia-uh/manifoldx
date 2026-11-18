"""Material colours are sRGB, like textures: color= and albedo_map agree."""
import numpy as np
import pytest

import manifoldx as mx
from manifoldx.components import Material, Mesh, Transform
from manifoldx.resources import (
    AtmosphereMaterial,
    BasicMaterial,
    PhongMaterial,
    PointLight,
    StandardMaterial,
    WaterMaterial,
    plane,
)


def _decode(c):
    return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4


EXPECTED = [_decode(0xCC / 255), _decode(0x22 / 255), _decode(0x22 / 255)]


@pytest.mark.parametrize("make", [
    lambda: BasicMaterial("#cc2222"),
    lambda: PhongMaterial("#cc2222"),
    lambda: StandardMaterial(color="#cc2222"),
    lambda: StandardMaterial(color=(0xCC / 255, 0x22 / 255, 0x22 / 255)),
])
def test_material_colours_are_decoded_from_srgb(make):
    data = make().get_data(1, None)
    assert np.allclose(data[0, :3], EXPECTED, atol=1e-6)


@pytest.mark.parametrize("cls", [WaterMaterial, AtmosphereMaterial])
def test_special_materials_decode_their_colour_too(cls):
    import inspect
    m = cls(color="#cc2222") if "color" in inspect.signature(cls).parameters else cls("#cc2222")
    data = np.asarray(m.get_data(1, None)).ravel()
    assert np.allclose(data[:3], EXPECTED, atol=1e-6)


def test_color_and_albedo_map_of_the_same_value_render_the_same(tmp_path):
    # Review F5: under one point light, color="#cc2222" rendered (111, 50, 50)
    # and a #cc2222 albedo_map rendered (98, 19, 19).
    try:
        from manifoldx.backends import get_offscreen_canvas
        get_offscreen_canvas(width=8, height=8)
    except Exception as e:
        pytest.skip(f"offscreen canvas unavailable: {e}")
    from PIL import Image
    from manifoldx.textures import load_texture
    png = tmp_path / "red.png"
    Image.new("RGB", (8, 8), (0xCC, 0x22, 0x22)).save(png)
    e = mx.Engine("cs", width=64, height=32)
    e.add_light(PointLight(color="#ffffff", intensity=20.0, position=(0, 0, 2.5)))
    e.spawn(Mesh(plane(1.6, 1.6)), Material(StandardMaterial(color="#cc2222", roughness=0.9)), Transform(pos=(-1, 0, 0)))
    e.render_frame(supersample=1)  # creates the device load_texture needs
    tex = load_texture(e, png)
    e.spawn(Mesh(plane(1.6, 1.6)), Material(StandardMaterial(color="#ffffff", roughness=0.9, albedo_map=tex)), Transform(pos=(1, 0, 0)))
    e.camera.set_pose(position=(0, 0, 3), target=(0, 0, 0))
    img = e.render_frame(supersample=1).astype(int)
    left, right = img[16, 16], img[16, 48]
    assert np.abs(left - right).max() <= 3, (left, right)


def test_vertex_colours_are_srgb_like_material_colours():
    """#c62828 as color= and as per-vertex colours must render the same.

    Vertex colours were uploaded as if they were linear light, so the same sRGB
    hex painted per vertex came out pale: (201, 97, 98) against (211, 151, 153).
    """
    try:
        from manifoldx.backends import get_offscreen_canvas
        get_offscreen_canvas(width=8, height=8)
    except Exception as e:
        pytest.skip(f"offscreen canvas unavailable: {e}")
    from manifoldx.modeling.mesh import Mesh as ModelMesh
    from manifoldx.resources import DirectionalLight

    hex_colour = "#c62828"
    rgb = np.array([int(hex_colour[i:i + 2], 16) / 255 for i in (1, 3, 5)], np.float32)
    out = []
    for per_vertex in (False, True):
        box = ModelMesh.box(2, 2, 2).recompute_normals()
        e = mx.Engine("cs", width=64, height=64)
        if per_vertex:
            box = box.with_colors(np.tile(rgb, (len(box.positions), 1)))
            mat = StandardMaterial("#ffffff", roughness=0.6, vertex_colors=True)
        else:
            mat = StandardMaterial(hex_colour, roughness=0.6)
        e.spawn(Mesh(box.to_geometry()), Material(mat), Transform())
        e.camera.set_pose(position=(0, 0, -6), target=(0, 0, 0))
        e.set_sun(DirectionalLight(color="#ffffff", intensity=3.2, direction=(0, 0, 1)))
        out.append(e.render_frame(supersample=1)[32, 32, :3].astype(int))
    assert np.abs(out[0] - out[1]).max() <= 4, out


def _textured_vs_plain(tmp_path, light):
    """Centre pixel of a color="#cc2222" plane and of a #cc2222 albedo_map plane, each
    rendered alone at the same spot under `light` (the spot does not light the two
    halves of a frame equally, so side-by-side planes cannot be compared)."""
    try:
        from manifoldx.backends import get_offscreen_canvas
        get_offscreen_canvas(width=8, height=8)
    except Exception as e:
        pytest.skip(f"offscreen canvas unavailable: {e}")
    from PIL import Image
    from manifoldx.textures import load_texture
    png = tmp_path / "red.png"
    Image.new("RGB", (8, 8), (0xCC, 0x22, 0x22)).save(png)
    out = []
    for textured in (False, True):
        e = mx.Engine("cs", width=32, height=32)
        light(e)
        e.camera.set_pose(position=(0, 0, 3), target=(0, 0, 0))
        if textured:
            e.render_frame(supersample=1)  # creates the device load_texture needs
            mat = StandardMaterial(color="#ffffff", roughness=0.9, albedo_map=load_texture(e, png))
        else:
            mat = StandardMaterial(color="#cc2222", roughness=0.9)
        e.spawn(Mesh(plane(1.6, 1.6)), Material(mat), Transform(pos=(0, 0, 0)))
        out.append(e.render_frame(supersample=1).astype(int)[16, 16])
    return out


def test_albedo_map_is_used_under_the_sun(tmp_path):
    # Issue #5: under the sun the textured plane rendered its white color=, grey.
    from manifoldx.resources import DirectionalLight
    left, right = _textured_vs_plain(
        tmp_path, lambda e: e.set_sun(DirectionalLight(color="#ffffff", intensity=3.0, direction=(0, 0, -1))))
    assert left[0] > left[1] + 40, left  # the plain plane is visibly red
    assert np.abs(left - right).max() <= 3, (left, right)


def test_albedo_map_is_used_under_a_spotlight(tmp_path):
    from manifoldx.resources import SpotLight
    # The shadowed spot is the one the shader's calculateSpot lights; angles in radians.
    left, right = _textured_vs_plain(tmp_path, lambda e: e.set_spot(
        SpotLight(color="#ffffff", intensity=40.0, position=(0, 0, 2.5), direction=(0, 0, -1),
                  inner_angle=0.6, outer_angle=0.9)))
    assert left[0] > left[1] + 40, left
    assert np.abs(left - right).max() <= 3, (left, right)
