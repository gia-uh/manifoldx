import json
import struct
from pathlib import Path

import numpy as np
import pytest

from manifoldx.gltf import Node, export_gltf
from manifoldx.resources import BasicMaterial, FlatMaterial, PhongMaterial, StandardMaterial, cube
from manifoldx.textures import TextureHandle

from gltf_helpers import accessor, load, view_bytes


def tri(n=3):
    pos = np.zeros((n, 3), np.float32)
    pos[1] = (1, 0, 0)
    pos[2] = (0, 1, 0)
    return {"positions": pos, "normals": np.tile([0, 0, 1], (n, 1)).astype(np.float32),
            "uvs": np.zeros((n, 2), np.float32), "indices": np.array([0, 1, 2], np.uint32)}


def test_glb_header_and_alignment(tmp_path):
    out = tmp_path / "a.glb"
    # tri()'s three uint16 indices take 6 bytes, so the next view needs padding
    export_gltf(out, [Node(tri(), StandardMaterial("#ffffff")), Node(cube(1, 1, 1), StandardMaterial("#ffffff"))])
    data = out.read_bytes()
    magic, version, length = struct.unpack("<III", data[:12])
    assert (magic, version, length) == (0x46546C67, 2, len(data))
    json_len, json_type = struct.unpack("<II", data[12:20])
    assert json_type == 0x4E4F534A and json_len % 4 == 0
    doc = json.loads(data[20:20 + json_len])
    assert all(v["byteOffset"] % 4 == 0 for v in doc["bufferViews"])


def test_geometry_and_transform_round_trip(tmp_path):
    geo = cube(2, 1, 1)
    out = tmp_path / "a.glb"
    export_gltf(out, [Node(geo, StandardMaterial("#ffffff"), pos=(1, 2, 3), rot=(0, 0.7071068, 0, 0.7071068),
                           scale=(2, 2, 2), name="box", extras={"pick": "b1"})])
    g = load(out)
    node = g.nodes[0]
    assert node.name == "box" and node.extras == {"pick": "b1"}
    assert node.translation == [1, 2, 3] and node.scale == [2, 2, 2]
    assert node.rotation == pytest.approx([0, 0.7071068, 0, 0.7071068])
    prim = g.meshes[node.mesh].primitives[0]
    np.testing.assert_allclose(accessor(g, prim.attributes.POSITION), geo["positions"])
    np.testing.assert_allclose(accessor(g, prim.attributes.NORMAL), geo["normals"])
    np.testing.assert_array_equal(accessor(g, prim.indices), geo["indices"])
    assert g.accessors[prim.attributes.POSITION].min == pytest.approx(geo["positions"].min(axis=0).tolist())


def test_identity_transform_is_omitted(tmp_path):
    out = tmp_path / "a.glb"
    export_gltf(out, [Node(cube(1, 1, 1), StandardMaterial("#ffffff"))])
    node = load(out).nodes[0]
    assert node.translation is None and node.rotation is None and node.scale is None
    assert node.name == "node_0"


def test_shared_geometry_is_written_once(tmp_path):
    geo, mat = cube(1, 1, 1), StandardMaterial("#ffffff")
    out = tmp_path / "a.glb"
    export_gltf(out, [Node(geo, mat, pos=(0, 0, 0)), Node(geo, mat, pos=(3, 0, 0))])
    g = load(out)
    assert len(g.meshes) == 1 and g.nodes[0].mesh == g.nodes[1].mesh == 0


def test_large_meshes_use_uint32_indices(tmp_path):
    n = 70_000
    geo = {"positions": np.random.default_rng(0).random((n, 3), dtype=np.float32),
           "indices": np.arange(n - n % 3, dtype=np.uint32)}
    small = tri()
    out = tmp_path / "a.glb"
    export_gltf(out, [Node(geo, StandardMaterial("#ffffff")), Node(small, StandardMaterial("#ffffff"))])
    g = load(out)
    big_idx = g.meshes[g.nodes[0].mesh].primitives[0].indices
    small_idx = g.meshes[g.nodes[1].mesh].primitives[0].indices
    assert g.accessors[big_idx].componentType == 5125
    assert g.accessors[small_idx].componentType == 5123


def test_standard_material_is_linear_pbr(tmp_path):
    out = tmp_path / "a.glb"
    export_gltf(out, [Node(tri(), StandardMaterial("#cc2222", roughness=0.4, metallic=0.6))])
    m = load(out).materials[0]
    r = (0.8 + 0.055) / 1.055
    assert m.pbrMetallicRoughness.baseColorFactor == pytest.approx([r ** 2.4, 0.016, 0.016, 1.0], abs=1e-3)
    assert m.pbrMetallicRoughness.roughnessFactor == pytest.approx(0.4)
    assert m.pbrMetallicRoughness.metallicFactor == pytest.approx(0.6)


def test_albedo_map_embeds_the_original_bytes(tmp_path):
    from PIL import Image

    src = tmp_path / "t.jpg"
    Image.new("RGB", (4, 4), (200, 100, 50)).save(src)
    tex = TextureHandle(id=7, texture=None, view=None, sampler=None, size=(4, 4), source=src)
    out = tmp_path / "a.glb"
    export_gltf(out, [Node(tri(), StandardMaterial("#ffffff", albedo_map=tex)),
                      Node(tri(), StandardMaterial("#ffffff", albedo_map=tex))])
    g = load(out)
    assert len(g.images) == 1 and g.images[0].mimeType == "image/jpeg"
    assert view_bytes(g, g.images[0].bufferView) == src.read_bytes()
    s = g.samplers[0]
    assert (s.wrapS, s.wrapT, s.minFilter, s.magFilter) == (10497, 10497, 9987, 9729)
    assert g.materials[0].pbrMetallicRoughness.baseColorTexture.index == 0


def test_texture_without_source_is_reported(tmp_path):
    tex = TextureHandle(id=3, texture=None, view=None, sampler=None, size=(4, 4))
    report = export_gltf(tmp_path / "a.glb", [Node(tri(), StandardMaterial("#ffffff", albedo_map=tex))])
    assert [(e.kind, e.effect) for e in report] == [("texture", "dropped")]
    assert load(tmp_path / "a.glb").materials[0].pbrMetallicRoughness.baseColorTexture is None


def test_vertex_colors_only_with_a_vertex_color_material(tmp_path):
    geo = tri() | {"colors": np.full((3, 3), 0.5, np.float32)}
    out = tmp_path / "a.glb"
    export_gltf(out, [Node(geo, StandardMaterial("#ffffff", vertex_colors=True)),
                      Node(geo, StandardMaterial("#ffffff"))])
    g = load(out)
    with_c = g.meshes[g.nodes[0].mesh].primitives[0].attributes
    without = g.meshes[g.nodes[1].mesh].primitives[0].attributes
    np.testing.assert_allclose(accessor(g, with_c.COLOR_0), geo["colors"])
    assert without.COLOR_0 is None


def test_flat_is_unlit_and_basic_is_approximated(tmp_path):
    out = tmp_path / "a.glb"
    report = export_gltf(out, [Node(tri(), FlatMaterial("#ff0000")), Node(tri(), BasicMaterial("#00ff00")),
                               Node(tri(), PhongMaterial("#0000ff")), Node(tri(), BasicMaterial((1, 1, 1, 0.5)))])
    g = load(out)
    assert "KHR_materials_unlit" in g.materials[0].extensions
    assert "KHR_materials_unlit" in g.extensionsUsed
    assert g.materials[1].pbrMetallicRoughness.roughnessFactor == 1.0
    assert g.materials[3].alphaMode == "BLEND"
    rows = {(e.name, e.effect, e.count) for e in report}
    assert ("BasicMaterial", "approximated", 2) in rows and ("PhongMaterial", "approximated", 1) in rows


def test_ao_is_reported(tmp_path):
    report = export_gltf(tmp_path / "a.glb", [Node(tri(), StandardMaterial("#ffffff", ao=0.5))])
    assert [(e.kind, e.name, e.effect) for e in report] == [("material", "StandardMaterial.ao", "dropped")]


def test_unknown_material_drops_the_node(tmp_path):
    class Glow:
        pass

    out = tmp_path / "a.glb"
    report = export_gltf(out, [Node(tri(), Glow()), Node(tri(), Glow()), Node(tri(), StandardMaterial("#ffffff"))])
    assert len(load(out).nodes) == 1
    assert [(e.kind, e.name, e.effect, e.count) for e in report] == [("material", "Glow", "dropped", 2)]


def test_bad_extras_name_the_node(tmp_path):
    with pytest.raises(TypeError, match="'wall'"):
        export_gltf(tmp_path / "a.glb", [Node(tri(), StandardMaterial("#ffffff"), name="wall", extras={"p": Path("x")})])


def test_empty_scene_is_valid(tmp_path):
    out = tmp_path / "a.glb"
    export_gltf(out, [])
    g = load(out)
    assert g.asset.version == "2.0" and not g.meshes and not g.accessors


def test_modeling_mesh_is_accepted(tmp_path):
    from manifoldx.modeling import Mesh as GeoMesh

    out = tmp_path / "a.glb"
    export_gltf(out, [Node(GeoMesh.plane(width=2, depth=2, segments=4), StandardMaterial("#ffffff"))])
    assert len(load(out).meshes) == 1


def test_one_geometry_shares_its_accessors_across_materials(tmp_path):
    geo = cube(1, 1, 1)
    out = tmp_path / "a.glb"
    export_gltf(out, [Node(geo, StandardMaterial("#ffffff")) for _ in range(30)])
    g = load(out)
    assert len(g.accessors) == len(geo) and len({m.primitives[0].attributes.POSITION for m in g.meshes}) == 1


def test_nan_is_refused_with_the_node_name(tmp_path):
    with pytest.raises(ValueError, match="'wall'"):
        export_gltf(tmp_path / "a.glb", [Node(tri(), StandardMaterial("#ffffff"), name="wall",
                                               extras={"a": float("nan")})])


def test_camera_on_its_target_is_reported_not_written_as_nan(tmp_path):
    from manifoldx.camera import Camera

    cam = Camera(position=(1, 2, 3), target=(0, 0, 0))
    cam.target = cam.position.copy()
    report = export_gltf(tmp_path / "a.glb", [], camera=cam)
    assert b"NaN" not in (tmp_path / "a.glb").read_bytes()
    assert [(e.kind, e.effect) for e in report] == [("camera", "dropped")]


def test_texture_is_not_tinted_by_the_colour(tmp_path):
    """manifoldx's textured shader ignores color= and vertex colours; glTF would multiply them in."""
    from PIL import Image

    src = tmp_path / "t.png"
    Image.new("RGB", (2, 2)).save(src)
    tex = TextureHandle(id=1, texture=None, view=None, sampler=None, size=(2, 2), source=src)
    geo = tri() | {"colors": np.full((3, 3), 0.5, np.float32)}
    out = tmp_path / "a.glb"
    export_gltf(out, [Node(geo, StandardMaterial("#cc2222", albedo_map=tex, vertex_colors=True))])
    g = load(out)
    assert g.materials[0].pbrMetallicRoughness.baseColorFactor == [1, 1, 1, 1]
    assert g.meshes[0].primitives[0].attributes.COLOR_0 is None


def test_extra_materials_are_written_and_mapped(tmp_path):
    from manifoldx.gltf.writer import build_glb

    used, unused = StandardMaterial("#ffffff"), StandardMaterial("#ff0000")
    data, report = build_glb([Node(tri(), used)], extra_materials=[(1, used), (7, unused)])
    out = tmp_path / "a.glb"
    out.write_bytes(data)
    g = load(out)
    assert g.scenes[0].extras["manifoldx"]["mx_materials"] == {"1": 0, "7": 1}
    assert len(g.materials) == 2 and len(report) == 0
    assert b"mx_materials" not in build_glb([Node(tri(), used)])[0]  # only asked for


def test_standard_opacity_writes_alpha_and_blend(tmp_path):
    """The desktop blends a StandardMaterial with opacity < 1; glTF carries that as the base
    colour's alpha (colour alpha times opacity) and BLEND, textured or not."""
    png = tmp_path / "t.png"
    png.write_bytes(b"\x89PNG\r\n\x1a\n")  # the writer embeds the bytes; only the magic matters
    tex = TextureHandle(id=3, texture=None, view=None, sampler=None, size=(4, 4), source=png)
    out = tmp_path / "a.glb"
    export_gltf(out, [Node(tri(), StandardMaterial("#ffffff", opacity=0.0)),
                      Node(tri(), StandardMaterial((1, 1, 1, 0.5), opacity=0.5)),
                      Node(tri(), StandardMaterial("#ffffff", opacity=0.25, albedo_map=tex)),
                      Node(tri(), StandardMaterial("#ffffff"))])
    g = load(out)
    alphas = [m.pbrMetallicRoughness.baseColorFactor[3] for m in g.materials]
    assert alphas == [0.0, 0.25, 0.25, 1.0]
    assert [m.alphaMode for m in g.materials] == ["BLEND", "BLEND", "BLEND", "OPAQUE"]
