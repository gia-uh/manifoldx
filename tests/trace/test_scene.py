import numpy as np

from manifoldx.gltf import Node, build_glb
from manifoldx.resources import StandardMaterial
from manifoldx.trace.gltf_read import parse_glb
from manifoldx.trace.scene import Material, Scene
from manifoldx.trace.scenes import box


class _Cam:
    position, target, up, fov, near, far = (1.0, 2.0, 5.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), 45.0, 0.1, 100.0


def _glb(tmp_path, **node):
    geo = box((0, 0, 0), (1, 2, 3))
    mat = StandardMaterial(color=(1.0, 0.5, 0.25), roughness=0.3, metallic=0.7)
    data, _report = build_glb([Node(geo, mat, name="thing", **node)], camera=_Cam())
    path = tmp_path / "s.glb"
    path.write_bytes(data)
    return path, geo


def test_a_glb_reads_back_in_world_space(tmp_path):
    q = (0.0, np.sin(np.pi / 8), 0.0, np.cos(np.pi / 8))  # 45 degrees about y
    path, geo = _glb(tmp_path, pos=(1.0, 2.0, 3.0), rot=q)
    s = Scene.from_glb(path)
    c, sn = np.cos(np.pi / 4), np.sin(np.pi / 4)
    rot = np.array([[c, 0, sn], [0, 1, 0], [-sn, 0, c]])
    want = (geo["positions"] @ rot.T + (1, 2, 3))[geo["indices"].reshape(-1, 3)]
    assert s.triangle_count == 12
    assert np.allclose(s.positions, want, atol=1e-5)
    assert np.allclose(np.linalg.norm(s.normals, axis=2), 1, atol=1e-5)


def test_materials_and_camera_survive(tmp_path):
    path, _ = _glb(tmp_path)
    s = Scene.from_glb(path)
    m = s.materials[s.material_ids[0]]
    from manifoldx.resources import material_rgb

    assert np.allclose(m.base_color, material_rgb((1.0, 0.5, 0.25)), atol=1e-6)
    assert (m.roughness, m.metallic, m.transmission) == (0.3, 0.7, 0.0)
    assert np.allclose(s.camera.position, (1, 2, 5)) and abs(s.camera.fov - 45) < 1e-4
    f, _, _ = s.camera.basis()
    assert np.allclose(f, np.array([0, -2, -5]) / np.sqrt(29), atol=1e-6)


def test_overrides_match_node_then_material_names(tmp_path):
    path, _ = _glb(tmp_path)
    s = Scene.from_glb(path, overrides={"StandardMaterial": {"roughness": 0.9}, "th*": {"transmission": 1.0}})
    m = s.materials[s.material_ids[0]]
    assert (m.roughness, m.transmission) == (0.9, 1.0)


def test_blend_alpha_reads_as_thin_glass(tmp_path):
    path, _ = _glb(tmp_path)
    doc, binary = parse_glb(path.read_bytes())
    doc["materials"][0]["pbrMetallicRoughness"]["baseColorFactor"][3] = 0.25
    doc["materials"][0]["alphaMode"] = "BLEND"
    from manifoldx.trace.gltf_read import _Reader

    m = _Reader(doc, binary).materials[0]
    assert m.transmission == 0.75 and m.thin


def test_from_meshes_fills_flat_normals_and_white_colours():
    s = Scene.from_meshes([(box((0, 0, 0), (1, 1, 1)), Material("m"))])
    assert np.allclose(s.colors, 1)
    n = s.normals[:, 0]
    assert np.allclose(np.abs(n).max(axis=1), 1)


def test_an_override_drops_a_texture():
    from manifoldx.trace.scene import assemble

    tex = Material("floor", texture=0)
    parts = [("floor-a", box((0, 0, 0), (1, 1, 1))["positions"], np.arange(36).reshape(-1, 3) % 24, {}, 0)]
    s = assemble(parts, [tex], textures=[np.zeros((2, 2, 4), np.uint8)], overrides={"floor-*": {"texture": None}})
    assert s.materials[s.material_ids[0]].texture == -1
