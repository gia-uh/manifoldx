import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))

import numpy as np

from manifoldx.modeling import Mesh
from manifoldx.resources import StandardMaterial

from gltf_helpers import accessor, load


def test_plain_mesh_gets_a_white_rough_material(tmp_path):
    report = Mesh.plane(width=2, depth=2, segments=2).to_gltf(tmp_path / "p.glb")
    g = load(tmp_path / "p.glb")
    pbr = g.materials[0].pbrMetallicRoughness
    assert pbr.baseColorFactor == [1, 1, 1, 1] and pbr.roughnessFactor == 1.0
    assert len(report) == 0


def test_coloured_mesh_keeps_its_colours(tmp_path):
    from manifoldx.resources import srgb_to_linear

    m = Mesh.plane(width=2, depth=2, segments=2)
    m = m.with_colors(np.full((len(m.positions), 3), 0.25, np.float32))
    m.to_gltf(tmp_path / "c.glb")
    g = load(tmp_path / "c.glb")
    attrs = g.meshes[0].primitives[0].attributes
    # sRGB in manifoldx, linear in glTF's COLOR_0.
    np.testing.assert_allclose(accessor(g, attrs.COLOR_0), srgb_to_linear(0.25), atol=1e-6)


def test_explicit_material(tmp_path):
    Mesh.plane(width=2, depth=2, segments=2).to_gltf(tmp_path / "m.glb", material=StandardMaterial("#000000", metallic=1))
    assert load(tmp_path / "m.glb").materials[0].pbrMetallicRoughness.metallicFactor == 1.0
