"""Engine.export_gltf on a skinned entity: JOINTS_0, WEIGHTS_0, a skin and its current pose (#82)."""

import math

import numpy as np
import pytest
from manifoldx.gltf.engine_export import export_engine

from gltf_helpers import accessor, load, skinned_figure


def _export(tmp_path, posed=True, **kw):
    e, h, geo, P = skinned_figure(posed)
    path = tmp_path / "s.glb"
    report = e.export_gltf(path, **kw)
    return load(path), report, h, geo, P


@pytest.fixture
def skinned_glb(tmp_path):
    return _export(tmp_path)


def _skinned_node(g):
    (node,) = [n for n in g.nodes if n.skin is not None]
    return node


def test_a_skinned_entity_has_joints_weights_and_a_skin(skinned_glb):
    g, _report, _h, geo, _P = skinned_glb
    node = _skinned_node(g)
    prim = g.meshes[node.mesh].primitives[0]
    joints = accessor(g, prim.attributes.JOINTS_0)
    weights = accessor(g, prim.attributes.WEIGHTS_0)
    assert g.accessors[prim.attributes.JOINTS_0].componentType == 5121
    assert joints.shape == (len(geo["positions"]), 4) and weights.dtype == np.float32
    assert joints[:, 0].tolist() == np.asarray(geo["joints"]).tolist() and not joints[:, 1:].any()
    assert (weights == [1.0, 0.0, 0.0, 0.0]).all()
    skin = g.skins[node.skin]
    assert len(skin.joints) == 4
    assert (accessor(g, skin.inverseBindMatrices) == np.eye(4, dtype=np.float32).ravel()).all()


def test_the_joint_nodes_hold_the_pose_under_the_entitys_transform(skinned_glb):
    g, _report, h, _geo, P = skinned_glb
    node = _skinned_node(g)
    assert node.name == f"entity_{h.index}" and node.extras["mx"]["entity"] == h.index
    assert node.translation is None and node.rotation is None and node.scale is None and node.matrix is None
    skin = g.skins[node.skin]
    skeleton = g.nodes[skin.skeleton]
    assert skeleton.name == f"entity_{h.index}_skeleton"
    assert skeleton.translation == [1, 2, 3] and skeleton.scale == [2, 2, 2]
    assert np.allclose(skeleton.rotation, [0, math.sin(0.25), 0, math.cos(0.25)], atol=1e-7)
    assert skeleton.children == skin.joints
    assert skin.skeleton in g.scenes[0].nodes
    for j, k in enumerate(skin.joints):
        assert g.nodes[k].name == f"entity_{h.index}_joint_{j}"
        assert g.nodes[k].matrix == P[j].T.ravel().tolist()  # glTF matrices are column-major


def test_an_unposed_skinned_entity_has_identity_joints(tmp_path):
    g, *_ = _export(tmp_path, posed=False)
    skin = g.skins[_skinned_node(g).skin]
    for k in skin.joints:
        assert g.nodes[k].matrix == np.eye(4).ravel().tolist()


def test_strict_export_of_a_posed_skinned_entity_passes(tmp_path):
    _g, report, *_ = _export(tmp_path, strict=True)
    assert len(report) == 0


def test_library_mode_writes_joints_and_no_skin(tmp_path):
    e, _h, _geo, _P = skinned_figure()
    path = tmp_path / "lib.glb"
    export_engine(e, path, library=True)
    g = load(path)
    (gid,) = [gid for gid, geo in e._geometry_registry._geometries.items() if "joints" in geo]
    mesh = g.scenes[0].extras["manifoldx"]["mx_geometries"][str(gid)]
    attrs = g.meshes[mesh].primitives[0].attributes
    assert attrs.JOINTS_0 is not None and attrs.WEIGHTS_0 is not None
    assert not g.skins
    assert all(n.mesh != mesh for n in g.nodes)


def test_pose_reads_back_what_set_joints_wrote():
    e, h, _geo, P = skinned_figure()
    joints = e._render_pipeline._joints
    pose = joints.pose(h.index, 4)
    assert pose.dtype == np.float32 and (pose == P).all()
    other, *_ = skinned_figure(posed=False)
    assert (other._render_pipeline._joints.pose(0, 4) == np.eye(4, dtype=np.float32)).all()
    e.store.destroy([h.index])
    assert (joints.pose(h.index, 4) == np.eye(4, dtype=np.float32)).all()


def test_joint_count_is_the_largest_joint_plus_one():
    from manifoldx.skinning import joint_count

    _e, _h, geo, _P = skinned_figure(posed=False)
    assert joint_count(geo) == 4
