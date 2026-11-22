"""Blended skinning: two joints and a weight per vertex, packed in the joint attribute."""

from functools import lru_cache

import numpy as np
import pytest

import manifoldx as mx
from manifoldx.components import Material, Mesh as MeshC, Transform
from manifoldx.modeling import Mesh
from manifoldx.modeling.assets.human import HumanParams, build_human
from manifoldx.modeling.assets.human_sections import JOINTS
from manifoldx.resources import DirectionalLight, PointLight, StandardMaterial
from manifoldx.skinning import pack_joints, unpack_joints


@lru_cache(maxsize=None)
def _figure():
    return build_human(HumanParams())


def _rotation(axis, degrees, pivot):
    axis = np.asarray(axis, float) / np.linalg.norm(axis)
    t = np.radians(degrees)
    k = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    r = np.eye(3) + np.sin(t) * k + (1 - np.cos(t)) * k @ k
    m = np.eye(4)
    m[:3, :3] = r
    m[:3, 3] = pivot - r @ pivot
    return m


def _blend(mesh, mats):
    """CPU linear blend: each vertex by w * M[a] + (1 - w) * M[b]."""
    a, b = mesh.joints[:, 0], mesh.joints[:, 1]
    w = mesh.weights[:, None, None]
    m = w * mats[a] + (1 - w) * mats[b]
    p = np.c_[mesh.positions, np.ones(len(mesh.positions))]
    return np.einsum("nij,nj->ni", m, p)[:, :3]


def _pose(f, bends):
    """Joint matrices bending each named joint by its degrees, its subtree following."""
    names = list(JOINTS)
    mats = np.tile(np.eye(4), (len(names), 1, 1))
    for joint, degrees in bends.items():
        child = next(n for n, p in f.parents.items() if p == joint)
        bone = f.joints[child] - f.joints[joint]
        knee = joint.startswith("knee")
        axis = np.cross(bone, [0, 0, -1.0] if knee else [0, 0, 1.0])  # tips forward, a knee's backward
        r = _rotation(axis, degrees, f.joints[joint])
        sub, grew = {joint}, True
        while grew:
            more = {n for n, p in f.parents.items() if p in sub} - sub
            grew = bool(more)
            sub |= more
        for n in sub:
            mats[names.index(n)] = r @ mats[names.index(n)]
    return mats


def _cpu_skin(f, bends):
    """The figure's rest mesh with each named joint bent by its degrees."""
    return _blend(f.mesh, _pose(f, bends))


def _normals(positions, faces):
    a, b, c = (positions[faces[:, i]] for i in range(3))
    return np.cross(b - a, c - a)


def _flipped_faces(mesh, posed, mats):
    """Faces whose posed normal points against their vertices' blended rotation of the rest normal."""
    a, b = mesh.joints[:, 0], mesh.joints[:, 1]
    w = mesh.weights[:, None, None]
    m = (w * mats[a] + (1 - w) * mats[b])[:, :3, :3]
    turn = m[mesh.faces].mean(1)  # the face's vertices' blended linear part
    rest = np.einsum("fij,fj->fi", turn, _normals(mesh.positions.astype(float), mesh.faces))
    after = _normals(posed, mesh.faces)
    big = np.linalg.norm(rest, axis=1) > 1e-10
    return int(((np.einsum("ij,ij->i", rest, after) < 0) & big).sum())


# --- packing ---------------------------------------------------------------------------------------


def test_rigid_pack_roundtrip():
    a, b, w = unpack_joints(pack_joints(np.array([0, 3, 31])))
    assert list(a) == list(b) == [0, 3, 31] and np.all(w == 1)


def test_blend_pack_roundtrip():
    a, b, w = unpack_joints(pack_joints(np.array([[1, 2], [5, 6]]), np.array([0.25, 1.0])))
    assert list(a) == [1, 5] and list(b) == [2, 6] and np.allclose(w, [0.25, 1.0], atol=1e-4)


def test_mesh_carries_two_joints_and_weights():
    pos = np.zeros((3, 3), np.float32)
    m = Mesh(pos, np.array([[0, 1, 2]], np.uint32), joints=np.array([[0, 1], [1, 2], [2, 2]]),
             weights=np.array([0.5, 0.75, 1.0]))
    geo = m.to_geometry()
    assert geo["joints"].shape == (3, 2) and np.allclose(geo["weights"], [0.5, 0.75, 1.0])
    assert np.array_equal(Mesh.from_geometry(geo).weights, m.weights)


@pytest.mark.parametrize("weights", [None, np.array([0.5, 2.0, 1.0]), np.array([0.5, 0.5])])
def test_bad_blend_weights_are_rejected(weights):
    with pytest.raises(ValueError, match="weights"):
        Mesh(np.zeros((3, 3), np.float32), np.array([[0, 1, 2]], np.uint32),
             joints=np.array([[0, 1], [1, 2], [2, 2]]), weights=weights)


# --- the figure ------------------------------------------------------------------------------------


def test_weights_sum_and_bounds():
    f = _figure()
    assert f.mesh.joints.shape == (len(f.mesh.positions), 2)
    assert ((f.mesh.weights >= 0) & (f.mesh.weights <= 1)).all()


def test_rest_pose_is_the_rest_mesh():
    f = _figure()
    assert np.allclose(_cpu_skin(f, {}), f.mesh.positions, atol=1e-6)


def test_joints_blend_across_the_elbow():
    f = _figure()
    names = list(JOINTS)
    elbow, shoulder = names.index("elbow_l"), names.index("shoulder_l")
    pair = {tuple(sorted(p)) for p in f.mesh.joints.tolist()}
    assert tuple(sorted((shoulder, elbow))) in pair  # some vertices share the two


@pytest.mark.parametrize(
    "joint, degrees",
    [("elbow_l", 90), ("knee_r", 90), ("shoulder_l", 90), ("shoulder_r", 90), ("hip_l", 30), ("hip_r", -45)],
)
def test_bend_no_flips(joint, degrees):
    """CPU linear blend of the rest mesh: no face turns over. The hip is held to a walk's range: past about
    45 degrees of flexion a few faces fold inside the groin crease, as linear blending does."""
    f = _figure()
    mats = _pose(f, {joint: degrees})
    assert _flipped_faces(f.mesh, _blend(f.mesh, mats), mats) == 0


# --- on the GPU ------------------------------------------------------------------------------------


def _bar():
    """A square bar along x from -0.5 to 0.5, 20 segments: joint 0 left, joint 1 right, blended in the middle."""
    xs = np.linspace(-0.5, 0.5, 21)
    ring = np.array([[0, -0.08, -0.08], [0, 0.08, -0.08], [0, 0.08, 0.08], [0, -0.08, 0.08]])
    pos = np.concatenate([ring + [x, 0, 0] for x in xs]).astype(np.float32)
    faces = []
    for i in range(20):
        for k in range(4):
            a, b = 4 * i + k, 4 * i + (k + 1) % 4
            faces += [[a, b, b + 4], [a, b + 4, a + 4]]
    faces += [[0, 2, 1], [0, 3, 2], [80, 81, 82], [80, 82, 83]]
    w = np.clip(0.5 - np.repeat(xs, 4) / 0.3, 0.0, 1.0)  # 1 on the left, 0 on the right, over 0.3 m
    joints = np.tile([0, 1], (len(pos), 1))
    return Mesh(pos, np.array(faces, np.uint32), joints=joints, weights=w).recompute_normals()


def _engine():
    try:
        from manifoldx.backends import get_offscreen_canvas

        get_offscreen_canvas(width=8, height=8)
    except Exception as e:
        pytest.skip(f"offscreen canvas unavailable: {e}")
    e = mx.Engine("blend", width=64, height=48)
    e.add_light(PointLight(color="#ffffff", intensity=25.0, position=(1.5, 2.0, 3.0)))
    e.set_sun(DirectionalLight(color="#ffffff", intensity=2.0, direction=(-0.3, -0.5, -1.0)))
    e.camera.set_pose(position=(0.0, 0.4, 1.8), target=(0, 0, 0))
    return e


def test_gpu_matches_cpu():
    bar = _bar()
    mats = np.stack([np.eye(4), _rotation([0, 0, 1], 45, np.zeros(3))]).astype(np.float32)
    mat = StandardMaterial(color="#d0a060", roughness=0.6)

    e = _engine()
    h = e.spawn(MeshC(bar.to_geometry()), Material(mat), Transform(pos=(0, 0, 0)))
    h.set_joints(mats)
    gpu = e.render_frame(supersample=1).astype(np.float64)

    e = _engine()
    posed = Mesh(_blend(bar, mats.astype(float)).astype(np.float32), bar.faces).recompute_normals()
    e.spawn(MeshC(posed.to_geometry()), Material(mat), Transform(pos=(0, 0, 0)))
    cpu = e.render_frame(supersample=1).astype(np.float64)

    assert np.abs(gpu - cpu).mean() < 1.0
    assert int((np.abs(gpu - cpu).max(axis=2) > 16).sum()) <= 8
