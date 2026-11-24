"""Baked pose frames: shared joint blocks an entity plays, blended between two frames on the GPU."""

import numpy as np
import pytest

from manifoldx.components import Material, Mesh as MeshC, Transform
from manifoldx.modeling import Mesh
from manifoldx.resources import StandardMaterial
from manifoldx.skinning import JointStore
from tests.test_skinning_blend import _bar, _blend, _engine, _rotation


def _clip():
    return np.stack([
        np.stack([np.eye(4), _rotation([0, 0, 1], 0, np.zeros(3))]),
        np.stack([np.eye(4), _rotation([0, 0, 1], 60, np.zeros(3))]),
    ]).astype(np.float32)  # (2 frames, 2 joints, 4, 4)


def test_bake_returns_consecutive_shared_blocks():
    s = JointStore(8)
    first = s.bake(_clip())
    assert first >= 1
    assert np.allclose(s.mirror[first + 1, :2].transpose(0, 2, 1), _clip()[1])
    assert s.bake(_clip()) == first + 2


def test_bake_at_overwrites_a_clip_of_the_same_length():
    s = JointStore(8)
    first = s.bake(_clip())
    after = s._next
    again = _clip()[::-1].copy()
    assert s.bake(again, at=first) == first and s._next == after  # no new blocks
    assert np.allclose(s.mirror[first, :2].transpose(0, 2, 1), again[0])
    with pytest.raises(ValueError, match="frames"):
        s.bake(_clip()[:1], at=first)
    with pytest.raises(ValueError, match="frames"):
        s.bake(_clip(), at=first + 1)


def test_frames_pack_block_and_fraction():
    s = JointStore(8)
    first = s.bake(_clip())
    s.play([3, 4], first, [0.0, 0.5])
    blocks = s.gpu_blocks(np.array([2, 3, 4]))
    assert blocks[0] == 0  # never posed: the identity block
    assert blocks[1] == first
    assert blocks[2] & 0xFFFFF == first and round((blocks[2] >> 20) / 4095, 3) == 0.5


def test_set_joints_after_play_uses_its_own_block():
    s = JointStore(8)
    first = s.bake(_clip())
    s.play([3], first, [0.25])
    s.set(3, np.tile(np.eye(4, dtype=np.float32), (2, 1, 1)))
    assert s.gpu_blocks(np.array([3]))[0] not in (first, first + 1)
    assert np.allclose(s.mirror[first + 1, :2].transpose(0, 2, 1), _clip()[1])  # the clip is untouched


def test_release_stops_playing():
    s = JointStore(8)
    first = s.bake(_clip())
    s.play([3], first, [0.5])
    s.release([3])
    assert s.gpu_blocks(np.array([3]))[0] == 0


def test_pose_blends_the_two_frames():
    s = JointStore(8)
    first = s.bake(_clip())
    s.play([5], first, [0.5])
    want = 0.5 * _clip()[0] + 0.5 * _clip()[1]
    assert np.allclose(s.pose(5, 2), want, atol=1e-3)
    assert np.allclose(s.current(np.array([5]))[0, :2].transpose(0, 2, 1), want, atol=1e-3)


@pytest.mark.parametrize("position", [-0.5, 1.5])
def test_play_refuses_positions_outside_the_clip(position):
    """A 2-frame clip plays positions 0 to 1: frame floor(p) blends with the next, which must be in the clip."""
    s = JointStore(8)
    first = s.bake(_clip())
    with pytest.raises(ValueError, match="frame"):
        s.play([1], first, [position])


@pytest.mark.parametrize("position", [0.0, 0.5])
def test_gpu_plays_the_blend(position):
    """A played clip renders as set_joints of the same blend of its two frames."""
    bar, clip = _bar(), _clip()
    mat = StandardMaterial(color="#d0a060", roughness=0.6)
    e = _engine()
    h = e.spawn(MeshC(bar.to_geometry()), Material(mat), Transform(pos=(0, 0, 0)))
    first = e.bake_poses(clip)
    e.set_pose_frames([h.index], first, [position])
    played = e.render_frame(supersample=1).astype(np.float64)

    e = _engine()
    posed = Mesh(_blend(bar, ((1 - position) * clip[0] + position * clip[1]).astype(float)).astype(np.float32),
                 bar.faces).recompute_normals()
    e.spawn(MeshC(posed.to_geometry()), Material(mat), Transform(pos=(0, 0, 0)))
    cpu = e.render_frame(supersample=1).astype(np.float64)

    assert np.abs(played - cpu).mean() < 1.0
    assert int((np.abs(played - cpu).max(axis=2) > 16).sum()) <= 8
