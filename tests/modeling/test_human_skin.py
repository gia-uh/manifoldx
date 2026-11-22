"""The section table lofted into one closed skin whose topology depends only on the level."""

from functools import lru_cache

import numpy as np
import pytest

from manifoldx.modeling.assets.human_sections import load_table
from manifoldx.modeling.assets.human_skin import BRIDGE, HEAD, LEVELS, skin


@lru_cache(maxsize=None)
def _skin(body, level):
    return skin(load_table(body), level)


def _signed_volume(s):
    a, b, c = (s.positions[s.faces[:, i]].astype(float) for i in range(3))
    return np.einsum("ij,ij->i", a, np.cross(b, c)).sum() / 6


@pytest.mark.parametrize("level", sorted(LEVELS))
def test_closed_and_oriented(level):
    s = _skin("female", level)
    edges = np.sort(s.faces[:, [0, 1, 1, 2, 2, 0]].reshape(-1, 2), axis=1)
    _, counts = np.unique(edges, axis=0, return_counts=True)
    assert (counts == 2).all()  # closed, every edge between two faces
    assert (
        0.03 < _signed_volume(s) < 0.09
    )  # outward, about a 1.63 m body's volume in m^3 (less at K = 6)


@pytest.mark.parametrize("level", sorted(LEVELS))
def test_topology_shared_by_bodies(level):
    assert np.array_equal(_skin("female", level).faces, _skin("male", level).faces)


def test_level_budget():
    assert len(_skin("male", 0).faces) < 7000
    assert len(_skin("male", 1).faces) < len(_skin("male", 0).faces) / 3
    assert len(_skin("male", 2).faces) < 900


def test_head_is_not_a_helmet():
    s = _skin("female", 0)
    head = s.positions[s.region == HEAD]
    chin_depth = head[head[:, 1] < np.percentile(head[:, 1], 20), 2].max()
    neck_depth = s.positions[s.ring["torso"][-1], 2].max()
    assert chin_depth > neck_depth + 0.02  # the chin stands in front of the neck


@pytest.mark.parametrize("level", sorted(LEVELS))
def test_on_the_ground(level):
    s = _skin("male", level)
    assert s.positions[:, 1].min() >= -1e-6 and s.positions[:, 1].min() < 0.03


def test_bridges_join_torso_and_limbs():
    s = _skin("female", 0)
    bridge = np.flatnonzero(s.region == BRIDGE)
    assert len(bridge) > 0
    # the crotch midline lies on the body's midplane, below the torso's lowest ring
    low = s.positions[bridge][s.positions[bridge][:, 1] < s.positions[s.ring["torso"][0], 1].min()]
    assert len(low) > 0 and np.abs(low[:, 0]).max() < 1e-6


def test_rings_index_the_chains():
    s = _skin("female", 1)
    assert s.ring["arm_l"].shape == (14, 8) and s.ring["leg_r"].shape == (12, 8)
    assert (s.ring["arm_l"] >= 0).all()
