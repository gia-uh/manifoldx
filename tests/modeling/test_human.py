"""HumanParams and build_human: a figure from two base tables and their variant tables."""

from functools import lru_cache

import numpy as np

from manifoldx.modeling.assets.human import HumanParams, build_human
from manifoldx.modeling.assets.human_sections import JOINTS, load_table
from manifoldx.modeling.assets.human_skin import skin


@lru_cache(maxsize=None)
def _build(**kw):
    return build_human(HumanParams(**kw))


def _closed(mesh):
    edges = np.sort(mesh.faces[:, [0, 1, 1, 2, 2, 0]].reshape(-1, 2), axis=1)
    _, counts = np.unique(edges, axis=0, return_counts=True)
    return bool((counts == 2).all())


def _signed_volume(mesh):
    a, b, c = (mesh.positions[mesh.faces[:, i]].astype(float) for i in range(3))
    return np.einsum("ij,ij->i", a, np.cross(b, c)).sum() / 6


def _degenerate_faces(mesh):
    a, b, c = (mesh.positions[mesh.faces[:, i]].astype(float) for i in range(3))
    return int((np.linalg.norm(np.cross(b - a, c - a), axis=1) < 1e-10).sum())


def _perimeter(f, ring):
    p = f.mesh.positions[np.asarray(ring)].astype(float)
    return float(np.linalg.norm(p - np.roll(p, -1, axis=0), axis=1).sum())


def _torso_rings_below_axilla(f):
    axilla = f.joints["chest"][1]
    rings = [r for r in f.rings["torso"] if f.mesh.positions[np.asarray(r), 1].max() < axilla - 0.01]
    return rings[1:]  # the lowest ring is the crotch stitch's


def _girth(f, joint):
    rings = _torso_rings_below_axilla(f)
    y = [f.mesh.positions[np.asarray(r), 1].mean() for r in rings]
    return _perimeter(f, rings[int(np.argmin(np.abs(np.array(y) - f.joints[joint][1])))])


def _max_ring_jump(f):
    worst = 0.0
    # Hands are wider than wrists and feet flare from ankles in the bodies themselves: limbs stop short of them.
    legs = [list(f.rings[n])[2:-4] for n in ("leg_l", "leg_r")]
    arms = [list(f.rings[n])[2 : int(0.7 * len(f.rings[n]))] for n in ("arm_l", "arm_r")]
    chains = [_torso_rings_below_axilla(f)] + legs + arms
    for rings in chains:
        p = np.array([_perimeter(f, r) for r in rings])
        mid = (p[:-2] + p[2:]) / 2
        worst = max(worst, float(np.max(np.abs(p[1:-1] - mid) / mid)))
    return worst


def test_stature():
    for h in (1.45, 1.70, 1.95):
        f = _build(stature=h)
        assert abs(f.mesh.positions[:, 1].max() - h) < 0.01 and f.mesh.positions[:, 1].min() > -1e-4


def test_build_ends_are_the_tables():
    f = _build(build=0.0, stature=1.63, weight=1, muscle=1, shoulders=1, hips=1, leg_ratio=1)
    assert np.allclose(f.mesh.positions, skin(load_table("female")).positions, atol=2e-3)


def test_extreme_corners():
    lo = {k: v["min"] for k, v in HumanParams.ranges().items() if "min" in v and k != "detail"}
    hi = {k: v["max"] for k, v in HumanParams.ranges().items() if "max" in v and k != "detail"}
    n = len(_build().mesh.faces)
    for p in (lo, hi):
        for age in ("adult", "elder"):
            f = build_human(HumanParams(**p, age=age))
            assert len(f.mesh.faces) == n and _closed(f.mesh), (p, age)
            assert _signed_volume(f.mesh) > 0 and _degenerate_faces(f.mesh) == 0, (p, age)


def test_weight_moves_the_waist_most():
    a, b = _build(), _build(weight=1.5)
    assert _girth(b, "spine") / _girth(a, "spine") > _girth(b, "pelvis") / _girth(a, "pelvis") > 1.05


def test_muscle_thickens_the_upper_arm():
    a, b = _build(), _build(muscle=1.3)
    ring = len(a.rings["arm_l"]) // 4
    assert _perimeter(b, b.rings["arm_l"][ring]) > 1.03 * _perimeter(a, a.rings["arm_l"][ring])


def test_elder_leans_forward():
    a, b = _build(), _build(age="elder")
    assert b.joints["head"][2] - b.joints["pelvis"][2] > a.joints["head"][2] - a.joints["pelvis"][2] + 0.02


def test_shoulders_and_hips_widen():
    a = _build()
    s, h = _build(shoulders=1.2), _build(hips=1.25)
    assert s.joints["shoulder_l"][0] > 1.15 * a.joints["shoulder_l"][0]
    assert h.joints["hip_l"][0] > 1.2 * a.joints["hip_l"][0]
    assert np.isclose(s.joints["hip_l"][0], a.joints["hip_l"][0])


def test_leg_ratio_moves_the_crotch():
    a, b = _build(), _build(leg_ratio=1.06)
    assert np.isclose(b.joints["knee_l"][1] / a.joints["knee_l"][1], 1.06, atol=0.01)


def test_no_ledges():  # no ring jumps more than 6 % against its neighbours' mean
    f = _build(weight=1.5, hips=1.25, shoulders=1.2)
    assert _max_ring_jump(f) < 0.06


def test_joints_and_parents():
    f = _build()
    assert list(f.joints) == list(JOINTS) and set(f.parents) == set(JOINTS)
    assert [n for n, p in f.parents.items() if p is None] == ["pelvis"]


def test_skin_colour():
    light, dark = _build(skin="light"), _build(skin="dark")
    assert light.mesh.colors.shape == (len(light.mesh.positions), 3)
    assert light.mesh.colors.mean() > dark.mesh.colors.mean() + 0.2


def test_sample_is_reproducible():
    assert HumanParams.sample(7) == HumanParams.sample(7) != HumanParams.sample(8)
    HumanParams.sample(7).validate()


def test_detail_scales_level_zero():
    assert len(_build(detail=2.0).mesh.faces) > len(_build().mesh.faces)
    assert len(_build(detail=0.5).mesh.faces) < len(_build().mesh.faces)
