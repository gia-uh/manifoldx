"""A base body read into a section table: rings along each chain, a head grid, the rest joints."""

from functools import lru_cache

import numpy as np
import pytest

from manifoldx.modeling import skeleton
from manifoldx.modeling.assets.human_sections import (
    _read_chain,
    CHAINS,
    HEAD,
    K,
    load_table,
    read_sections,
    resample,
    save_table,
)

LEG_RADIUS = 0.06
ARM_ANGLE = np.radians(40)


@lru_cache(maxsize=None)
def _capsule_body(stature: float = 1.70, pelvis: float = 0.15, leg_x: float = 0.09, arm_deg: float = 40):
    """An A-pose figure of round cones, 1.70 m tall, facing +Z: legs of constant radius, arms 40 degrees out."""
    d = np.array([np.sin(np.radians(arm_deg)), -np.cos(np.radians(arm_deg)), 0.0])
    j = {"pelvis": (0, 0.88, 0), "chest": (0, 1.28, 0), "neck": (0, 1.45, 0), "head": (0, 1.60, 0)}
    r = {"pelvis": pelvis, "chest": 0.14, "neck": 0.05, "head": 0.10}
    bones = [("pelvis", "chest"), ("chest", "neck"), ("neck", "head")]
    if pelvis > 0.15:  # a wide hip and belly band, as tall as a heavy body's
        j["belly"], r["belly"] = (0, 1.08, 0), pelvis
        bones += [("pelvis", "belly"), ("belly", "chest")]
    for s, side in (("l", 1), ("r", -1)):
        sh = np.array([side * 0.20, 1.34, 0.0])
        dd = d * [side, 1, 1]
        j |= {
            f"hip_{s}": (side * leg_x, 0.84, 0),
            f"knee_{s}": (side * leg_x, 0.48, 0),
            f"ankle_{s}": (side * leg_x, 0.08, 0),
            f"toe_{s}": (side * leg_x, 0.04, 0.17),
            f"shoulder_{s}": tuple(sh),
            f"elbow_{s}": tuple(sh + dd * 0.30),
            f"wrist_{s}": tuple(sh + dd * 0.56),
            f"tip_{s}": tuple(sh + dd * 0.72),
        }
        r |= {
            f"hip_{s}": LEG_RADIUS,
            f"knee_{s}": LEG_RADIUS,
            f"ankle_{s}": LEG_RADIUS,
            f"toe_{s}": 0.04,
            f"shoulder_{s}": 0.045,
            f"elbow_{s}": 0.04,
            f"wrist_{s}": 0.03,
            f"tip_{s}": 0.025,
        }
        bones += [
            ("pelvis", f"hip_{s}"),
            (f"hip_{s}", f"knee_{s}"),
            (f"knee_{s}", f"ankle_{s}"),
            (f"ankle_{s}", f"toe_{s}"),
            ("chest", f"shoulder_{s}"),
            (f"shoulder_{s}", f"elbow_{s}"),
            (f"elbow_{s}", f"wrist_{s}"),
            (f"wrist_{s}", f"tip_{s}"),
        ]
    names = list(j)
    mesh = skeleton.skin(
        [j[n] for n in names],
        [r[n] for n in names],
        [(names.index(a), names.index(b)) for a, b in bones],
        step=0.008,
        blend=0.02,
    )
    pos = mesh.positions.astype(np.float64)
    pos[:, 2] *= -1  # an image-to-mesh model's figure faces -Z; read_sections turns it round
    pos[:, 0] *= -1
    return pos, mesh.faces.astype(np.int64)


def _with_inner_shell_and_debris(pos, faces):
    c = pos.mean(0)
    inner = c + (pos - c) * 0.9
    box = np.array([[x, y, z] for x in (0, 0.02) for y in (0, 0.02) for z in (0, 0.02)], float)
    bf = np.array(
        [
            [0, 1, 3],
            [0, 3, 2],
            [4, 6, 7],
            [4, 7, 5],
            [0, 4, 5],
            [0, 5, 1],
            [2, 3, 7],
            [2, 7, 6],
            [0, 2, 6],
            [0, 6, 4],
            [1, 5, 7],
            [1, 7, 3],
        ]
    )
    parts = [
        (pos, faces),
        (inner, faces[:, ::-1]),
        (box + [0.6, 1.5, 0.0], bf),
        (box + [-0.6, 0.2, 0.3], bf),
    ]
    out_p, out_f, n = [], [], 0
    for p, f in parts:
        out_p.append(p)
        out_f.append(f + n)
        n += len(p)
    return np.concatenate(out_p), np.concatenate(out_f)


@lru_cache(maxsize=None)
def _table():
    return read_sections(*_capsule_body(), stature=1.70)


def test_chain_shapes():
    t = _table()
    assert {n: len(c.t) for n, c in t.chains.items()} == CHAINS
    assert all(c.r.shape == (CHAINS[n], K) for n, c in t.chains.items())
    assert t.head_r.shape == HEAD and len(t.joints) == 19


def test_radii_match_capsules():
    mid = _table().chains["leg_l"].r[12]
    assert abs(mid.mean() - LEG_RADIUS) < 0.004 and mid.std() < 0.003


def test_turned_to_face_plus_z():
    t = _table()
    assert t.joints["toe_l"][2] > t.joints["ankle_l"][2] + 0.04  # toes point to +Z
    assert t.joints["shoulder_l"][0] > 0  # the figure's left is +X


def test_heights_in_order():
    t = _table()
    assert 0.40 * 1.70 < t.crotch_y < t.axilla_y < t.acromion_y < 0.90 * 1.70


def test_reads_outer_shell():
    pos, faces = _with_inner_shell_and_debris(*_capsule_body())
    b = read_sections(pos, faces, stature=1.70)
    assert np.allclose(_table().chains["torso"].r, b.chains["torso"].r, atol=2e-3)


def test_round_trip(tmp_path):
    t = _table()
    save_table(t, tmp_path / "t.npz")
    u = load_table(tmp_path / "t.npz")
    assert np.array_equal(t.chains["arm_l"].r, u.chains["arm_l"].r)
    assert np.array_equal(t.head_r, u.head_r) and u.crotch_y == t.crotch_y
    assert all(np.array_equal(t.joints[n], u.joints[n]) for n in t.joints)


@pytest.mark.parametrize(
    "body", ["female", "male"] + [f"{b}_{v}" for b in ("female", "male") for v in ("heavy", "muscular", "elder")]
)
def test_packaged_tables(body):
    t = load_table(body)
    assert {n: len(c.t) for n, c in t.chains.items()} == CHAINS and t.head_r.shape == HEAD
    assert (t.head_r > 0.01).all() and all((c.r.mean(1) > 0.002).all() for c in t.chains.values())


def test_resample_levels():
    t = load_table("female")
    assert resample(t, 8, 2).chains["torso"].r.shape == (20, 8)
    assert resample(t, 6, 4).chains["leg_l"].r.shape == (6, 6)
    assert resample(t, 16, 1).chains["arm_r"].r.shape == (28, 16)


@pytest.mark.parametrize("k, step, rows", [(16, 1, 5), (8, 2, 3), (6, 4, 3)])
def test_torso_keeps_shoulder_rows(k, step, rows):
    t = resample(load_table("female"), k, step)
    torso = t.chains["torso"]
    y = torso.start[1] + (torso.end[1] - torso.start[1]) * torso.t
    assert ((y >= t.axilla_y - 1e-6) & (y <= t.acromion_y + 1e-6)).sum() == rows


def test_arm_axis_ignores_wide_hips():
    """Hips wider than the shoulders must not pull the arm's axis toward them."""
    t = read_sections(*_capsule_body(pelvis=0.30), stature=1.70)
    for s, side in (("l", 1), ("r", -1)):
        arm = t.chains[f"arm_{s}"]
        d = (arm.end - arm.start) / np.linalg.norm(arm.end - arm.start)
        true = np.array([side * np.sin(ARM_ANGLE), -np.cos(ARM_ANGLE), 0.0])
        assert np.degrees(np.arccos(d @ true)) < 4


def test_chain_stays_on_its_own_limb():
    """From a ring where two legs join, a chain follows the leg its axis runs through, not the nearer loop."""
    j = [(0.07, 1.0, 0), (0.07, 0.0, 0), (-0.09, 1.0, 0), (-0.09, 0.0, 0)]
    mesh = skeleton.skin(j, [0.05, 0.05, 0.08, 0.08], [(0, 1), (2, 3), (0, 2)], step=0.008, blend=0.02)
    pos, faces = mesh.positions.astype(np.float64), mesh.faces.astype(np.int64)
    ch = _read_chain(
        pos, faces, np.array([0.07, 0.99, 0.0]), np.array([0.07, 0.1, 0.0]), np.linspace(0, 1, 12),
        np.array([1.0, 0, 0]),
    )
    assert (ch.centres()[3:, 0] > 0.04).all()


def _landmarks():
    return _table().crotch_y, _table().axilla_y


def test_touching_thighs_read_as_two_legs():
    """With the base body's crotch, each leg keeps its own half of a cut where the thighs touch."""
    t = read_sections(*_capsule_body(leg_x=0.05), stature=1.70, landmarks=_landmarks())
    assert abs(t.crotch_y - _table().crotch_y) < 1e-9
    for s, side in (("l", 1), ("r", -1)):
        leg = t.chains[f"leg_{s}"]
        assert (leg.centres()[:4, 0] * side > 0).all()
        assert leg.r[:4].mean() < 1.4 * LEG_RADIUS


def test_arms_against_the_belly_stay_out_of_the_torso():
    """Below the axilla a torso cut that runs into an arm keeps the torso alone."""
    t = read_sections(*_capsule_body(pelvis=0.30, arm_deg=25), stature=1.70, landmarks=_landmarks())
    torso = t.chains["torso"]
    y = torso.start[1] + (torso.end[1] - torso.start[1]) * torso.t
    near = (y > 1.0) & (y < t.axilla_y - 0.02)
    assert near.any() and torso.r[near].max() < 0.34
