"""Idle, walk, run and cross cycles for built figures, and the levels in motion."""

import numpy as np
import pytest

from manifoldx.humanoid.cycles import (
    KEY_PHASES,
    cadence,
    cycle_angles,
    pose_matrices,
    rest_angles,
    root_offset,
    snap_phase,
    update_every,
)
from manifoldx.modeling.assets.human import HumanParams, build_human
from manifoldx.modeling.assets.human_sections import JOINTS


_FIGURES = {}


def _figure(params, level=0):
    key = (repr(params), level)
    if key not in _FIGURES:
        _FIGURES[key] = build_human(params, level=level)
    return _FIGURES[key]


def _world(f, mats, name):
    return (mats[JOINTS.index(name)] @ np.r_[f.joints[name], 1.0])[:3]


def _ankle_world_track(f, kind, speed, frames, fps=60):
    """Both ankles in world space over `frames`, the root advancing speed / fps per frame along +Z."""
    rate = cadence(kind, speed, f.params)
    track = []
    for i in range(frames):
        t = i / fps
        phase = (rate * t) % 1.0
        angles = cycle_angles(kind, phase, speed, f.params)
        mats = pose_matrices(f, angles, root_offset(kind, phase, speed, f.params))
        advance = np.array([0.0, 0.0, speed * t])
        track.append([_world(f, mats, f"ankle_{s}") + advance for s in "lr"])
    return np.array(track)  # (frames, 2, 3)


def _max_slide_in_stance(trace):
    """Largest move of an ankle while it is down (within 5 mm of its lowest height), per stance run."""
    worst = 0.0
    for side in range(trace.shape[1]):
        ankle = trace[:, side]
        down = ankle[:, 1] < ankle[:, 1].min() + 0.005
        start = None
        for i, d in enumerate(np.r_[down, False]):
            if d and start is None:
                start = i
            elif not d and start is not None:
                run = ankle[start:i]
                worst = max(worst, float(np.linalg.norm(run - run[0], axis=1).max()))
                start = None
    return worst


def test_rest_is_identity():
    f = build_human(HumanParams())
    assert np.allclose(pose_matrices(f, rest_angles(f)), np.eye(4), atol=1e-6)


def test_rest_is_identity_for_an_elder():
    f = _figure(HumanParams(age="elder", weight=1.4))
    assert np.allclose(pose_matrices(f, rest_angles(f)), np.eye(4), atol=1e-6)


def test_hip_flex_swings_the_leg_forward():
    """The angle convention of humanoid.solve: zero is the leg straight down, positive hip flex is forward (+Z)."""
    f = _figure(HumanParams())
    angles = rest_angles(f)
    angles["hip_l"] = {"flex": 90.0}
    mats = pose_matrices(f, angles)
    knee = _world(f, mats, "knee_l") - _world(f, mats, "hip_l")
    assert knee[2] > 0.9 * np.linalg.norm(knee)
    angles["hip_l"] = {}
    mats = pose_matrices(f, angles)
    knee = _world(f, mats, "knee_l") - _world(f, mats, "hip_l")
    assert -knee[1] > 0.999 * np.linalg.norm(knee)


@pytest.mark.parametrize("stature", [1.45, 1.70, 1.95])
def test_planted_foot_holds(stature):
    p = HumanParams(stature=stature)
    f = _figure(p)
    trace = _ankle_world_track(f, "walk", speed=1.4, frames=120)  # root advances speed / fps per frame
    assert _max_slide_in_stance(trace) < 0.02


def test_planted_foot_holds_running():
    f = _figure(HumanParams())
    assert _max_slide_in_stance(_ankle_world_track(f, "run", speed=3.5, frames=120)) < 0.02


def test_walking_feet_leave_the_ground():
    trace = _ankle_world_track(_figure(HumanParams()), "walk", speed=1.4, frames=120)
    assert np.ptp(trace[:, 0, 1]) > 0.05


def test_run_is_faster_and_longer():
    p = HumanParams()
    assert cadence("run", 3.5, p) > cadence("walk", 1.4, p)


def test_elder_steps_shorter():
    assert cadence("walk", 1.2, HumanParams(age="elder")) > cadence("walk", 1.2, HumanParams())


def test_cross_turns_the_head():
    yaw = [cycle_angles("cross", ph, 1.0, HumanParams())["neck"].get("twist", 0) for ph in np.linspace(0, 1, 50)]
    assert max(yaw) > 35 and min(yaw) < -35


def test_arms_swing_against_the_legs():
    p = HumanParams()
    a = cycle_angles("walk", 0.0, 1.4, p)
    b = cycle_angles("walk", 0.5, 1.4, p)
    assert (a["shoulder_l"]["flex"] - b["shoulder_l"]["flex"]) * (a["hip_l"]["flex"] - b["hip_l"]["flex"]) < 0


def test_unknown_kind_is_refused():
    with pytest.raises(ValueError, match="kind"):
        cycle_angles("dance", 0.0, 1.0, HumanParams())


def test_levels_in_motion():
    assert [update_every(level) for level in (0, 1, 2)] == [1, 2, 1]
    assert KEY_PHASES == 8
    assert snap_phase(0.3, 0) == 0.3
    assert snap_phase(0.3, 2) == 0.25


def test_far_level_is_rigid():
    m = _figure(HumanParams(), level=2).mesh
    assert np.all(m.weights == 1)


@pytest.mark.parametrize("kind", ["idle", "walk"])
def test_angles_are_finite_at_the_wrap(kind):
    """(phase - 0.5) % 1 rounds to 1.0 just below a half cycle; no foot may divide by zero there."""
    a = cycle_angles(kind, float(np.nextafter(0.5, 0.0)), 1.4, HumanParams())
    assert all(np.isfinite(v) for joint in a.values() for v in joint.values())
