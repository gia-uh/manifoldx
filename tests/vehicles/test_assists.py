import numpy as np
import pytest

from manifoldx.vehicles import ASSISTS, FlatGround, Vehicles, apply_assists, strengths_for
from manifoldx.vehicles.body import matrices

from .specs import car


def start(speed, drive="front"):
    fleet, ground = Vehicles([car(drive=drive)]), FlatGround()
    fleet.place(0, (0.0, 0.0, 0.0), 0.0, ground)
    for _ in range(60):
        fleet.step(np.zeros((1, 4)), ground)
    fleet.velocity[0] = (speed, 0.0, 0.0)
    fleet.omega[0] = speed / fleet.p.radius[0]
    if speed > 20:
        fleet.gear[0] = 4
    return fleet, ground


def strengths(**on):
    return np.array([[on.get(a, 0.0) for a in ASSISTS]])


def drive(fleet, ground, c, s, seconds, each=None):
    for _ in range(int(round(seconds / Vehicles.DT))):
        fleet.step(apply_assists(fleet, np.array([c], float), s), ground)
        if each:
            each(fleet)


def test_abs_keeps_the_wheels_turning():
    for k, expect_lock in ((0.0, True), (1.0, False)):
        fleet, ground = start(100 / 3.6)
        locked = []
        drive(fleet, ground, [0.5, 0.0, 1.0, 0.0], strengths(abs=k), 2.0,
              each=lambda f: locked.append(bool(np.any((f.omega[0, :4] == 0) & (f.velocity[0, 0] > 5)))))
        assert any(locked) == expect_lock


def test_traction_control_stops_wheelspin():
    for k, worst in ((0.0, lambda m: m > 0.3), (1.0, lambda m: m <= 0.2)):
        fleet, ground = start(0.0, drive="rear")
        most = [0.0]
        drive(fleet, ground, [0.0, 1.0, 0.0, 0.0], strengths(traction=k), 2.0,
              each=lambda f: most.__setitem__(0, max(most[0], f.slip[0, 2:4, 0].max())))
        assert worst(most[0]), most[0]


def slide_angle(fleet):
    """Degrees between where the car points and where it goes, over the ground."""
    h = matrices(fleet.rotation)[0][:, 0]
    v = fleet.velocity[0]
    return np.degrees(abs(np.arctan2(h[0] * v[2] - h[2] * v[0], h[0] * v[0] + h[2] * v[2])))


def brake_mid_corner(k):
    """Throttle through a right-hander at 80 km/h, then stand on the brake still turning: the tail comes round."""
    fleet, ground = start(80 / 3.6, drive="rear")
    fleet.gear[0] = 3
    worst = [0.0]

    def watch(f):
        if np.linalg.norm(f.velocity[0]) > 3:
            worst[0] = max(worst[0], slide_angle(f))

    s = strengths(steer=k, traction=k, abs=k, stability=k)
    drive(fleet, ground, [0.6, 1.0, 0.0, 0.0], s, 1.0, watch)
    drive(fleet, ground, [0.6, 0.0, 1.0, 0.0], s, 1.0, watch)
    drive(fleet, ground, [0.0, 0.0, 0.0, 0.0], s, 1.0, watch)
    return worst[0]


def test_assists_keep_a_braked_corner_from_spinning():
    assert brake_mid_corner(0.0) > 90
    assert brake_mid_corner(1.0) <= 30


def test_half_strength_corrects_half():
    fleet, ground = start(25.0)
    fleet.angular[0] = (0.0, -0.6, 0.0)  # yawing right hard
    fleet.velocity[0] = (25.0, 0.0, 3.0)  # and sliding
    c = np.array([[0.2, 0.0, 0.0, 0.0]])
    full = apply_assists(fleet, c, strengths(steer=1.0))[0, 0] - 0.2
    half = apply_assists(fleet, c, strengths(steer=0.5))[0, 0] - 0.2
    assert abs(full) > 0.05
    assert half == pytest.approx(full / 2, rel=1e-6)


def test_strengths_from_realism():
    np.testing.assert_allclose(strengths_for(30), [0.7, 0.7, 0.7, 0.7])
    np.testing.assert_allclose(strengths_for(30, locked=("abs",)), [0.7, 0.7, 0.0, 0.7])
    np.testing.assert_allclose(strengths_for(100), [0, 0, 0, 0])


def test_abs_stops_shorter_than_locked_wheels():
    def stop(k):
        fleet, ground = start(100 / 3.6)
        while fleet.velocity[0, 0] > 0.5:
            drive(fleet, ground, [0.0, 0.0, 1.0, 0.0], strengths(abs=k), 1 / 60)
        return fleet.position[0, 0]

    assert stop(1.0) < stop(0.0)


def test_steering_assist_leaves_reverse_alone():
    def yaw_turned(k):
        fleet, ground = start(0.0)
        fleet.gear[0] = -1
        fleet.velocity[0] = (-5.0, 0.0, 0.0)
        fleet.omega[0] = -5.0 / fleet.p.radius[0]
        h0 = matrices(fleet.rotation)[0][:, 0].copy()
        drive(fleet, ground, [1.0, 0.0, 0.3, 0.0], strengths(steer=k, stability=k), 1.5)  # brake = reverse throttle
        h1 = matrices(fleet.rotation)[0][:, 0]
        return np.degrees(abs(np.arctan2(h0[0] * h1[2] - h0[2] * h1[0], h0 @ h1)))

    assert yaw_turned(1.0) >= 0.7 * yaw_turned(0.0)
