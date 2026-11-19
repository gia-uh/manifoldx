"""The plain-Python path for one vehicle (single.py) against the vectorised one, step by step."""

import numpy as np
import pytest

from manifoldx.vehicles import ShapeGround, Vehicles, strengths_for

from .specs import car

STATE = ("position", "velocity", "angular", "rotation", "load", "compression", "ground_y", "omega", "wheel_turn",
         "slip", "steer_angle", "gear", "shift_timer", "rpm", "impulse", "drive_cut", "brake_scale", "extra_brake",
         "lit")
ROAD = ShapeGround([(-50, 0.0), (20, 0.0), (40, 1.5), (60, 0.0), (400, 0.0)], kerbs=[30.0, 70.0],
                   bridge=(90.0, 110.0, 3.0))
WALLS = np.array([[[-50.0, -6.0], [400.0, -6.0]], [[-50.0, 6.0], [400.0, 6.0]], [[150.0, -6.0], [150.0, 6.0]]])


def twins(spec):
    fast, slow = Vehicles([spec]), Vehicles([spec])
    slow.SINGLE = False
    for f in (fast, slow):
        f.place(0, (0.0, 0.0, 0.0), 0.0, ROAD)
    return fast, slow


def drive(fast, slow, controls, seconds, realism):
    s = strengths_for(realism)
    for i in range(int(seconds * 60)):
        c = np.array([controls(i / 60)])
        for f in (fast, slow):
            f.advance(c, ROAD, WALLS, 1 / 60, s)
        assert "_single" in vars(fast) and "_single" not in vars(slow)
        for name in STATE:
            a, b = getattr(fast, name), getattr(slow, name)
            assert a.shape == b.shape, name
            assert np.allclose(a, b, rtol=1e-4, atol=1e-4), f"{name} apart at {i / 60:.2f} s: {a} vs {b}"


def weave(t):
    """Full throttle weaving down the road over the hill, the kerbs and the bridge, then hard on the brakes."""
    if t < 6:
        return [0.2 * np.sin(t * 2.0), 1.0, 0.0, 0.0]
    return [0.3, 0.0, 1.0, 1.0 if t > 7 else 0.0]


@pytest.mark.parametrize("drive_side", ["rear", "front", "all"])
@pytest.mark.parametrize("realism", [0, 100])
def test_one_vehicle_in_plain_python_drives_as_the_vectorised_fleet(drive_side, realism):
    fast, slow = twins(car(drive=drive_side))
    for f in (fast, slow):
        f.velocity[0] = (15.0, 0.0, 0.0)
        f.omega[0] = 15.0 / f.p.radius[0]
        f.pick_gear(0)
    drive(fast, slow, weave, 8.0, realism)
    assert fast.position[0, 0] > 50  # it went over the hill and the kerbs


def test_one_vehicle_in_plain_python_hits_a_wall_and_backs_up_as_the_vectorised_fleet():
    fast, slow = twins(car())
    for f in (fast, slow):
        f.velocity[0] = (25.0, 0.0, 0.0)
        f.omega[0] = 25.0 / f.p.radius[0]
        f.position[0, 0] = 120.0
    drive(fast, slow, lambda t: [0.8, 1.0, 0.0, 0.0] if t < 2.5 else [0.0, 0.0, 1.0, 0.0], 5.0, 50)
    assert fast.impulse.max() >= 0 and fast.position[0, 0] < 150


def test_one_vehicle_in_plain_python_rolls_over_as_the_vectorised_fleet():
    fast, slow = twins(car())
    for f in (fast, slow):
        f.position[0, 1] += 1.0
        f.angular[0] = (6.0, 0.0, 0.0)  # a roll that lands it on its side
    drive(fast, slow, lambda t: [0.0, 0.0, 0.0, 0.0], 3.0, 100)
