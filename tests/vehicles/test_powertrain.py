import numpy as np
import pytest

from manifoldx.vehicles import FlatGround, ShapeGround, Vehicles

from .specs import car

STEER, GAS, BRAKE, HAND = range(4)


def controls(**kw):
    c = np.zeros((1, 4))
    for k, v in kw.items():
        c[0, {"steer": STEER, "gas": GAS, "brake": BRAKE, "hand": HAND}[k]] = v
    return c


def launched(speed=0.0, spec=None, ground=None, gear=None):
    fleet, ground = Vehicles([spec or car(drive="front")]), ground or FlatGround()
    fleet.place(0, (0.0, 0.0, 0.0), 0.0, ground)
    for _ in range(60):
        fleet.step(controls(), ground)
    fleet.velocity[0] = (speed, 0.0, 0.0)
    fleet.omega[0] = speed / fleet.p.radius[0]
    if gear:
        fleet.gear[0] = gear
    return fleet, ground


def drive(fleet, ground, c, seconds, each=None):
    for _ in range(int(round(seconds / Vehicles.DT))):
        fleet.step(c, ground)
        if each:
            each(fleet)


def forward_speed(fleet):
    return float(fleet.velocity[0, 0])


def test_top_speed_balances_power_and_drag():
    v = 50.0
    for _ in range(50):  # 0.42 v³ + rolling·m·g·v = 85 kW
        v -= (0.5 * 1.2 * 0.35 * 2.0 * v**3 + 0.015 * 1200 * 9.81 * v - 85000) / (3 * 0.42 * v**2 + 176.6)
    fleet, ground = launched(0.9 * v, gear=5)
    drive(fleet, ground, controls(gas=1.0), 40.0)
    assert forward_speed(fleet) == pytest.approx(v, rel=0.05)


def test_shifts_up_through_every_gear_in_order():
    fleet, ground = launched()
    seen = []
    drive(fleet, ground, controls(gas=1.0), 30.0,
          each=lambda f: seen.append(int(f.gear[0])) if not seen or seen[-1] != f.gear[0] else None)
    assert seen == [1, 2, 3, 4, 5]


def test_brake_at_rest_backs_up():
    fleet, ground = launched()
    drive(fleet, ground, controls(brake=1.0), 2.0)
    assert fleet.gear[0] == -1
    assert forward_speed(fleet) < -0.5


def test_throttle_while_rolling_back_brakes_first():
    fleet, ground = launched(-2.0, gear=-1)
    speeds = []
    drive(fleet, ground, controls(gas=1.0), 3.0, each=lambda f: speeds.append(forward_speed(f)))
    stopped = next(i for i, v in enumerate(speeds) if v > -0.05)
    assert all(b >= a - 0.03 for a, b in zip(speeds[:stopped], speeds[1:stopped + 1]))  # only slowing, but for the tread's spring letting go
    assert stopped < 120  # stopped within a second
    assert speeds[-1] > 0.5 and fleet.gear[0] == 1  # then drove forward


def test_handbrake_locks_the_rear_only():
    fleet, ground = launched(15.0)
    drive(fleet, ground, controls(hand=1.0), 0.3)
    assert np.all(fleet.omega[0, 2:4] == 0.0)
    assert np.all(fleet.omega[0, :2] > 1.0)


def test_parked_on_a_slope_with_the_handbrake_holds():
    ground = ShapeGround(profile=[(-200, 10.0), (200, -10.0)])  # 5 %
    fleet, ground = launched(ground=ground)
    drive(fleet, ground, controls(hand=1.0), 1.0)
    start = fleet.position[0].copy()
    drive(fleet, ground, controls(hand=1.0), 5.0)
    assert np.hypot(*(fleet.position[0] - start)[[0, 2]]) < 0.01


def test_ackermann_turns_the_inner_wheel_more():
    fleet, ground = launched()
    fleet.step(controls(steer=1.0), ground)
    left, right = fleet.steer_angle[0, 0], fleet.steer_angle[0, 1]  # turning right: the right wheel is inside
    assert left == pytest.approx(fleet.p.max_steer[0])
    assert right > left
    fleet.step(controls(steer=-1.0), ground)
    assert fleet.steer_angle[0, 1] == pytest.approx(-fleet.p.max_steer[0])
    assert fleet.steer_angle[0, 0] < fleet.steer_angle[0, 1]
