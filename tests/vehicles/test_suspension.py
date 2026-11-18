import numpy as np
import pytest

from manifoldx.vehicles import FlatGround, ShapeGround, Tyre, Vehicles

from .specs import A, B, car

G = 9.81
STILL = np.zeros((1, 4))


def run(fleet, ground, seconds, controls=STILL, each=None):
    for _ in range(int(round(seconds / Vehicles.DT))):
        fleet.step(controls, ground)
        if each:
            each(fleet)


def test_settles_with_the_static_loads():
    fleet, ground = Vehicles([car()]), FlatGround()
    fleet.place(0, (0.0, 0.0, 0.0), 0.0, ground)
    fleet.position[0, 1] += 0.5
    run(fleet, ground, 2.0)
    assert abs(fleet.velocity[0, 1]) < 0.02  # settled after one bounce
    run(fleet, ground, 2.0)
    assert abs(fleet.velocity[0, 1]) < 1e-3  # and nothing trembles
    m = 1200 * G
    front, rear = m * B / (A + B) / 2, m * A / (A + B) / 2
    np.testing.assert_allclose(fleet.load[0, :4], [front, front, rear, rear], rtol=0.02)


def energy(fleet, spec):
    v, w = fleet.velocity[0], fleet.angular_body()[0]
    e = 0.5 * spec.mass * v @ v + 0.5 * np.sum(np.array(spec.inertia) * w * w) + spec.mass * G * fleet.position[0, 1]
    k = np.array([wh.spring for wh in spec.wheels])
    return e + np.sum(0.5 * k * fleet.compression[0, :4] ** 2)


def test_no_energy_gain_over_ramps():
    spec = car(cd=0.0, rolling=0.0)
    ground = ShapeGround(profile=[(-100, 0.0), (10, 0.0), (30, -3.0), (50, -3.0), (70, 0.0), (300, 0.0)])
    fleet = Vehicles([spec])
    fleet.place(0, (0.0, 0.0, 0.0), 0.0, ground)
    run(fleet, ground, 1.0)  # settle
    fleet.velocity[0] = (10.0, 0.0, 0.0)
    fleet.omega[0] = 10.0 / fleet.p.radius[0]
    e0, worst = energy(fleet, spec), [0.0]
    run(fleet, ground, 8.0, each=lambda f: worst.__setitem__(0, max(worst[0], energy(f, spec))))
    assert fleet.position[0, 0] > 70  # it went down and back up
    assert worst[0] <= e0 + 0.005 * abs(e0)


def test_climbs_a_kerb_without_taking_off():
    ground = ShapeGround(profile=[(-100, 0.0), (100, 0.0)], kerbs=[5.0], kerb_height=0.15)
    fleet = Vehicles([car()])
    fleet.place(0, (0.0, 0.0, 0.0), 0.0, ground)
    run(fleet, ground, 1.0)
    fleet.velocity[0] = (10 / 3.6, 0.0, 0.0)
    fleet.omega[0] = 10 / 3.6 / fleet.p.radius[0]
    airborne, longest = np.zeros(4, int), [0]

    def watch(f):
        airborne[:] = np.where(f.load[0, :4] <= 0, airborne + 1, 0)
        longest[0] = max(longest[0], airborne.max())

    run(fleet, ground, 3.0, controls=np.array([[0.0, 0.0, 0.0, 0.0]]), each=watch)
    assert fleet.position[0, 0] > 5 + B  # the rear wheels are up too
    assert longest[0] <= 2


def test_stays_under_a_bridge():
    ground = ShapeGround(profile=[(-100, 0.0), (100, 0.0)], bridge=(5.0, 15.0, 6.0))
    fleet = Vehicles([car()])
    fleet.place(0, (0.0, 0.0, 0.0), 0.0, ground)
    run(fleet, ground, 0.5)
    y0 = fleet.position[0, 1]
    fleet.velocity[0] = (5.0, 0.0, 0.0)
    fleet.omega[0] = 5.0 / fleet.p.radius[0]
    run(fleet, ground, 4.0)
    assert fleet.position[0, 0] > 20
    assert fleet.position[0, 1] == pytest.approx(y0, abs=0.05)
