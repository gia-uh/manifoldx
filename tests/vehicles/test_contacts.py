import numpy as np

from manifoldx.vehicles import FlatGround, Vehicles
from manifoldx.vehicles.body import matrices

from .specs import car

STILL = np.zeros((1, 4))


def rolling(speed, yaw=0.0, z=0.0):
    fleet, ground = Vehicles([car()]), FlatGround()
    fleet.place(0, (0.0, 0.0, z), yaw, ground)
    fleet.gear[0] = 0
    for _ in range(60):
        fleet.step(STILL, ground)
    heading = matrices(fleet.rotation)[0][:, 0]
    fleet.velocity[0] = heading * speed
    fleet.omega[0] = speed / fleet.p.radius[0]
    return fleet, ground


def wall(x0, z0, x1, z1):
    return np.array([[[x0, z0], [x1, z1]]], float)


def front_x(fleet):
    """How far forward the car's nose reaches along x, from its rotation and length."""
    R = matrices(fleet.rotation)[0]
    half = fleet.p.size[0, 0] / 2
    corners = [fleet.position[0] + R @ (sx * half, 0, sz * fleet.p.size[0, 2] / 2) for sx in (-1, 1) for sz in (-1, 1)]
    return max(c[0] for c in corners)


def test_head_on_stops_without_passing():
    fleet, ground = rolling(50 / 3.6)
    w, deepest, hit = wall(10.0, -5.0, 10.0, 5.0), 0.0, 0.0
    for _ in range(240):
        fleet.step(STILL, ground, w)
        deepest, hit = max(deepest, front_x(fleet) - 10.0), max(hit, fleet.impulse[0])
    assert fleet.velocity[0, 0] < 1.0
    assert deepest <= 0.05
    assert fleet.position[0, 0] < 10.0
    assert hit > 1200 * 50 / 3.6 * 0.8  # the blow is published


def test_glancing_blow_keeps_going_along_the_wall():
    fleet, ground = rolling(50 / 3.6, yaw=-np.radians(15), z=-4.0)  # heading toward +z, the wall at z = 0
    w = wall(-50.0, 0.0, 200.0, 0.0)
    for _ in range(240):
        fleet.step(STILL, ground, w)
    assert np.hypot(fleet.velocity[0, 0], fleet.velocity[0, 2]) > 0.5 * 50 / 3.6
    assert fleet.position[0, 2] < 0.0
    assert fleet.velocity[0, 0] > 5.0


def test_clipping_a_wall_end_spins_the_car():
    fleet, ground = rolling(40 / 3.6, z=0.6)  # the wall's end sits under the car's right front corner
    w = wall(8.0, 1.2, 8.0, 10.0)
    yaw_rates = []
    for _ in range(120):
        fleet.step(STILL, ground, w)
        yaw_rates.append(fleet.angular[0, 1])
    assert max(abs(r) for r in yaw_rates) > 0.5


def test_fast_car_does_not_tunnel_through_a_thin_wall():
    fleet, ground = rolling(160 / 3.6)
    w = wall(20.0, -5.0, 20.0, 5.0)
    for _ in range(120):
        fleet.step(STILL, ground, w)
    assert fleet.position[0, 0] < 20.0


def test_upside_down_rests_on_the_body():
    fleet, ground = Vehicles([car()]), FlatGround()
    fleet.place(0, (0.0, 0.0, 0.0), 0.0, ground)
    fleet.rotation[0] = (1.0, 0.0, 0.0, 0.0)  # half a turn about x: on its roof
    fleet.position[0, 1] = 2.5
    lowest = []
    for _ in range(360):
        fleet.step(STILL, ground)
        R = matrices(fleet.rotation)[0]
        top = fleet.p.mount[0, :, 1].min() + fleet.p.size[0, 1]
        lowest.append(min((fleet.position[0] + R @ (sx * 2.2, top, sz * 0.875))[1] for sx in (-1, 1) for sz in (-1, 1)))
    assert min(lowest[-120:]) > -0.05
    assert np.linalg.norm(fleet.velocity[0]) < 0.05
