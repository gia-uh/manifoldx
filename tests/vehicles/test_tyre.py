import numpy as np
import pytest

from manifoldx.vehicles import FlatGround, ShapeGround, Tyre, Vehicles, pack
from manifoldx.vehicles.tyre import forces, magic

from .specs import car

T = Tyre()
S = np.linspace(0, 1.5, 15001)


@pytest.mark.parametrize("b, c, e, lo, hi", [(T.b_long, T.c_long, T.e_long, 0.08, 0.15),
                                             (T.b_lat, T.c_lat, T.e_lat, 0.12, 0.20)])
def test_default_tyre_peaks_then_falls(b, c, e, lo, hi):
    y = magic(S, b, c, e)
    assert lo <= S[y.argmax()] <= hi
    assert 0.75 <= magic(np.array(1.0), b, c, e) / y.max() <= 0.85


def tyre_arrays(n=1):
    p = pack([car()])
    return p.tyre[0, :n], p.kappa_peak[0, :n], p.alpha_peak[0, :n]


def test_pure_slip_follows_each_curve():
    tyre, kp, ap = tyre_arrays()
    fx, fy = forces(np.array([0.05]), np.array([0.0]), np.array([3000.0]), np.array([1.0]), tyre, kp, ap)
    assert fy[0] == pytest.approx(0.0)
    assert fx[0] == pytest.approx(3000 * magic(np.array(0.05), T.b_long, T.c_long, T.e_long), rel=1e-6)


def test_combined_slip_stays_inside_mu_load_and_follows_the_slip():
    tyre, kp, ap = tyre_arrays()
    k, a = np.array([0.5]), np.array([0.3])
    fx, fy = forces(k, a, np.array([3000.0]), np.array([0.9]), tyre, kp, ap)
    assert np.hypot(fx, fy)[0] <= 0.9 * 3000 + 1e-6
    assert fx[0] > 0 and fy[0] < 0  # drives forward, pushes against a slide to the right
    assert abs(fx[0] / fy[0]) == pytest.approx((0.5 / kp[0]) / (0.3 / ap[0]), rel=0.35)


def run(fleet, ground, seconds):
    for _ in range(int(round(seconds / Vehicles.DT))):
        fleet.step(np.zeros((1, 4)), ground)


def test_parked_on_the_flat_does_not_creep():
    fleet, ground = Vehicles([car()]), FlatGround()
    fleet.place(0, (0.0, 0.0, 0.0), 0.3, ground)
    run(fleet, ground, 2.0)
    start = fleet.position[0].copy()
    run(fleet, ground, 5.0)
    assert np.hypot(*(fleet.position[0] - start)[[0, 2]]) < 1e-3


def test_rolls_down_a_slope_without_brakes():
    ground = ShapeGround(profile=[(-200, 10.0), (200, -10.0)])  # 5 % down toward +x
    fleet = Vehicles([car()])
    fleet.place(0, (0.0, 0.0, 0.0), 0.0, ground)
    run(fleet, ground, 3.0)
    assert fleet.velocity[0, 0] > 0.5


def test_coasts_straight():
    fleet, ground = Vehicles([car()]), FlatGround()
    fleet.place(0, (0.0, 0.0, 0.0), 0.0, ground)
    run(fleet, ground, 1.0)
    fleet.velocity[0] = (20.0, 0.0, 0.0)
    fleet.omega[0] = 20.0 / fleet.p.radius[0]
    while fleet.position[0, 0] < 50:
        run(fleet, ground, 0.1)
    assert abs(fleet.position[0, 2]) < 0.05


def test_a_sideways_slide_stops():
    fleet, ground = Vehicles([car()]), FlatGround()
    fleet.place(0, (0.0, 0.0, 0.0), 0.0, ground)
    run(fleet, ground, 1.0)
    fleet.velocity[0] = (0.0, 0.0, 3.0)
    run(fleet, ground, 2.0)
    assert abs(fleet.velocity[0, 2]) < 0.05
    assert fleet.position[0, 2] < 3.0
