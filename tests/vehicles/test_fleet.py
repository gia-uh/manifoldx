import numpy as np
import pytest

from manifoldx.vehicles import FlatGround, ShapeGround, Vehicles, strengths_for

from .specs import car

GAS_RIGHT = np.array([[0.3, 0.8, 0.0, 0.0]])


def fresh(ground=None):
    fleet, ground = Vehicles([car()]), ground or FlatGround()
    fleet.place(0, (0.0, 0.0, 0.0), 0.0, ground)
    return fleet, ground


def test_same_controls_same_path_at_any_frame_rate():
    ends = []
    for frame in (1 / 30, 1 / 60, 1 / 144):
        fleet, ground = fresh()
        steps, at = 0, {}
        while steps < 1200:
            steps += fleet.advance(GAS_RIGHT, ground, None, frame, strengths_for(30))
            at[steps] = fleet.position[0].copy()
        ends.append(at)
    common = sorted(set(ends[0]) & set(ends[1]) & set(ends[2]))
    assert common and common[-1] >= 1100
    for k in common[-3:]:
        assert np.array_equal(ends[0][k], ends[1][k]) and np.array_equal(ends[0][k], ends[2][k])


def test_a_long_frame_runs_six_steps_and_still_stops_at_a_wall():
    fleet, ground = fresh()
    fleet.gear[0] = 0
    nose = fleet.p.size[0, 0] / 2
    fleet.velocity[0] = (30.0, 0.0, 0.0)
    fleet.omega[0] = 30.0 / fleet.p.radius[0]
    wall = np.array([[[nose + 0.3, -5.0], [nose + 0.3, 5.0]]])
    assert fleet.advance(np.zeros((1, 4)), ground, wall, 0.2) == 6
    assert fleet.position[0, 0] + nose <= nose + 0.3 + 0.05


@pytest.mark.parametrize("y, expect", [(6.0, 6.0), (0.0, 0.0)])
def test_place_on_or_under_a_bridge(y, expect):
    ground = ShapeGround(profile=[(-100, 0.0), (100, 0.0)], bridge=(5.0, 15.0, 6.0))
    fleet = Vehicles([car()])
    fleet.place(0, (10.0, y, 0.0), 0.0, ground)
    for _ in range(120):
        fleet.step(np.zeros((1, 4)), ground)
    assert fleet.ground_y[0, :4] == pytest.approx(np.full(4, expect))
    assert np.linalg.norm(fleet.velocity[0]) < 0.05


def test_pose_interpolates_between_steps():
    fleet, ground = fresh()
    for _ in range(30):
        fleet.advance(GAS_RIGHT, ground, None, 1 / 60)
    before = fleet.position[0].copy()
    fleet.advance(GAS_RIGHT, ground, None, Vehicles.DT * 1.5)
    after = fleet.position[0].copy()
    mid, _ = fleet.pose()
    assert np.all(np.minimum(before, after) - 1e-9 <= mid[0]) and np.all(mid[0] <= np.maximum(before, after) + 1e-9)
    assert not np.allclose(mid[0], after)


def test_telemetry_reports_the_car():
    fleet, ground = fresh()
    for _ in range(120):
        fleet.advance(GAS_RIGHT, ground, None, 1 / 60, strengths_for(0))
    t = fleet.telemetry(0)
    assert t["speed"] > 1.0 and t["gear"] >= 1 and t["rpm"] > 800
    assert len(t["load"]) == 4 and len(t["slip"]) == 4 and len(t["travel"]) == 4
    assert set(t["lit"]) == {"steer", "traction", "abs", "stability"}


def test_the_drawn_model_sits_on_the_ground_at_rest():
    from manifoldx.vehicles import car_spec

    spec = car_spec(length=4.25, width=1.78, height=1.48, wheelbase=2.62, front_overhang=0.86, track=1.53,
                    wheel_radius=0.315, mass=1250, power_kw=85)
    fleet, ground = Vehicles([spec]), FlatGround()
    fleet.place(0, (0.0, 0.0, 0.0), 0.4, ground)
    for _ in range(240):
        fleet.step(np.zeros((1, 4)), ground)
    at, rot = fleet.model_transform(0)
    assert at[1] == pytest.approx(0.0, abs=0.02)
    assert np.allclose(fleet.travel(0), 0.0, atol=0.01)


@pytest.mark.parametrize("builder", ["bus", "truck"])
def test_buses_and_trucks_lift_a_wheel_by_its_travel(builder):
    if builder == "bus":
        from manifoldx.modeling.assets.bus import BusParams, build_bus as build
        model = build(BusParams())
    else:
        from manifoldx.modeling.assets.truck import TruckParams, build_truck as build
        model = build(TruckParams())
    wheels = [p.name for p in model.parts if p.kind == "wheel"]
    rest, lifted = model.pose(), model.pose(travel=[0.05] + [0.0] * (len(wheels) - 1))
    assert lifted[wheels[0]][0][1] == pytest.approx(rest[wheels[0]][0][1] + 0.05)
    assert np.allclose(lifted[wheels[1]][0], rest[wheels[1]][0])


class Counting(FlatGround):
    def __init__(self):
        super().__init__()
        self.calls = 0

    def height(self, x, z, near=None):
        self.calls += 1
        return super().height(x, z, near)


def test_an_upright_car_reads_the_ground_once_per_step():
    ground = Counting()
    fleet = Vehicles([car()])
    fleet.place(0, (0.0, 0.0, 0.0), 0.0, ground)
    for _ in range(60):
        fleet.step(np.zeros((1, 4)), ground)
    ground.calls = 0
    for _ in range(10):
        fleet.step(np.array([[0.2, 0.5, 0.0, 0.0]]), ground)
    assert ground.calls == 10


@pytest.mark.parametrize("single", [True, False])
def test_far_walls_do_not_reach_the_contact_test(monkeypatch, single):
    from manifoldx.vehicles import contacts
    from manifoldx.vehicles import single as one

    seen = []
    module, name = (one, "_footprint_hit") if single else (contacts, "footprint_hit")
    real = getattr(module, name)
    monkeypatch.setattr(module, name, lambda w, *a: seen.append(len(np.reshape(w, (-1, 2, 2)))) or real(w, *a))
    fleet, ground = fresh()
    fleet.SINGLE = single
    far = np.array([[[100.0 + i, -5.0], [100.0 + i, 5.0]] for i in range(2000)])
    near = np.array([[[2.6, -5.0], [2.6, 5.0]]])  # 0.4 m before the nose
    for _ in range(5):
        fleet.step(np.zeros((1, 4)), ground, np.concatenate([far, near]))
    assert seen and max(seen) <= 10


def two_cars(offset=0.0, speed=10.0):
    """A car rolling at speed along +x toward a parked car 3 m ahead of its nose, offset metres to its side."""
    ground = FlatGround()
    fleet = Vehicles([car(), car()])
    fleet.place(0, (0.0, 0.0, 0.0), 0.0, ground)
    fleet.place(1, (fleet.p.size[0, 0] + 3.0, 0.0, offset), 0.0, ground)
    for _ in range(60):
        fleet.step(np.zeros((2, 4)), ground)
    fleet.velocity[0] = (speed, 0.0, 0.0)
    fleet.omega[0] = speed / fleet.p.radius[0]
    return fleet, ground


def test_a_car_hitting_a_parked_one_passes_it_momentum():
    fleet, ground = two_cars()
    before = fleet.p.mass @ fleet.velocity[:, 0]
    hit = 0
    for _ in range(240):  # until a few steps after the first contact
        fleet.step(np.zeros((2, 4)), ground)
        hit += hit > 0 or fleet.impulse.max() > 0
        if hit >= 12:
            break
    after = fleet.p.mass @ fleet.velocity[:, 0]
    assert fleet.velocity[1, 0] > 2.0 and fleet.velocity[0, 0] < 8.0
    assert after == pytest.approx(before, rel=0.1)  # the tyres take a little in those few steps
    gap = fleet.position[1, 0] - fleet.position[0, 0] - fleet.p.size[0, 0]
    assert gap > -0.05  # pushed apart, not through each other


def test_an_off_centre_hit_spins_the_parked_car():
    fleet, ground = two_cars(offset=1.3, speed=12.0)
    for _ in range(90):
        fleet.step(np.zeros((2, 4)), ground)
    assert abs(fleet.angular[1, 1]) > 0.3  # rad/s about the vertical


def test_cars_apart_never_touch():
    fleet, ground = two_cars(speed=0.0)
    for _ in range(120):
        fleet.step(np.zeros((2, 4)), ground)
    assert fleet.impulse.max() == 0.0 and np.linalg.norm(fleet.velocity[1]) < 0.05


def test_add_and_remove_keep_the_other_vehicles_state():
    fleet, ground = fresh()
    fleet.velocity[0] = (5.0, 0.0, 0.0)
    kept = {k: np.array(getattr(fleet, k)[0]) for k in Vehicles.STATE}
    i = fleet.add(car(n_wheels=6))  # more wheels: every row widens
    assert i == 1 and len(fleet.position) == 2
    fleet.place(1, (20.0, 0.0, 0.0), 0.0, ground)
    for k, v in kept.items():
        assert np.array_equal(getattr(fleet, k)[0][: v.shape[0]] if v.ndim else getattr(fleet, k)[0], v), k
    fleet.remove(0)
    assert len(fleet.position) == 1 and fleet.position[0, 0] == pytest.approx(20.0)
    for _ in range(60):
        fleet.step(np.zeros((1, 4)), ground)
    assert np.hypot(fleet.velocity[0, 0], fleet.velocity[0, 2]) < 0.05  # it settles on its springs, standing still
