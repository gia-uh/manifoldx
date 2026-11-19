import numpy as np
import pytest

from manifoldx.vehicles import Wheel, box_inertia, pack

from .specs import car


def test_pack_shapes_with_mixed_wheel_counts():
    p = pack([car(), car(n_wheels=6)])
    assert p.mass.shape == (2,)
    assert p.mount.shape == (2, 6, 3)
    assert p.wheel_mask.sum(axis=1).tolist() == [4, 6]


def test_brake_shares_must_sum_to_one():
    c = car()
    bad = c.wheels[:3] + (Wheel(**{**c.wheels[3].__dict__, "brake_share": 0.5}),)
    with pytest.raises(ValueError, match="brake_share"):
        pack([type(c)(**{**c.__dict__, "wheels": bad})])


def test_box_inertia_is_a_solid_box():
    ix, iy, iz = box_inertia(12.0, (1.0, 2.0, 3.0))  # length (x), height (y), width (z)
    assert ix == pytest.approx(12 / 12 * (2**2 + 3**2))
    assert iy == pytest.approx(12 / 12 * (1**2 + 3**2))
    assert iz == pytest.approx(12 / 12 * (1**2 + 2**2))


def test_axle_pairs_and_front_flags():
    p = pack([car()])
    assert p.pair[0].tolist() == [1, 0, 3, 2]
    assert p.front[0].tolist() == [True, True, False, False]
    assert np.allclose(p.arb[0], [9000, 9000, 4000, 4000])


def test_car_spec_puts_the_cg_and_the_model_origin_where_asked():
    from manifoldx.vehicles import car_spec

    s = car_spec(length=4.25, width=1.78, height=1.48, wheelbase=2.62, front_overhang=0.86, track=1.53,
                 wheel_radius=0.315, mass=1250, power_kw=85, drive="front", ratios=(3.6, 2.1, 1.4, 1.0, 0.8))
    front = [w for w in s.wheels if w.mount[0] > 0]
    rear = [w for w in s.wheels if w.mount[0] < 0]
    assert len(front) == len(rear) == 2 and all(w.driven for w in front) and not any(w.driven for w in rear)
    assert front[0].mount[0] - rear[0].mount[0] == pytest.approx(2.62)
    assert -s.model_origin[1] == pytest.approx(0.38 * 1.48, rel=0.02)  # the CG's height over the ground at rest
    assert s.model_origin[0] == pytest.approx(-(4.25 / 2 - 0.86 - 0.45 * 2.62), abs=1e-9)
    assert sum(w.brake_share for w in s.wheels) == pytest.approx(1.0)
