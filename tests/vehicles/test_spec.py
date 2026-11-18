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
