"""StandardMaterial.get_data keeps its row until a field is written (#311)."""

import numpy as np

from manifoldx.resources import StandardMaterial


def test_an_unchanged_material_gives_the_same_row():
    m = StandardMaterial("#ff8800", roughness=0.3, vertex_colors=True)
    np.testing.assert_array_equal(m.get_data(1, None), m.get_data(1, None))


def test_writing_a_field_changes_the_next_row():
    m = StandardMaterial("#ff8800", vertex_colors=True)
    before = m.get_data(1, None).copy()
    for name, value in (("color", "#0044ff"), ("roughness", 0.9), ("opacity", 0.5), ("cut", (0, 0, 1, 1))):
        setattr(m, name, value)
        after = m.get_data(1, None)
        assert not np.array_equal(before, after), name
        before = after.copy()


def test_spawning_n_entities_gets_its_own_array():
    m = StandardMaterial("#ff8800")
    one = m.get_data(1, None)
    many = m.get_data(3, None)
    assert many.shape[0] == 3 and many is not one
    many[:] = 0
    assert m.get_data(1, None).any()
