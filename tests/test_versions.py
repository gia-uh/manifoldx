"""Check mode's version counters: writes to the store's memory bump, private copies do not."""

import numpy as np

from manifoldx._versions import _Counter, _VersionedArray


def _store_array():
    counter = _Counter()
    return _VersionedArray(np.zeros((8, 3), np.float32), counter), counter


def test_a_view_write_bumps_the_counter():
    a, counter = _store_array()
    a[:, 0:2][1] = 5.0
    assert counter.value == 1


def test_copies_do_not_carry_the_counter():
    a, counter = _store_array()
    picked = a[np.array([1, 2])]          # fancy indexing copies
    picked[0] = 1.0
    mask = a[:, 0] != 0                    # a ufunc result
    mask[:] = False
    copied = a.copy()
    copied += 1.0
    assert counter.value == 0
