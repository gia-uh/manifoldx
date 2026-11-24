"""Python reference NumPy ownership: public gathers copy, basic raw slices alias.

Source: ecs.py get_component_data, ComponentAccessor.data, _FieldView.data;
_versions.py _VersionedArray, at rust-core 2acebc0. Raw writable aliases are
measured compatibility, not a promise that every mutation bumps a counter.
"""

import numpy as np
import pytest

from manifoldx.components import Transform
from manifoldx.ecs import ComponentView

from .test_store_contract import _reference_store


def test_public_gathers_copy_but_retained_basic_slices_alias(reference_store):
    store = reference_store
    store.spawn(
        5,
        Transform=Transform(
            pos=[[0, 0, 0], [10, 0, 0], [20, 0, 0], [30, 0, 0], [40, 0, 0]]
        ).get_data(5),
    )
    ids = np.array([4, 0, 2])
    view = ComponentView(store, ["Transform"], ids)
    raw = store._components["Transform"]
    copies = [
        view[Transform].data,
        view[Transform].pos.data,
        view.get_component_data("Transform"),
        raw[ids],
    ]
    for data in copies:
        np.testing.assert_array_equal(data[:, 0], [40, 0, 20])
        assert not np.shares_memory(data, raw)
        data[:] = -99
    np.testing.assert_array_equal(raw[:5, 0], [0, 10, 20, 30, 40])
    retained = store.get_component_data("Transform", slice(0, 5))[:, :3]
    assert np.shares_memory(retained, raw)
    assert not retained.flags.c_contiguous
    retained[1] = [11, 12, 13]
    raw[ids, 0] = [44, 4, 24]  # Fancy assignment scatters, fancy reads copy.
    np.testing.assert_array_equal(
        retained, [[4, 0, 0], [11, 12, 13], [24, 0, 0], [30, 0, 0], [44, 0, 0]]
    )


def test_watched_alias_writes_and_version_observation_are_separate(
    reference_store, record_property
):
    store = reference_store
    store.spawn(2, Transform=Transform().get_data(2))
    store._watch()
    raw = store._components["Transform"]
    retained = raw[:2, :3]
    before = store._version_value
    retained[:] = 1
    assert store._version_value > before  # The intercepted assignment is observable.
    observations = []
    for operation, expected in (
        (lambda: np.add(retained, 2, out=retained), [[3, 3, 3], [3, 3, 3]]),
        (lambda: np.copyto(retained, np.full((2, 3), 7)), [[7, 7, 7], [7, 7, 7]]),
        (lambda: np.asarray(retained).__setitem__((0, 0), 9), [[9, 7, 7], [7, 7, 7]]),
    ):
        before = store._version_value
        operation()
        observations.append(store._version_value - before)
        np.testing.assert_array_equal(raw[:2, :3], expected)
    # Diagnostic only: detection may improve without breaking compatibility.
    record_property("version_deltas_add_copyto_asarray", observations)
    np.testing.assert_array_equal(raw[:2, :3], [[9, 7, 7], [7, 7, 7]])
    assert np.shares_memory(np.asarray(retained), raw)


@pytest.mark.parametrize("kind", ["strided", "reversed", "fortran", "readonly", "float64"])
def test_committed_rows_own_values_from_source_buffers(reference_store, kind):
    store = reference_store
    store.register_component("Input", np.dtype("f4"), (3,))
    if kind == "strided":
        owner = np.array([[1, 99, 2, 99, 3, 99], [4, 99, 5, 99, 6, 99]], dtype="f4")
        source = owner[:, ::2]
        assert not source.flags.c_contiguous
    elif kind == "reversed":
        owner = np.array([[4, 5, 6], [1, 2, 3]], dtype="f4")
        source = owner[::-1]
        assert source.strides[0] < 0
    else:
        owner = np.array(
            [[1, 2, 3], [4, 5, 6]],
            dtype="f8" if kind == "float64" else "f4",
            order="F" if kind == "fortran" else "C",
        )
        source = owner.view()
        if kind == "fortran":
            assert source.flags.f_contiguous and not source.flags.c_contiguous
        if kind == "readonly":
            source.flags.writeable = False
            assert not source.flags.writeable
    ids = store.spawn(2, Input=source, Transform=Transform().get_data(2))
    owner[:] = -10  # Also mutate the backing owner of the read-only view.
    assert not np.shares_memory(source, store._components["Input"])
    assert store._components["Input"].dtype == np.dtype("f4")
    np.testing.assert_array_equal(store.get_component_data("Input", ids), [[1, 2, 3], [4, 5, 6]])
