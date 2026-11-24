"""Phase-0 Python reference characterization, not native ECS parity.

Source: ecs.py EntityStore; components.py packing, at rust-core 2acebc0.
Fixed capacity and slot reuse are compatibility decisions. Invalid writes and
stale handles are deliberately outside these contracts; stronger atomicity and
generation policies need separate decisions before a native implementation.
"""

import numpy as np
import pytest

from manifoldx.components import Component, Material, Mesh, Transform
from manifoldx.ecs import EntityStore
from manifoldx.types import Float, Vector3


@pytest.fixture(name="reference_store")
def _reference_store():
    """The real Python implementation, with no backend parameter or skip."""
    store = EntityStore(max_entities=20)
    assert type(store) is EntityStore
    assert EntityStore.__module__ == "manifoldx.ecs"
    Transform.register(store)
    return store


def test_packed_layouts_and_manual_multidimensional_rows(reference_store):
    store = reference_store

    class ContractMotion(Component):
        velocity: Vector3 = (1, 2, 3)
        mass: Float = 4

    class ContractMarker(Component):
        pass

    for component in (Mesh, Material, ContractMotion, ContractMarker):
        component.register(store)
    store.register_component("Tile", np.dtype("i2"), (2, 2))
    ids = store.spawn(
        2,
        Transform=Transform().get_data(2),
        Mesh=np.array([[11], [12]], dtype="u4"),
        Material=np.array([[21], [22]], dtype="u4"),
        ContractMotion=ContractMotion(mass=[5, 6]).get_data(2),
        ContractMarker=ContractMarker().get_data(2),
        Tile=np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype="i2"),
    )
    np.testing.assert_array_equal(ids, [0, 1])
    assert ids.dtype == np.dtype("i4")
    for name, dtype, shape in (
        ("Transform", "f4", (20, 10)),
        ("Mesh", "u4", (20, 1)),
        ("Material", "u4", (20, 1)),
        ("ContractMotion", "f4", (20, 4)),
        ("ContractMarker", "f4", (20, 0)),
        ("Tile", "i2", (20, 2, 2)),
    ):
        assert store._components[name].shape == shape
        assert store._components[name].dtype == np.dtype(dtype)
    np.testing.assert_array_equal(
        store.get_component_data("Transform", ids),
        [[0, 0, 0, 0, 0, 0, 1, 1, 1, 1], [0, 0, 0, 0, 0, 0, 1, 1, 1, 1]],
    )
    assert ContractMotion._layout == {"velocity": (0, 3), "mass": (3, 1)}
    np.testing.assert_array_equal(ContractMotion().get_data(1), [[1, 2, 3, 4]])
    np.testing.assert_array_equal(
        store.get_component_data("ContractMotion", ids), [[1, 2, 3, 5], [1, 2, 3, 6]]
    )
    np.testing.assert_array_equal(store.get_component_data("Mesh", ids), [[11], [12]])
    np.testing.assert_array_equal(store.get_component_data("Material", ids), [[21], [22]])
    np.testing.assert_array_equal(
        store.get_component_data("Tile", ids), [[[1, 2], [3, 4]], [[5, 6], [7, 8]]]
    )
    assert list(store.get_component_view(["ContractMarker"])) == [0, 1]


def test_reuse_order_rows_and_high_water(reference_store):
    store = reference_store
    store.spawn(10, Transform=Transform().get_data(10))
    store.destroy(np.array([2, 5, 7]))
    ids = store.spawn(
        5,
        Transform=Transform(
            pos=[[70, 0, 0], [50, 0, 0], [20, 0, 0], [100, 0, 0], [110, 0, 0]]
        ).get_data(5),
    )
    np.testing.assert_array_equal(ids, [7, 5, 2, 10, 11])
    np.testing.assert_array_equal(
        store._components["Transform"][:12, 0], [0, 0, 20, 0, 0, 50, 0, 70, 0, 0, 100, 110]
    )
    np.testing.assert_array_equal(store._alive, [True] * 12 + [False] * 8)
    assert store._alive.sum() == 12
    assert store._high_water == 12
    store.destroy(np.array([11]))
    assert store._high_water == 12


def test_overflow_does_not_consume_slots_or_fire_hooks():
    store = EntityStore(max_entities=10)
    Transform.register(store)
    store.spawn(8, Transform=Transform().get_data(8))
    store.destroy(np.array([0]))
    calls = []
    store._release_hooks.append(lambda ids: calls.append(ids.copy()))
    with pytest.raises(RuntimeError, match="Max entities 10"):
        store.spawn(4, Transform=Transform().get_data(4))
    assert calls == []
    assert store.max_entities == 10
    assert store._components["Transform"].shape == (10, 10)
    assert store._high_water == 8
    np.testing.assert_array_equal(store._reserved, [False] * 10)
    np.testing.assert_array_equal(store._alive, [False] + [True] * 7 + [False] * 2)
    ids = store.spawn(3, Transform=Transform().get_data(3))
    np.testing.assert_array_equal(ids, [0, 8, 9])
    assert len(calls) == 1
    np.testing.assert_array_equal(calls[0], [0, 8, 9])
    assert store._high_water == 10


def test_query_intersection_and_refill_membership(reference_store):
    store = reference_store
    store.register_component("Velocity", np.dtype("f4"), (3,))
    store.spawn(2, Transform=Transform().get_data(2))
    store.spawn(3, Transform=Transform().get_data(3), Velocity=np.ones((3, 3)))
    assert list(store.get_component_view([])) == [0, 1, 2, 3, 4]
    assert list(store.get_component_view(["Transform", "Velocity"])) == [2, 3, 4]
    assert list(store.get_component_view(["Unknown"])) == []
    assert list(store.get_component_view(["Transform", "Unknown"])) == []
    store.destroy(np.array([3]))
    np.testing.assert_array_equal(store.spawn(1, Transform=Transform().get_data(1)), [3])
    assert list(store.get_component_view(["Transform"])) == [0, 1, 2, 3, 4]
    assert list(store.get_component_view(["Velocity"])) == [2, 4]
    np.testing.assert_array_equal(store._members["Velocity"][:5], [False, False, True, False, True])


def test_reservations_are_disjoint_and_invisible_until_applied(reference_store):
    store = reference_store
    first, second = store.reserve(2), store.reserve(3)
    np.testing.assert_array_equal(first, [0, 1])
    np.testing.assert_array_equal(second, [2, 3, 4])
    assert list(store.get_component_view([])) == []
    assert list(store.get_component_view(["Transform"])) == []
    assert store._high_water == 0
    store.write_reserved(second, Transform=Transform(pos=(2, 3, 4)).get_data(3))
    assert list(store.get_component_view(["Transform"])) == [2, 3, 4]
    np.testing.assert_array_equal(store._reserved[:5], [True, True, False, False, False])
    store.write_reserved(first, Transform=Transform(pos=(5, 6, 7)).get_data(2))
    assert list(store.get_component_view([])) == [0, 1, 2, 3, 4]
    np.testing.assert_array_equal(
        store._components["Transform"][:5, :3],
        [[5, 6, 7], [5, 6, 7], [2, 3, 4], [2, 3, 4], [2, 3, 4]],
    )
    assert store._high_water == 5
