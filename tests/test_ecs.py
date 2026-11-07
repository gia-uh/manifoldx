"""Tests for manifoldx.ecs module."""
import numpy as np
import pytest


def test_entity_store_creation():
    """EntityStore should initialize with max_entities."""
    from manifoldx.ecs import EntityStore
    store = EntityStore(max_entities=1000)
    assert store.max_entities == 1000


def test_register_component():
    """Registering a component should create storage array."""
    from manifoldx.ecs import EntityStore
    store = EntityStore()
    store.register_component("velocity", np.dtype("f4"), shape=(3,))
    assert "velocity" in store._components
    # SoA: each component stored separately
    assert store._components["velocity"].shape == (100_000, 3)


def test_register_multiple_components():
    """Multiple components should each have separate storage."""
    from manifoldx.ecs import EntityStore
    store = EntityStore()
    store.register_component("position", np.dtype("f4"), shape=(3,))
    store.register_component("velocity", np.dtype("f4"), shape=(3,))
    # Check separate arrays per component (SoA)
    assert store._components["position"].shape == (100_000, 3)
    assert store._components["velocity"].shape == (100_000, 3)


def test_spawn_single():
    """Spawning a single entity should return valid index."""
    from manifoldx.ecs import EntityStore
    store = EntityStore()
    store.register_component("position", np.dtype("f4"), shape=(3,))
    indices = store.spawn(n=1, position=[[1, 2, 3]])
    assert len(indices) == 1
    assert store._alive[indices[0]] == True


def test_spawn_multiple():
    """Spawning multiple entities should return all indices."""
    from manifoldx.ecs import EntityStore
    store = EntityStore()
    store.register_component("position", np.dtype("f4"), shape=(3,))
    indices = store.spawn(n=100, position=np.random.rand(100, 3))
    assert len(indices) == 100


def test_spawn_reuses_dead():
    """Spawning after destroy should reuse dead slots."""
    from manifoldx.ecs import EntityStore
    store = EntityStore()
    store.register_component("position", np.dtype("f4"), shape=(3,))
    # Spawn and kill
    idx1 = store.spawn(n=1, position=[[1, 2, 3]])
    store.destroy(idx1)
    # Spawn again - should reuse slot
    idx2 = store.spawn(n=1, position=[[4, 5, 6]])
    assert idx2[0] == idx1[0]


def test_destroy():
    """Destroying entities should mark them as dead."""
    from manifoldx.ecs import EntityStore
    store = EntityStore()
    store.register_component("position", np.dtype("f4"), shape=(3,))
    indices = store.spawn(n=10, position=np.random.rand(10, 3))
    store.destroy(indices[:5])
    assert np.all(store._alive[indices[:5]] == False)
    assert np.all(store._alive[indices[5:]] == True)


def test_query_basic():
    """Query should return entities with matching components."""
    from manifoldx.ecs import EntityStore
    store = EntityStore()
    store.register_component("position", np.dtype("f4"), shape=(3,))
    store.register_component("velocity", np.dtype("f4"), shape=(3,))
    
    # Spawn with both components
    idx = store.spawn(n=5, position=np.zeros((5, 3)), velocity=np.ones((5, 3)))
    
    view = store.get_component_view(['position', 'velocity'])
    assert len(view) == 5


def test_component_view_getitem():
    """ComponentView should allow accessing component data."""
    from manifoldx.ecs import EntityStore
    store = EntityStore()
    store.register_component("position", np.dtype("f4"), shape=(3,))
    indices = store.spawn(n=10, position=np.random.rand(10, 3))
    
    view = store.get_component_view(['position'])
    pos_data = view.get_component_data('position')
    assert pos_data.shape == (10, 3)


# =============================================================================
# _FieldView Operator Tests
# =============================================================================

def _make_field_view(n=5):
    """Helper to create a _FieldView for testing."""
    from manifoldx.ecs import EntityStore, _FieldView
    store = EntityStore(max_entities=100)
    store.register_component("Test", np.dtype("f4"), (3,))
    indices = store.spawn(n)
    # Data: [[0,1,2], [3,4,5], [6,7,8], [9,10,11], [12,13,14]]
    store._components["Test"][indices] = np.arange(n * 3).reshape(n, 3).astype(np.float32)
    return _FieldView(store, indices, "Test", 0, 3, "data", None), store._components["Test"][indices]


def test_fieldview_add_scalar():
    """_FieldView + scalar should return ndarray."""
    fv, data = _make_field_view()
    result = fv + 1
    assert isinstance(result, np.ndarray)
    assert np.allclose(result, data + 1)


def test_fieldview_add_array():
    """_FieldView + array should return ndarray."""
    fv, data = _make_field_view()
    result = fv + np.array([1, 2, 3])
    assert isinstance(result, np.ndarray)
    assert np.allclose(result, data + np.array([1, 2, 3]))


def test_fieldview_add_fieldview():
    """_FieldView + _FieldView should add element-wise."""
    fv, data = _make_field_view()
    result = fv + fv
    assert isinstance(result, np.ndarray)
    assert np.allclose(result, data + data)


def test_fieldview_sub_scalar():
    """_FieldView - scalar should return ndarray."""
    fv, data = _make_field_view()
    result = fv - 1
    assert isinstance(result, np.ndarray)
    assert np.allclose(result, data - 1)


def test_fieldview_sub_array():
    """_FieldView - array should return ndarray."""
    fv, data = _make_field_view()
    result = fv - np.array([1, 2, 3])
    assert isinstance(result, np.ndarray)
    assert np.allclose(result, data - np.array([1, 2, 3]))


def test_fieldview_sub_fieldview():
    """_FieldView - _FieldView should subtract element-wise."""
    fv, data = _make_field_view()
    result = fv - fv
    assert isinstance(result, np.ndarray)
    assert np.allclose(result, data - data)


def test_fieldview_mul_scalar():
    """_FieldView * scalar should return ndarray."""
    fv, data = _make_field_view()
    result = fv * 2
    assert isinstance(result, np.ndarray)
    assert np.allclose(result, data * 2)


def test_fieldview_mul_array():
    """_FieldView * array should multiply element-wise."""
    fv, data = _make_field_view()
    result = fv * np.array([1, 2, 3])
    assert isinstance(result, np.ndarray)
    assert np.allclose(result, data * np.array([1, 2, 3]))


def test_fieldview_mul_fieldview():
    """_FieldView * _FieldView should multiply element-wise."""
    fv, data = _make_field_view()
    result = fv * fv
    assert isinstance(result, np.ndarray)
    assert np.allclose(result, data * data)


def test_fieldview_div_scalar():
    """_FieldView / scalar should return ndarray."""
    fv, data = _make_field_view()
    result = fv / 2
    assert isinstance(result, np.ndarray)
    assert np.allclose(result, data / 2)


def test_fieldview_radd_scalar():
    """scalar + _FieldView should return ndarray."""
    fv, data = _make_field_view()
    result = 1 + fv
    assert isinstance(result, np.ndarray)
    assert np.allclose(result, 1 + data)


def test_fieldview_rsub_scalar():
    """scalar - _FieldView should return ndarray."""
    fv, data = _make_field_view()
    result = 10 - fv
    assert isinstance(result, np.ndarray)
    assert np.allclose(result, 10 - data)


def test_fieldview_rtruediv_scalar():
    """scalar / _FieldView should return ndarray."""
    fv, data = _make_field_view(5)
    # Avoid zeros in data for division
    store = fv._store
    store._components["Test"][fv._indices] = np.arange(1, 16).reshape(5, 3).astype(np.float32)
    data = store._components["Test"][fv._indices]
    result = 10 / fv
    assert isinstance(result, np.ndarray)
    assert np.allclose(result, 10 / data)


def test_fieldview_combined_expression():
    """Combined expressions like np.log(fv + 1) should work."""
    fv, data = _make_field_view()
    result = np.log(fv + 1)
    assert isinstance(result, np.ndarray)
    assert np.allclose(result, np.log(data + 1))


def test_fieldview_expression_order():
    """Expression order should follow standard python precedence."""
    fv, data = _make_field_view()
    result = 3 * fv + 2
    assert isinstance(result, np.ndarray)
    assert np.allclose(result, 3 * data + 2)


def test_spawn_many_is_fast():
    """One large spawn assigns slots in one vectorised step (issue #107)."""
    import time
    from manifoldx.ecs import EntityStore
    store = EntityStore(max_entities=300_000)
    store.register_component("position", np.dtype("f4"), shape=(3,))
    start = time.perf_counter()
    indices = store.spawn(n=300_000, position=np.zeros((300_000, 3)))
    elapsed = time.perf_counter() - start
    np.testing.assert_array_equal(indices, np.arange(300_000))
    assert store._alive.all()
    # The per-entity scan took minutes here; the vectorised path takes milliseconds.
    assert elapsed < 2.0


def test_spawn_reuses_freed_slots_then_fresh_ones():
    """Freed slots come back newest first, then the lowest never-used slots."""
    from manifoldx.ecs import EntityStore
    store = EntityStore(max_entities=20)
    store.register_component("position", np.dtype("f4"), shape=(3,))
    store.spawn(n=10)
    store.destroy(np.array([2, 5, 7]))
    indices = store.spawn(n=5, position=np.arange(15).reshape(5, 3))
    np.testing.assert_array_equal(indices, [7, 5, 2, 10, 11])
    assert store._free_list == []
    np.testing.assert_array_equal(store._components["position"][7], [0, 1, 2])
    np.testing.assert_array_equal(store._components["position"][11], [12, 13, 14])
    assert store._alive.sum() == 12


def test_spawn_partial_reuse_keeps_rest_of_free_list():
    """A spawn smaller than the free list leaves the older freed slots for later."""
    from manifoldx.ecs import EntityStore
    store = EntityStore(max_entities=10)
    store.spawn(n=6)
    store.destroy(np.array([1, 3, 4]))
    np.testing.assert_array_equal(store.spawn(n=2), [4, 3])
    assert store._free_list == [1]
    np.testing.assert_array_equal(store.spawn(n=2), [1, 6])


def test_spawn_over_capacity_spawns_nothing():
    """Running out of slots raises before any slot is taken."""
    from manifoldx.ecs import EntityStore
    store = EntityStore(max_entities=10)
    hooked = []
    store._release_hooks.append(hooked.append)
    store.spawn(n=8)
    store.destroy(np.array([0]))
    hooked.clear()
    with pytest.raises(RuntimeError, match=r"Engine\(max_entities=\.\.\.\)"):
        store.spawn(n=4)
    assert store._alive.sum() == 7
    assert store._free_list == [0]
    assert hooked == []
    np.testing.assert_array_equal(store.spawn(n=3), [0, 8, 9])


def test_high_water_tracks_highest_slot_ever_spawned():
    """Destroying never lowers the mark; reusing freed slots does not raise it."""
    from manifoldx.ecs import EntityStore
    store = EntityStore(max_entities=10)
    assert store._high_water == 0
    store.spawn(n=0)
    assert store._high_water == 0
    store.spawn(n=6)
    assert store._high_water == 6
    store.destroy(np.array([2, 5]))
    assert store._high_water == 6
    store.spawn(n=2)
    assert store._high_water == 6
    store.spawn(n=1)
    assert store._high_water == 7


# A query's membership: only the alive entities that have every named component.

from manifoldx.components import Component, Transform  # noqa: E402
from manifoldx.ecs import EntityStore  # noqa: E402
from manifoldx.types import Vector3  # noqa: E402


def _store():
    store = EntityStore(16)
    Transform.register(store)
    store.register_component("Velocity", np.dtype("f4"), (3,))
    return store


def test_a_query_returns_only_entities_with_all_its_components():
    store = _store()
    store.spawn(2, Transform=Transform.get_default_data(2))
    b = store.spawn(
        3, Transform=Transform.get_default_data(3), Velocity=np.ones((3, 3), np.float32)
    )
    assert store.get_component_view(["Transform", "Velocity"])._indices.tolist() == b.tolist()
    assert len(store.get_component_view(["Transform"])) == 5
    assert len(store.get_component_view([])) == 5


def test_transform_is_a_member_when_not_given():
    store = _store()
    store.spawn(1, Velocity=np.zeros((1, 3), np.float32))
    assert len(store.get_component_view(["Transform"])) == 1


def test_a_refilled_slot_forgets_the_components_it_had():
    store = _store()
    v = store.spawn(
        1, Transform=Transform.get_default_data(1), Velocity=np.ones((1, 3), np.float32)
    )
    store.destroy(v)
    w = store.spawn(1, Transform=Transform.get_default_data(1))
    assert w[0] == v[0]
    assert len(store.get_component_view(["Velocity"])) == 0


def test_an_unregistered_component_matches_nothing():
    store = _store()
    store.spawn(2, Transform=Transform.get_default_data(2))
    assert len(store.get_component_view(["Nope"])) == 0


class _Drift(Component):
    v: Vector3


def test_a_system_skips_entities_without_its_component():
    import manifoldx as mx

    engine = mx.Engine("membership")
    still = engine.spawn(Transform(pos=(0, 0, 0)))
    moving = engine.spawn(Transform(pos=(0, 0, 0)), _Drift(v=(1, 0, 0)))
    engine.spawn(Transform(pos=(0, 0, 0)))

    @engine.system
    def push(q: mx.Query[Transform, _Drift], dt: float):
        q[Transform].pos += np.array([1.0, 0.0, 0.0], np.float32)

    engine.systems.run_all(engine, 1 / 60)
    engine.commands.execute(engine.store)
    assert engine.store._components["Transform"][moving.index, 0] == 1.0
    assert engine.store._components["Transform"][still.index, 0] == 0.0
