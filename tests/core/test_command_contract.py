"""Python reference enqueue-time capture, not stronger transaction semantics.

Source: ecs.py _FieldView._set_data/__iadd__, ComponentAccessor._queue_update;
commands.py CommandBuffer.execute, at rust-core 2acebc0. Repeated deferred
arithmetic reads committed data (existing issue #218); it does not accumulate
earlier queued writes. Quaternion += left-multiplies the committed rotation.
"""

from types import SimpleNamespace

import numpy as np
import pytest

from manifoldx.commands import CommandBuffer
from manifoldx.components import Transform
from manifoldx.ecs import ComponentView

from .test_store_contract import _reference_store


@pytest.mark.parametrize("whole_component", [False, True])
def test_enqueued_indices_and_values_survive_caller_mutation(reference_store, whole_component):
    store = reference_store
    store.spawn(5, Transform=Transform().get_data(5))
    ids = np.array([4, 0, 2])
    engine = SimpleNamespace(commands=CommandBuffer())
    accessor = ComponentView(store, ["Transform"], ids, engine)[Transform]
    values = np.array([[40, 41, 42], [10, 11, 12], [20, 21, 22]], dtype="f4")
    if whole_component:
        # Whole-component arithmetic also goes through a copied command payload.
        values = np.array([[40] * 10, [10] * 10, [20] * 10], dtype="f4")
        accessor += values
    else:
        accessor.pos = values
    ids[:] = 1
    values[:] = -99
    np.testing.assert_array_equal(store._components["Transform"][:5, :3], np.zeros((5, 3)))
    assert len(engine.commands) == 1
    engine.commands.execute(store)
    expected = (
        [[10, 10, 10], [0, 0, 0], [20, 20, 20], [0, 0, 0], [40, 40, 40]]
        if whole_component
        else [[10, 11, 12], [0, 0, 0], [20, 21, 22], [0, 0, 0], [40, 41, 42]]
    )
    np.testing.assert_array_equal(store._components["Transform"][:5, :3], expected)


@pytest.mark.parametrize("deferred,expected", [(False, 2), (True, 1)])
def test_repeated_arithmetic_reads_committed_rows(reference_store, deferred, expected):
    store = reference_store
    store.spawn(1, Transform=Transform().get_data(1))
    engine = SimpleNamespace(commands=CommandBuffer()) if deferred else None
    accessor = store.get_component_view(["Transform"], engine)[Transform]
    accessor.pos += 1
    accessor.pos += 1
    if deferred:
        np.testing.assert_array_equal(accessor.pos.data, [[0, 0, 0]])
        assert len(engine.commands) == 2
        engine.commands.execute(store)
    np.testing.assert_array_equal(accessor.pos.data, [[expected] * 3])
    np.testing.assert_array_equal(accessor.data[:, 3:], [[0, 0, 0, 1, 1, 1, 1]])


@pytest.mark.parametrize("deferred", [False, True])
def test_quaternion_composition_order_and_committed_capture(reference_store, deferred):
    store = reference_store
    # Start at quarter-X, then apply quarter-Z twice. z*x != x*z.
    half_sqrt = 0.7071067811865476
    store.spawn(1, Transform=Transform(rot=(half_sqrt, 0, 0, half_sqrt)).get_data(1))
    engine = SimpleNamespace(commands=CommandBuffer()) if deferred else None
    accessor = store.get_component_view(["Transform"], engine)[Transform]
    accessor.rot += [0, 0, half_sqrt, half_sqrt]
    accessor.rot += [0, 0, half_sqrt, half_sqrt]
    if deferred:
        engine.commands.execute(store)
    expected = [[0.5, 0.5, 0.5, 0.5]] if deferred else [[0, half_sqrt, half_sqrt, 0]]
    np.testing.assert_allclose(accessor.rot.data, expected, atol=1e-6, rtol=0)
    np.testing.assert_array_equal(accessor.pos.data, [[0, 0, 0]])
    np.testing.assert_array_equal(accessor.scale.data, [[1, 1, 1]])
