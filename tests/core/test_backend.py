"""Fast shared-backend parity for the private native allocator slice.

The same three cases run against the installed Python ``EntityStore`` and the
compiled production ``_EntitySlots``. This is not the full ECS: no components,
queries or Engine activation. The module is marked ``slow`` so the ordinary
Python-only suite does not collect it; ``make parity-fast`` overrides addopts
and runs it against a selected installed interpreter. A missing extension
fails, and a Python module cannot stand in for it: no skip and no fallback.
"""

from __future__ import annotations

import importlib.machinery
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.slow


def _installed(path):
    return Path(path).resolve().is_relative_to(Path(sys.prefix).resolve())


class PythonBackend:
    def __init__(self, capacity):
        from manifoldx.ecs import EntityStore

        self.store = EntityStore(max_entities=capacity)

    def spawn(self, n):
        return [int(index) for index in self.store.spawn(n)]

    def destroy(self, ids):
        self.store.destroy(np.asarray(ids, dtype=np.int32))

    def reserve(self, n):
        return np.asarray(self.store.reserve(n), dtype=np.int32)

    def ids(self, reservation):
        return [int(index) for index in reservation]

    def commit(self, reservation):
        self.store.write_reserved(reservation)
        return self.ids(reservation)

    def visible(self):
        return [int(index) for index in self.store.get_component_view([])]

    def alive(self):
        return int(self.store._alive.sum())

    def high_water(self):
        return int(self.store._high_water)


class NativeBackend:
    def __init__(self, capacity):
        from manifoldx._core_native import _EntitySlots

        self.store = _EntitySlots(capacity)

    def spawn(self, n):
        return list(self.store.spawn(n))

    def destroy(self, ids):
        self.store.destroy([self.store.current(slot) for slot in ids])

    def reserve(self, n):
        return list(self.store.reserve(n))

    def ids(self, reservation):
        return [slot for slot, _ in reservation]

    def commit(self, reservation):
        self.store.commit(reservation)
        return self.ids(reservation)

    def visible(self):
        snapshot = self.store.snapshot()
        return [slot for slot, live in zip(snapshot["slots"], snapshot["alive"]) if live]

    def alive(self):
        return int(self.store.alive_count)

    def high_water(self):
        return int(self.store.high_water)


@pytest.fixture(params=[PythonBackend, NativeBackend], ids=["python", "native"])
def make_backend(request):
    return request.param


def test_capacity20_reuse_and_counters(make_backend):
    backend = make_backend(20)
    assert backend.spawn(10) == list(range(10))
    backend.destroy([2, 5, 7])
    assert backend.spawn(5) == [7, 5, 2, 10, 11]
    assert backend.alive() == 12
    assert backend.high_water() == 12


def test_capacity10_overflow_keeps_slots(make_backend):
    backend = make_backend(10)
    assert backend.spawn(8) == list(range(8))
    backend.destroy([0])
    with pytest.raises(RuntimeError, match="Max entities 10"):
        backend.reserve(4)
    assert backend.alive() == 7
    assert backend.high_water() == 8
    assert backend.visible() == list(range(1, 8))
    assert backend.commit(backend.reserve(3)) == [0, 8, 9]
    assert backend.alive() == 10
    assert backend.high_water() == 10


def test_two_reservations_are_disjoint_until_commit(make_backend):
    backend = make_backend(20)
    first = backend.reserve(2)
    second = backend.reserve(2)
    assert backend.ids(first) == [0, 1]
    assert backend.ids(second) == [2, 3]
    assert backend.visible() == []
    assert backend.alive() == 0
    assert backend.high_water() == 0
    assert backend.commit(second) == [2, 3]
    assert backend.visible() == [2, 3]
    assert backend.commit(first) == [0, 1]
    assert backend.visible() == [0, 1, 2, 3]


def test_python_reference_is_the_installed_entity_store():
    import manifoldx.ecs as ecs

    assert _installed(ecs.__file__)
    assert type(ecs.EntityStore(1)) is ecs.EntityStore
    assert ecs.EntityStore.__module__ == "manifoldx.ecs"


def test_native_extension_is_installed_compiled_and_native():
    import manifoldx

    assert _installed(manifoldx.__file__)
    spec = importlib.util.find_spec("manifoldx._core_native")
    assert spec is not None and spec.origin is not None
    assert _installed(spec.origin)
    assert isinstance(spec.loader, importlib.machinery.ExtensionFileLoader)
    native = importlib.import_module("manifoldx._core_native")
    assert _installed(native.__file__)
    slots = native._EntitySlots
    assert slots.__module__ == "manifoldx._core_native"
    assert type(slots(4)) is slots
