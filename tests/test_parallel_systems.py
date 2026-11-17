"""Parallel systems: groups of consecutive `parallel=True` systems, one buffer each.

Every test drives the systems phase directly (`systems.run_all` then the frame
flush) on an engine with no canvas, so it needs no GPU.
"""

import sys
import threading
import time
import warnings

import numpy as np
import pytest

import manifoldx as mx
from manifoldx.components import Component
from manifoldx.types import Float

WARN = "GIL on: parallel systems run one after another"


class Value(Component):
    """One float per entity: the column the systems below contend for."""

    value: Float


def _engine(max_entities: int = 64):
    return mx.Engine("parallel", width=64, height=64, max_entities=max_entities)


def _spawn(engine, n: int = 12):
    engine.spawn(Value(value=np.arange(n, dtype=np.float32)), n=n)


def _step(engine, dt: float = 1 / 60):
    """One frame's systems phase and command flush, without the renderer."""
    engine.systems.run_all(engine, dt)
    engine.commands.execute(engine.store)
    engine.commands.clear()


def _store_bytes(engine) -> bytes:
    return b"".join(engine.store._components[name].tobytes() for name in sorted(engine.store._components))


def _overlapping_run(flags, frames: int = 200) -> bytes:
    """Six systems writing the same column, run `frames` times; returns the store's bytes."""
    engine = _engine()
    _spawn(engine)
    for i, parallel in enumerate(flags):

        def step(q: mx.Query[Value], dt: float, k=i):
            q[Value].value = q[Value].value * np.float32(1.0 + 0.01 * (k + 1)) + np.float32(k)

        engine.system(step, parallel=parallel)
    for _ in range(frames):
        _step(engine)
    return _store_bytes(engine)


def test_group_equals_serial_run():
    """6 systems writing overlapping columns, 200 frames, parallel vs plain: stores equal byte for byte."""
    flags = [True, True, True, False, True, True]
    assert _overlapping_run(flags) == _overlapping_run([False] * len(flags))


def test_commands_apply_in_registration_order():
    """A and B write the same rows; A sleeps so B finishes first; the store holds B's value."""
    engine = _engine()
    _spawn(engine, 4)

    @engine.system(parallel=True)
    def slow(q: mx.Query[Value], dt: float):
        q[Value].value = np.ones((len(q), 1), np.float32)
        time.sleep(0.05)

    @engine.system(parallel=True)
    def quick(q: mx.Query[Value], dt: float):
        q[Value].value = np.full((len(q), 1), 2.0, np.float32)

    _step(engine)
    assert engine.store._components["Value"][:4, 0].tolist() == [2.0, 2.0, 2.0, 2.0]


def test_serial_system_is_a_barrier():
    """P1, S, P2: S starts after P1 ends, P2 starts after S ends (timestamps)."""
    engine = _engine()
    _spawn(engine)
    stamps = []

    def mark(name: str, when: str):
        stamps.append((name, when, time.perf_counter()))

    def first(q, dt: float):
        mark("p1a", "start")
        time.sleep(0.03)
        mark("p1a", "end")

    def second(q, dt: float):
        mark("p1b", "start")
        time.sleep(0.03)
        mark("p1b", "end")

    @engine.system
    def between(q, dt: float):
        mark("s", "start")
        time.sleep(0.02)
        mark("s", "end")

    def third(q, dt: float):
        mark("p2a", "start")
        time.sleep(0.01)
        mark("p2a", "end")

    def fourth(q, dt: float):
        mark("p2b", "start")
        time.sleep(0.01)
        mark("p2b", "end")

    for func in (first, second):
        engine.system(func, parallel=True)
    engine.system(between)
    for func in (third, fourth):
        engine.system(func, parallel=True)

    _step(engine)

    at = {(name, when): t for name, when, t in stamps}
    assert at[("s", "start")] >= max(at[("p1a", "end")], at[("p1b", "end")])
    assert min(at[("p2a", "start")], at[("p2b", "start")]) >= at[("s", "end")]


def test_destroy_in_parallel_system_lands_in_order():
    """Destroy from two parallel systems: same survivors as serial."""
    def survivors(parallel: bool):
        engine = _engine()
        engine.spawn(Value(value=np.zeros(8, np.float32)), n=8)

        def first(q, dt: float):
            engine.destroy(np.array([1, 2], np.int32))

        def second(q, dt: float):
            engine.destroy(np.array([3, 4], np.int32))

        engine.system(first, parallel=parallel)
        engine.system(second, parallel=parallel)
        _step(engine)
        return engine.store._alive.tolist()

    assert survivors(True) == survivors(False)
    assert survivors(True)[1:5] == [False, False, False, False]


def test_spawn_in_parallel_system_defers_until_the_group_applies():
    """A parallel spawn: the handle is valid at once, the row is invisible to a query until the
    group's buffers apply, and it is alive and filled afterwards."""
    engine = _engine()
    Value.register(engine.store)
    seen = {}

    @engine.system(parallel=True)
    def grower(q: mx.Query[Value], dt: float):
        handle = engine.spawn(Value(value=7.0))
        seen["handle"] = handle
        seen["reserved"] = bool(engine.store._reserved[handle.index])
        seen["alive"] = bool(engine.store._alive[handle.index])
        seen["rows"] = len(engine.store.get_component_view(["Value"], engine))

    _step(engine)

    handle = seen["handle"]
    assert handle is not None and handle.index >= 0
    # Valid at once: the slot is reserved, so no other spawn takes it, and no row exists yet.
    assert seen["reserved"] is True
    assert seen["alive"] is False
    assert seen["rows"] == 0
    # Applied at the frame's flush, like the group's query writes.
    assert engine.store._alive[handle.index]
    assert engine.store._components["Value"][handle.index, 0] == pytest.approx(7.0)
    assert list(engine.store.get_component_view(["Value"], engine)) == [handle.index]


def test_two_parallel_systems_spawn_disjoint_indices_in_registration_order():
    """Two systems spawning in one group: disjoint indices, and their writes reach the store in
    registration order (the store's release hooks hear each spawn in that order)."""
    engine = _engine()
    Value.register(engine.store)
    handles = {}
    released = []
    engine.store._release_hooks.append(
        lambda indices: released.append(tuple(int(i) for i in indices))
    )

    def first(q, dt: float):
        handles["first"] = [engine.spawn(Value(value=1.0)).index for _ in range(3)]

    def second(q, dt: float):
        handles["second"] = [engine.spawn(Value(value=2.0)).index for _ in range(3)]

    engine.system(first, parallel=True)
    engine.system(second, parallel=True)
    _step(engine)

    a, b = handles["first"], handles["second"]
    assert len(a) == len(b) == 3
    assert set(a).isdisjoint(b)
    assert sorted(a + b) == list(range(6))
    assert released == [(i,) for i in a] + [(i,) for i in b]
    assert engine.store._alive[a + b].all()


def test_serial_system_after_the_group_sees_the_spawn():
    """A spawn from a parallel group is in the store once its buffers apply, so a serial
    system that runs after the group finds it in its query; inside the same frame it does
    not, as query writes already behave."""
    engine = _engine()
    Value.register(engine.store)
    counts = []

    @engine.system(parallel=True)
    def grower(q: mx.Query[Value], dt: float):
        if len(q) == 0:
            engine.spawn(Value(value=3.0))

    @engine.system
    def watcher(q: mx.Query[Value], dt: float):
        counts.append(len(q))

    _step(engine)  # the group spawns; the serial system runs before the flush
    _step(engine)  # the serial system's query sees the spawned entity

    assert counts == [0, 1]
    assert engine.store._components["Value"][0, 0] == pytest.approx(3.0)


def test_exception_names_the_system():
    """A parallel system raising ValueError: the error reaches the caller of the frame with the
    system's name in its notes."""
    engine = _engine()

    @engine.system(parallel=True)
    def explode(q, dt: float):
        raise ValueError("boom")

    @engine.system(parallel=True)
    def still_runs(q, dt: float):
        pass

    with pytest.raises(ValueError, match="boom") as caught:
        _step(engine)
    assert any(f"in parallel system {explode.__qualname__}" in note
               for note in getattr(caught.value, "__notes__", []))


def test_system_removed_by_a_serial_one_skips():
    """S removes P2 before P2's group: P2 does not run that frame."""
    engine = _engine()
    ran = []

    def p2(q, dt: float):
        ran.append("p2")

    @engine.system
    def remover(q, dt: float):
        engine.remove_system(p2)

    engine.system(p2, parallel=True)
    _step(engine)
    assert ran == []


def test_bare_decorator_still_works():
    """@engine.system without parentheses registers a serial system."""
    engine = _engine()
    _spawn(engine)
    ran = []

    @engine.system
    def tick(q: mx.Query[Value], dt: float):
        ran.append(len(q))

    assert engine.systems._systems[0].parallel is False
    _step(engine)
    assert ran == [12]


def test_system_split_has_every_system():
    """After a frame, engine.system_split has one entry per system name, in seconds."""
    engine = _engine()
    _spawn(engine)

    @engine.system(parallel=True)
    def first(q: mx.Query[Value], dt: float):
        q[Value].value = q[Value].value + np.float32(1.0)

    @engine.system(parallel=True)
    def second(q: mx.Query[Value], dt: float):
        q[Value].value = q[Value].value + np.float32(2.0)

    @engine.system
    def bar(q: mx.Query[Value], dt: float):
        q[Value].value = q[Value].value + np.float32(3.0)

    _step(engine)
    names = {first.__qualname__, second.__qualname__, bar.__qualname__}
    assert set(engine.system_split) == names
    assert all(isinstance(seconds, float) and seconds >= 0.0 for seconds in engine.system_split.values())
    assert sum(engine.system_split.values()) > 0.0
    # One parallel group, then the serial system's own run.
    assert len(engine.group_split) == 1
    assert all(isinstance(seconds, float) and seconds >= 0.0 for seconds in engine.group_split)


def test_serial_fallback_runs_every_system_on_the_main_thread(monkeypatch):
    """MANIFOLDX_PARALLEL=serial runs a group one system after another, each with its own buffer."""
    monkeypatch.setenv("MANIFOLDX_PARALLEL", "serial")
    engine = _engine()
    _spawn(engine)
    seen = []

    def record(q, dt: float):
        seen.append((threading.get_ident(), engine.commands is engine._frame_commands))

    for _ in range(3):
        engine.system(record, parallel=True)

    _step(engine)
    assert seen == [(threading.get_ident(), False)] * 3


@pytest.mark.skipif(
    not getattr(sys, "_is_gil_enabled", lambda: True)(),
    reason="a free-threaded build runs groups on the pool",
)
def test_gil_warning_is_emitted_once_per_process(monkeypatch):
    """The first parallel system registered on a GIL build warns once, not every time."""
    import manifoldx.systems as systems

    monkeypatch.setattr(systems, "_gil_warned", False)
    engine = _engine()

    with pytest.warns(RuntimeWarning, match=WARN):
        engine.system(lambda q, dt: None, parallel=True)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        engine.system(lambda q, dt: None, parallel=True)
    assert not [warning for warning in caught if WARN in str(warning.message)]
