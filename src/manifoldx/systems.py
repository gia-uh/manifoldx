"""System registration and execution."""

import os
import sys
import time
import typing
import warnings
from concurrent.futures import wait
from typing import Any, Callable

from manifoldx.commands import CommandBuffer

# The GIL warning is per process, not per engine: every parallel system on a
# build with the GIL on would repeat the same advice.
_gil_warned = False


def _gil_is_on() -> bool:
    """Whether this interpreter has the GIL (always True before 3.13's free-threading build)."""
    checker = getattr(sys, "_is_gil_enabled", None)
    return True if checker is None else bool(checker())


def _parallel_is_off() -> bool:
    """MANIFOLDX_PARALLEL=serial runs groups one system after another, to tell a race from a bug."""
    return os.environ.get("MANIFOLDX_PARALLEL", "").strip().lower() == "serial"


# =============================================================================
# System Registry
# =============================================================================


class System:
    """A registered system function."""

    def __init__(self, func: Callable, component_names: list, parallel: bool = False):
        self.func = func
        self.component_names = component_names
        self.parallel = bool(parallel)
        self.name = func.__qualname__
        self.removed = False

    def run(self, engine, dt: float):
        """Execute the system with component view."""
        view = engine.store.get_component_view(self.component_names, engine)
        self.func(view, dt)


class Query:
    """
    Type annotation for systems. Specifies required components.

    Usage:
        @engine.system
        def my_system(query: Query[Transform, Mesh], dt: float):
            ...
    """

    def __init__(self, components: tuple):
        self.components = components

    @classmethod
    def __class_getitem__(cls, components: tuple):
        return cls(components)


class SystemRegistry:
    """Registry of all systems."""

    def __init__(self):
        self._systems: list[System] = []

    def register(self, func: Callable, component_names: list = None, parallel: bool = False) -> Callable:
        """Register a system function with optional component names."""
        global _gil_warned
        if component_names is None:
            component_names = []

        if parallel and not _gil_warned and _gil_is_on():
            warnings.warn(
                "GIL on: parallel systems run one after another", RuntimeWarning, stacklevel=2
            )
            _gil_warned = True

        system = System(func, component_names, parallel)
        self._systems.append(system)
        return func

    def remove(self, func: Callable) -> None:
        """Unregister every registration of `func`; ValueError if it has none.

        Takes effect at once: a system removed mid-frame (even by itself, or by
        an earlier system) does not run again, this frame included. A parallel
        system removed before its group starts is skipped; one removed while
        its group is running finishes this frame.
        """
        matches = [s for s in self._systems if s.func is func]
        if not matches:
            raise ValueError(f"{func!r} is not a registered system")
        for system in matches:
            system.removed = True
        self._systems = [s for s in self._systems if s.func is not func]

    def run_all(self, engine, dt: float):
        """Execute all systems in order.

        Consecutive `parallel=True` systems form one group: each runs on the
        frame pool with a command buffer of its own, and the buffers are
        appended to the frame's in registration order afterwards. A serial
        system is a barrier and runs on the calling thread as before.

        Iterates a snapshot, so systems may be added or removed while it runs;
        systems added mid-frame start on the next frame.

        Each system's seconds land in `engine.system_split` and each group's
        wall seconds in `engine.group_split`; both are reset here, once a frame.
        """
        engine.system_split = {}
        engine.group_split = []
        snapshot = list(self._systems)
        i = 0
        while i < len(snapshot):
            system = snapshot[i]
            if system.removed:
                i += 1
                continue
            if not system.parallel:
                self._run_one(engine, system, dt)
                i += 1
                continue
            group = []
            while i < len(snapshot) and snapshot[i].parallel:
                if not snapshot[i].removed:
                    group.append(snapshot[i])
                i += 1
            if group:
                self._run_group(engine, group, dt)

    def _time(self, engine, system: System, run):
        """Run `run()` for `system`, adding its seconds to engine.system_split."""
        started = time.perf_counter()
        try:
            run()
        finally:
            seconds = time.perf_counter() - started
            engine.system_split[system.name] = engine.system_split.get(system.name, 0.0) + seconds

    def _run_one(self, engine, system: System, dt: float):
        """A serial system: the calling thread, no buffer bound (the frame's is used)."""
        self._time(engine, system, lambda: system.run(engine, dt))

    def _run_group(self, engine, group: list[System], dt: float):
        """A run of consecutive parallel systems, on the pool or one after another."""
        started = time.perf_counter()
        one_by_one = _parallel_is_off() or _gil_is_on() or len(group) == 1
        pairs = [(system, CommandBuffer()) for system in group]
        errors: list[tuple[System, BaseException]] = []

        if one_by_one:
            # Still one buffer per system: a parallel system's commands must not
            # mix with the frame's just because the pool is unavailable.
            for system, buf in pairs:
                try:
                    with engine._bind_commands(buf), engine._in_parallel(system.name):
                        self._time(engine, system, lambda s=system: s.run(engine, dt))
                except Exception as error:  # every system of a group runs, then the first fails
                    errors.append((system, error))
        else:
            pool = engine._get_frame_pool()

            def work(system: System, buf: CommandBuffer):
                with engine._bind_commands(buf), engine._in_parallel(system.name):
                    self._time(engine, system, lambda: system.run(engine, dt))

            futures = [(system, buf, pool.submit(work, system, buf)) for system, buf in pairs]
            wait([future for _, _, future in futures])
            for system, _, future in futures:
                error = future.exception()
                if error is not None:
                    errors.append((system, error))

        engine.group_split.append(time.perf_counter() - started)
        # In registration order, whatever order the systems finished in.
        for _, buf in pairs:
            engine._frame_commands.extend(buf)
        if errors:
            for system, error in errors:
                error.add_note(f"in parallel system {system.name}")
            raise errors[0][1]


__all__ = [
    "System",
    "Query",
    "SystemRegistry",
]
