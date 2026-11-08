"""Event-driven system: bus, handlers, read-only views, frame waiters."""

from __future__ import annotations

import asyncio
import inspect
from dataclasses import dataclass
from typing import Any, Callable

_RO_MUTATION_MSG = (
    "Event handlers cannot mutate ECS data directly. Use "
    "engine.commands.append(...), engine.spawn(...), or engine.destroy(...)."
)


class _ReadOnlyAccessor:
    """Wraps a ComponentAccessor to forbid attribute writes."""

    def __init__(self, accessor):
        object.__setattr__(self, "_accessor", accessor)

    def __getattr__(self, name):
        return getattr(self._accessor, name)

    def __setattr__(self, name, value):
        raise RuntimeError(_RO_MUTATION_MSG)


class ReadOnlyView:
    """Wraps a ComponentView; reads pass through, writes raise.

    Caveat: numpy arrays returned by reads are not frozen — in-place
    mutations like `view[C].field[i] = x` cannot be intercepted at the
    Python level and remain undefined behavior in event handlers.
    """

    def __init__(self, view):
        object.__setattr__(self, "_view", view)

    def __getitem__(self, component):
        accessor = self._view[component]
        return _ReadOnlyAccessor(accessor)

    def __setitem__(self, component, value):
        raise RuntimeError(_RO_MUTATION_MSG)

    def __setattr__(self, name, value):
        raise RuntimeError(_RO_MUTATION_MSG)

    def __len__(self):
        return len(self._view)

    def __iter__(self):
        return iter(self._view)

    def get_component_data(self, component_name):
        return self._view.get_component_data(component_name)


@dataclass
class Handler:
    func: Callable
    is_async: bool
    query_components: list[str] | None = None
    removed: bool = False


def _parse_query_components(func) -> list[str] | None:
    """Parse Query[A, B] annotation from a function's signature.

    Returns the list of component names if found, else None.
    Mirrors the logic in Engine.system().
    """
    from manifoldx.systems import Query

    hints = getattr(func, "__annotations__", {})
    for hint in hints.values():
        if isinstance(hint, Query):
            comps = hint.components
            if not isinstance(comps, tuple):
                comps = (comps,)
            names = []
            for c in comps:
                if isinstance(c, str):
                    names.append(c)
                elif hasattr(c, "__name__"):
                    names.append(c.__name__)
            return names
    return None


class EventBus:
    """Per-frame queued event bus.

    Emits go into `_pending`; `dispatch_pending` drains and delivers them.
    `dispatch_immediate` is for the engine's own 'startup' / 'shutdown'
    phase events, which must not wait a frame.
    """

    def __init__(self) -> None:
        self._handlers: dict[str, list[Handler]] = {}
        self._pending: list[tuple[str, Any]] = []

    def on(self, event: str):
        def decorator(func):
            handler = Handler(
                func=func,
                is_async=inspect.iscoroutinefunction(func),
                query_components=_parse_query_components(func),
            )
            self._handlers.setdefault(event, []).append(handler)
            return func

        return decorator

    def off(self, event: str, func) -> None:
        """Unregister every registration of `func` for `event`; ValueError if none.

        Takes effect at once, also mid-dispatch: a removed handler is not called
        again, not even for the event being delivered.
        """
        handlers = self._handlers.get(event, [])
        matches = [h for h in handlers if h.func is func]
        if not matches:
            raise ValueError(f"{func!r} is not a handler for {event!r}")
        for handler in matches:
            handler.removed = True
        self._handlers[event] = [h for h in handlers if h.func is not func]

    def emit(self, event: str, payload: Any = None) -> None:
        self._pending.append((event, payload))

    def dispatch_pending(self, engine) -> None:
        drained = self._pending
        self._pending = []
        for event, payload in drained:
            self._deliver(engine, event, payload)

    def dispatch_immediate(self, engine, event: str, payload: Any = None) -> None:
        self._deliver(engine, event, payload)

    def _deliver(self, engine, event: str, payload: Any) -> None:
        # A snapshot, so handlers may call on() / off() while it runs.
        for handler in list(self._handlers.get(event, ())):
            if handler.removed:
                continue
            if handler.is_async:
                _invoke_async(handler, payload, engine)
            else:
                _invoke_sync(handler, payload, engine)


def _invoke_sync(handler: Handler, payload, engine) -> None:
    if handler.query_components:
        view = engine.store.get_component_view(handler.query_components, engine)
        handler.func(payload, ReadOnlyView(view))
    else:
        handler.func(payload)


def _invoke_async(handler: Handler, payload, engine) -> None:
    if handler.query_components:
        view = engine.store.get_component_view(handler.query_components, engine)
        coro = handler.func(payload, ReadOnlyView(view))
    else:
        coro = handler.func(payload)
    loop = engine._get_active_loop()
    task = loop.create_task(coro)
    engine._tasks.add(task)
    task.add_done_callback(engine._on_task_done)


class FrameWaiters:
    """Per-frame inboxes for tick / delay / elapsed_at futures.

    Each call to add_* registers a fresh future on the appropriate list.
    `resolve(elapsed)` walks the lists, sets futures whose predicates hold,
    and removes them from their lists. The lists are not standing
    subscriptions — coroutines that loop must call add_* again next frame.

    The waiter does not own a loop. It accepts a `loop_provider` callable
    (typically engine._get_active_loop), invoked each time we need to
    create a future. This lets us adopt the running asyncio loop in
    interactive mode (rendercanvas) and a private fallback in headless
    mode (tests, render-to-MP4).
    """

    def __init__(self, loop_provider) -> None:
        self._loop_provider = loop_provider
        self._tick: list[asyncio.Future] = []
        self._deadlines: list[tuple[asyncio.Future, float]] = []  # delay + elapsed_at share storage

    def add_tick(self) -> asyncio.Future:
        fut = self._loop_provider().create_future()
        self._tick.append(fut)
        return fut

    def add_delay(self, seconds: float, current_elapsed: float) -> asyncio.Future:
        fut = self._loop_provider().create_future()
        self._deadlines.append((fut, current_elapsed + seconds))
        return fut

    def add_elapsed_at(self, target: float) -> asyncio.Future:
        fut = self._loop_provider().create_future()
        self._deadlines.append((fut, target))
        return fut

    def resolve(self, elapsed: float) -> None:
        # tick waiters resolve unconditionally
        for fut in self._tick:
            if not fut.done():
                fut.set_result(None)
        self._tick.clear()

        # deadline waiters: resolve those whose deadline is reached
        kept: list[tuple[asyncio.Future, float]] = []
        for fut, deadline in self._deadlines:
            if elapsed >= deadline - 1e-9:  # a step sum may land an ulp short
                if not fut.done():
                    fut.set_result(None)
            else:
                kept.append((fut, deadline))
        self._deadlines = kept

    def cancel_all(self) -> None:
        """Cancel every outstanding future (used at shutdown)."""
        for fut in self._tick:
            if not fut.done():
                fut.cancel()
        for fut, _ in self._deadlines:
            if not fut.done():
                fut.cancel()
        self._tick.clear()
        self._deadlines.clear()
