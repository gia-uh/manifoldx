"""Deferring engine calls into the current command buffer.

With pipelining off, which is the default and always the case in headless
frames, every engine call acts at once, exactly as before. While a pipelined
frame's systems run, a thread is marked as deferring (``Engine._defer_calls()``
in tests, the systems thread in a pipelined frame): the calls that change what
the render reads append a ``CALL`` command to that thread's command buffer
instead, and run, in call order, on the main thread at the apply step.

Read-your-writes inside one frame is the cost: a system that sets the camera
and reads it back gets the previous pose, as query writes already behave.
"""

from __future__ import annotations

import functools
import threading
from contextlib import contextmanager

from manifoldx.commands import Command, CommandType

_state = threading.local()  # .engine: the engine whose calls this thread defers


def current():
    """The engine whose calls this thread defers, or None outside such a block."""
    return getattr(_state, "engine", None)


def deferring(engine=None) -> bool:
    """Whether this thread defers calls: of `engine` when given, else of any engine."""
    active = current()
    return active is not None and (engine is None or active is engine)


@contextmanager
def defer_calls(engine):
    """Mark this thread's calls to `engine` as deferred until the block ends."""
    previous = current()
    _state.engine = engine
    try:
        yield
    finally:
        _state.engine = previous


def origin(engine) -> str | None:
    """The name of the system this thread runs, for a command's notes."""
    return getattr(engine._command_tls, "system_name", None)


def defer(engine, fn, *args, **kwargs) -> None:
    """Queue `fn(*args, **kwargs)` on `engine`'s current command buffer."""
    engine.commands.append(
        Command(
            CommandType.CALL,
            {"fn": fn, "args": args, "kwargs": kwargs, "origin": origin(engine)},
        )
    )


def _find_engine(args, kwargs):
    """The engine a call belongs to: its receiver, the object's engine, or None."""
    if not args:
        return None
    obj = args[0]
    if hasattr(obj, "_deferring"):
        return obj
    for name in ("_engine", "engine"):
        owner = getattr(obj, name, None)
        if owner is not None and hasattr(owner, "_deferring"):
            return owner
    return None


def deferred_in_pipeline(method=None, *, find=None):
    """Defer `method` into the engine's command buffer while the engine defers.

    With the flag off the call runs through unchanged. `find(args, kwargs)`
    locates the engine for an object that is not one (a handle, camera,
    registry, streamer); the default reads its `_engine` or `engine`, and
    falls back to the engine this thread defers for, so module-level calls
    (a GUI theme, a texture write) defer too.
    """

    def decorate(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            engine = (find or _find_engine)(args, kwargs) or current()
            if engine is not None and engine._deferring():
                defer(engine, fn, *args, **kwargs)
                return None
            return fn(*args, **kwargs)

        return wrapper

    return decorate if method is None else decorate(method)


class _DeferredWrites:
    """Mix in so public field writes defer while the current engine defers.

    Only a rewrite of a field the object already holds defers: the first
    assignment is construction or setup, which nothing has read yet, so it
    acts at once even inside a deferring system. Subclasses whose fields are
    properties (a widget's `.text`) override `_defer_write`.
    """

    def __setattr__(self, name, value):
        engine = current()
        if self._defer_write(engine, name):
            defer(engine, object.__setattr__, self, name, value)
            return
        object.__setattr__(self, name, value)

    def _defer_write(self, engine, name) -> bool:
        return (
            engine is not None
            and engine._deferring()
            and not name.startswith("_")
            and name in self.__dict__
        )


class _DeferredList:
    """A list view whose mutations defer while the engine defers.

    Reads forward to the real list, so a system sees the values applied so
    far; a mutation appends a CALL that runs at the apply step, in call order.
    """

    def __init__(self, engine, target: list):
        self._engine = engine
        self._target = target

    # -- reads ---------------------------------------------------------------
    def __len__(self):
        return len(self._target)

    def __iter__(self):
        return iter(self._target)

    def __getitem__(self, key):
        return self._target[key]

    def __contains__(self, item):
        return item in self._target

    def __eq__(self, other):
        return self._target == (other._target if isinstance(other, _DeferredList) else other)

    def __repr__(self):
        return repr(self._target)

    # -- mutations, queued ---------------------------------------------------
    def _queue(self, name, *args):
        defer(self._engine, getattr(self._target, name), *args)

    def append(self, item):
        self._queue("append", item)

    def extend(self, items):
        self._queue("extend", list(items))

    def insert(self, index, item):
        self._queue("insert", index, item)

    def remove(self, item):
        self._queue("remove", item)

    def pop(self, index=-1):
        self._queue("pop", index)
        return None

    def clear(self):
        self._queue("clear")

    def sort(self, *, key=None, reverse=False):
        self._queue("sort", key=key, reverse=reverse)

    def reverse(self):
        self._queue("reverse")

    def __setitem__(self, key, value):
        self._queue("__setitem__", key, value)

    def __delitem__(self, key):
        self._queue("__delitem__", key)

    def __iadd__(self, items):
        self.extend(items)
        return self


def deferring_list(engine, target: list):
    """`target` itself, or a deferring view of it while `engine` defers."""
    return _DeferredList(engine, target) if engine._deferring() else target


__all__ = [
    "current",
    "defer",
    "defer_calls",
    "deferring",
    "deferring_list",
    "deferred_in_pipeline",
    "origin",
]
