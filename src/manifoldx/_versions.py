"""Version counters for ``MANIFOLDX_PIPELINE=check``.

A pipelined frame renders frame N on the main thread while frame N+1's systems
run on another thread. The contract is that a system changes only its command
buffer, never anything the render reads; check mode pins that contract.

Set ``MANIFOLDX_PIPELINE=check`` before the engine is built. The store's arrays
(component data, members and alive), the geometry/material/volume registries,
the camera and the lights then carry a counter that every write bumps. The
render records the counters when it starts and compares them when it ends; a
change names the structure that broke the contract.

The counters cost a few integer increments a write, and are always maintained;
only the array wrapping is conditional on check mode, because a plain ndarray
cannot report an in-place write by a system.
"""

from __future__ import annotations

import os

import numpy as np


def _check_enabled() -> bool:
    """Whether ``MANIFOLDX_PIPELINE`` asks for the pipelined-frame check."""
    return os.environ.get("MANIFOLDX_PIPELINE", "").strip().lower() == "check"


class _Counter:
    """A writable version number, held by a structure the render reads."""

    __slots__ = ("value",)

    def __init__(self):
        self.value = 0

    def bump(self) -> None:
        self.value += 1

    def __eq__(self, other) -> bool:
        return isinstance(other, _Counter) and self.value == other.value

    def __repr__(self) -> str:
        return f"_Counter({self.value})"


class _VersionedArray(np.ndarray):
    """An ndarray whose writes bump a `_Counter`, views included.

    Only the store wraps its arrays, and only in check mode: it is the one
    structure whose rows a system may write by hand, bypassing every method
    that could bump a counter.
    """

    def __new__(cls, data, counter: _Counter):
        obj = np.asarray(data).view(cls)
        obj._version = counter
        return obj

    def __array_finalize__(self, obj):
        if obj is not None:
            self._version = getattr(obj, "_version", None)

    def _bump(self) -> None:
        counter = getattr(self, "_version", None)
        if counter is not None:
            counter.bump()

    def __setitem__(self, key, value):
        self._bump()
        return super().__setitem__(key, value)

    def __delitem__(self, key):
        self._bump()
        return super().__delitem__(key)

    def fill(self, value):
        self._bump()
        return super().fill(value)

    def put(self, indices, values, mode="raise"):
        self._bump()
        return super().put(indices, values, mode=mode)

    def __iadd__(self, other):
        self._bump()
        return super().__iadd__(other)

    def __isub__(self, other):
        self._bump()
        return super().__isub__(other)

    def __imul__(self, other):
        self._bump()
        return super().__imul__(other)

    def __itruediv__(self, other):
        self._bump()
        return super().__itruediv__(other)


__all__ = ["_Counter", "_VersionedArray", "_check_enabled"]
