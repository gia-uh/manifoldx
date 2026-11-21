"""Controls: named floats in shared memory that one process writes and another reads.

An app writes them once per frame (an engine's rpm, a listener's position); the sound
process reads them when it renders a block. Each name keeps this frame's value and the last
frame's, so a reader can ramp between the two across a block.
"""

from __future__ import annotations

from collections.abc import Sequence
from multiprocessing import shared_memory

import numpy as np


class Controls:
    """A table of float64 slots by name: row 0 holds the values set this frame, row 1 the
    values of the frame before, copied by tick()."""

    def __init__(self, shm: shared_memory.SharedMemory, names: Sequence[str]):
        self._shm = shm
        self.names = tuple(names)
        self._index = {n: i for i, n in enumerate(self.names)}
        self._v = np.ndarray((2, max(1, len(self.names))), np.float64, buffer=shm.buf)

    @classmethod
    def create(cls, names: Sequence[str]) -> Controls:
        """A new table, every value 0, owned by this process (close and unlink it)."""
        shm = shared_memory.SharedMemory(create=True, size=16 * max(1, len(names)))
        c = cls(shm, names)
        c._v[:] = 0.0
        return c

    @classmethod
    def attach(cls, spec: dict) -> Controls:
        """The table another process created, from its spec()."""
        return cls(shared_memory.SharedMemory(name=spec["shm"], track=False), spec["names"])

    def spec(self) -> dict:
        """What another process needs to attach: the block's name and the slot names."""
        return {"shm": self._shm.name, "names": list(self.names)}

    def _i(self, name: str) -> int:
        try:
            return self._index[name]
        except KeyError:
            raise KeyError(f"no control named {name!r}") from None

    def set(self, name: str, value: float) -> None:
        self._v[0, self._i(name)] = value

    def get(self, name: str) -> float:
        return float(self._v[0, self._i(name)])

    def prev(self, name: str) -> float:
        return float(self._v[1, self._i(name)])

    def tick(self) -> None:
        """End the frame: this frame's values become the previous ones."""
        self._v[1] = self._v[0]

    def close(self) -> None:
        del self._v
        self._shm.close()

    def unlink(self) -> None:
        self._shm.unlink()
