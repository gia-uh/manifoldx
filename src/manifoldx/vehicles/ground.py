"""What the wheels stand on. The game passes its own; FlatGround and ShapeGround serve tests and examples."""

from __future__ import annotations

from typing import Protocol, Sequence

import numpy as np


class Ground(Protocol):
    def height(self, x: np.ndarray, z: np.ndarray, near: np.ndarray) -> np.ndarray:
        """The surface height under each (x, z); where there are several levels, the one nearest near."""

    def grip(self, x: np.ndarray, z: np.ndarray) -> np.ndarray:
        """The surface's factor on each tyre's mu: 1.0 on dry road."""


class FlatGround:
    """The plane y = level, with the same grip everywhere."""

    def __init__(self, level: float = 0.0, grip: float = 1.0):
        self.level, self._grip = level, grip

    def height(self, x, z, near=None):
        return np.full(np.shape(x), self.level, float)

    def grip(self, x, z):
        return np.full(np.shape(x), self._grip, float)


class ShapeGround:
    """A road along x for tests and examples: a height profile through (x, y) points, square kerbs that step up
    kerb_height at each x in kerbs, and a bridge deck at height y over x0 <= x <= x1 for every z."""

    def __init__(self, profile: Sequence[tuple[float, float]], kerbs: Sequence[float] = (), kerb_height: float = 0.15,
                 bridge: tuple[float, float, float] | None = None, grip: float = 1.0):
        self._px, self._py = np.array(profile, float).T
        self.kerbs, self.kerb_height, self.bridge, self._grip = np.array(kerbs, float), kerb_height, bridge, grip

    def height(self, x, z, near=None):
        x = np.asarray(x, float)
        base = np.interp(x, self._px, self._py)
        if len(self.kerbs):
            base = base + self.kerb_height * (x[..., None] >= self.kerbs).sum(-1)
        if self.bridge is None:
            return base
        x0, x1, deck = self.bridge
        near = base if near is None else np.broadcast_to(np.asarray(near, float), x.shape)
        over = (x >= x0) & (x <= x1) & (np.abs(near - deck) < np.abs(near - base))
        return np.where(over, deck, base)

    def grip(self, x, z):
        return np.full(np.shape(x), self._grip, float)
