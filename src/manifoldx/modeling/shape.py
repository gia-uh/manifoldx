"""A traced shape: named material layers of polygons, in units of the part's half-size.

A ``Shape`` is what a blueprint drawing becomes: a stack of ``Layer`` values,
each with a material name and a list of closed polygons. Coordinates are
``(u, v)`` with the part's half-size as the unit, so an outline that fills the
drawing spans ``-1 .. 1`` on its longer side. The outer side of the part is
``+u``: a shape traced from a drawing seen from behind must be mirrored first
(``mirror``), or its windows and lettering will face the wrong way.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Layer:
    """One material of a shape: a name and its closed polygons, ``(K, 2)``."""

    material: str
    polygons: list[np.ndarray]


@dataclass
class Shape:
    """Stacked material layers traced from a drawing, in half-size units."""

    layers: list[Layer]
    anchor: tuple[float, float] = (0.0, 0.0)
    meta: dict = field(default_factory=dict)
