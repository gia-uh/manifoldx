"""A traced shape: named material layers of polygons, in units of the part's half-size.

A ``Shape`` is what a blueprint drawing becomes: a stack of ``Layer`` values,
each with a material name and a list of closed polygons. Coordinates are
``(u, v)`` with the part's half-size as the unit, so an outline that fills the
drawing spans ``-1 .. 1`` on its longer side. The outer side of the part is
``+u``: a shape traced from a drawing seen from behind must be mirrored first
(``mirror``), or its windows and lettering will face the wrong way.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

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

    def to_json(self) -> str:
        """Serialise the shape to JSON text (polygons as nested lists)."""
        payload = {
            "version": 1,
            "anchor": [float(self.anchor[0]), float(self.anchor[1])],
            "meta": self.meta,
            "layers": [
                {"material": layer.material,
                 "polygons": [np.asarray(poly, dtype=float).tolist() for poly in layer.polygons]}
                for layer in self.layers
            ],
        }
        return json.dumps(payload, separators=(",", ":"))

    @staticmethod
    def from_json(text: str) -> "Shape":
        """Build a Shape back from ``to_json`` text."""
        data = json.loads(text)
        layers = [
            Layer(item["material"],
                  [np.asarray(poly, dtype=float) for poly in item.get("polygons", [])])
            for item in data.get("layers", [])
        ]
        anchor = tuple(float(value) for value in data.get("anchor", (0.0, 0.0)))
        return Shape(layers=layers, anchor=anchor, meta=dict(data.get("meta", {})))

    def save(self, path) -> None:
        """Write the shape to ``path`` as JSON."""
        Path(path).write_text(self.to_json())

    @staticmethod
    def load(path) -> "Shape":
        """Read a shape written by ``save``."""
        return Shape.from_json(Path(path).read_text())


def _copy(shape: Shape, layers: list[Layer], anchor=None) -> Shape:
    return Shape(layers=layers, anchor=shape.anchor if anchor is None else anchor,
                 meta=dict(shape.meta))


def _flip_u(polygon) -> np.ndarray:
    flipped = np.asarray(polygon, dtype=float).copy()
    flipped[:, 0] = -flipped[:, 0]
    return flipped[::-1]


def mirror(shape: Shape) -> Shape:
    """Mirror a shape across ``u = 0``, keeping every polygon counter-clockwise."""
    layers = [Layer(layer.material, [_flip_u(poly) for poly in layer.polygons])
              for layer in shape.layers]
    anchor = (-float(shape.anchor[0]), float(shape.anchor[1]))
    return _copy(shape, layers, anchor)


def _crosses_zero(previous, current):
    span = current[0] - previous[0]
    if span == 0:
        return np.array([0.0, previous[1]])
    t = -previous[0] / span
    return previous + t * (current - previous)


def _clip_half(polygon, keep_left: bool):
    """Clip a polygon to ``u <= 0`` (or ``u >= 0``); None when nothing is left."""
    points = np.asarray(polygon, dtype=float)
    if len(points) == 0:
        return None

    def inside(point):
        return point[0] <= 0.0 if keep_left else point[0] >= 0.0

    output = []
    for index in range(len(points)):
        current = points[index]
        previous = points[index - 1]
        current_inside, previous_inside = inside(current), inside(previous)
        if current_inside:
            if not previous_inside:
                output.append(_crosses_zero(previous, current))
            output.append(current)
        elif previous_inside:
            output.append(_crosses_zero(previous, current))
    if len(output) < 3:
        return None
    return np.array(output, dtype=float)


def mirror_half(shape: Shape, keep: str = "left") -> Shape:
    """Keep one half of a symmetric drawing and reflect it across ``u = 0``.

    ``keep="left"`` keeps the polygons on ``u <= 0`` and mirrors them to the
    right; ``keep="right"`` does the opposite.
    """
    if keep not in ("left", "right"):
        raise ValueError("keep must be 'left' or 'right'")
    layers = []
    for layer in shape.layers:
        polygons = []
        for polygon in layer.polygons:
            clipped = _clip_half(polygon, keep == "left")
            if clipped is None:
                continue
            polygons.append(clipped)
            polygons.append(_flip_u(clipped))
        layers.append(Layer(layer.material, polygons))
    return _copy(shape, layers)


def bounds(shape: Shape):
    """The shape's bounding box as ``(u_min, v_min, u_max, v_max)``."""
    polygons = [poly for layer in shape.layers for poly in layer.polygons if len(poly)]
    if not polygons:
        return (0.0, 0.0, 0.0, 0.0)
    points = np.concatenate([np.asarray(poly, dtype=float) for poly in polygons], axis=0)
    low, high = points.min(axis=0), points.max(axis=0)
    return (float(low[0]), float(low[1]), float(high[0]), float(high[1]))


def _map(shape: Shape, function) -> Shape:
    layers = [Layer(layer.material, [function(poly) for poly in layer.polygons])
              for layer in shape.layers]
    anchor = function(np.array([shape.anchor], dtype=float))[0]
    return _copy(shape, layers, (float(anchor[0]), float(anchor[1])))


def stretch_middle(shape: Shape, ratio: float, ends: float = 0.25) -> Shape:
    """Make the shape ``ratio`` times longer along ``u``, keeping the ends as drawn.

    ``ends`` is the fraction of the length held at each end; only the middle
    stretches, so a grille's slats or a door's handle keep their proportions.
    """
    if ratio <= 0:
        raise ValueError("ratio must be positive")
    if not 0.0 <= ends < 0.5:
        raise ValueError("ends must be in [0, 0.5)")
    u_min, _v_min, u_max, _v_max = bounds(shape)
    length = u_max - u_min
    if length == 0.0:
        return _copy(shape, [Layer(layer.material, [np.array(poly, dtype=float)
                                                    for poly in layer.polygons])
                             for layer in shape.layers])
    centre = (u_min + u_max) / 2.0
    middle_half = length * (0.5 - ends)
    shift = (ratio - 1.0) * middle_half

    def transform(polygon):
        points = np.asarray(polygon, dtype=float).copy()
        offset = points[:, 0] - centre
        points[:, 0] = centre + np.where(offset <= -middle_half, offset - shift,
                                         np.where(offset >= middle_half, offset + shift,
                                                  offset * ratio))
        return points

    return _map(shape, transform)


def fit_box(shape: Shape, half_w: float, half_h: float) -> Shape:
    """Scale and centre the shape so its bounds fill ``[-half_w, half_w]`` by ``[-half_h, half_h]``."""
    u_min, v_min, u_max, v_max = bounds(shape)
    width, height = u_max - u_min, v_max - v_min
    scale_u = 2.0 * half_w / width if width else 1.0
    scale_v = 2.0 * half_h / height if height else 1.0
    centre_u, centre_v = (u_min + u_max) / 2.0, (v_min + v_max) / 2.0

    def transform(polygon):
        points = np.asarray(polygon, dtype=float).copy()
        points[:, 0] = (points[:, 0] - centre_u) * scale_u
        points[:, 1] = (points[:, 1] - centre_v) * scale_v
        return points

    return _map(shape, transform)
