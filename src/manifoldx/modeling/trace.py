"""Trace a flat-colour blueprint drawing into :class:`~manifoldx.modeling.shape.Shape`.

A blueprint is an image whose regions are a fixed colour code: a ``palette``
maps material names to RGB. Everything here works on numpy arrays; a mask is a
``bool (H, W)`` array, and images are ``uint8 (H, W, 3)`` with row 0 at the top.
"""

from __future__ import annotations

import numpy as np

from manifoldx.modeling.shape import Layer, Shape

__all__ = [
    "classify", "open_mask", "close_mask", "dilate", "erode", "components",
    "contours", "simplify", "cut_by_rays", "trace_layers", "redraw", "iou",
]


# --- Colour classification -------------------------------------------------


def classify(img, palette) -> np.ndarray:
    """Return, per pixel, the index of the nearest palette colour (RGB distance).

    ``palette`` is a mapping ``name -> (r, g, b)``; its order is the label order.
    An RGBA image is classified on its colour channels.
    """
    pixels = np.asarray(img)
    if pixels.ndim != 3 or pixels.shape[2] not in (3, 4):
        raise ValueError("images must be (H, W, 3) or (H, W, 4)")
    colours = np.array(list(palette.values()), dtype=np.float32).reshape(-1, 3)
    if not len(colours):
        raise ValueError("palette is empty")
    work = pixels[..., :3].astype(np.float32)
    # One distance plane per colour; argmin keeps the first palette entry on a
    # tie, exactly as comparing each colour in order against the best so far.
    distances = np.stack([np.sum((work - colour) ** 2, axis=2) for colour in colours])
    return distances.argmin(axis=0).astype(np.int32)


# --- Morphology ------------------------------------------------------------


def _offsets(radius: int, shape: str) -> np.ndarray:
    r = max(0, int(radius))
    if shape not in ("square", "disc"):
        raise ValueError("shape must be 'square' or 'disc'")
    dy, dx = np.mgrid[-r:r + 1, -r:r + 1]
    keep = np.ones_like(dy, dtype=bool) if shape == "square" else dx * dx + dy * dy <= r * r
    return np.stack([dy[keep], dx[keep]], axis=1)


def _shift(mask: np.ndarray, dy: int, dx: int) -> np.ndarray:
    out = np.zeros_like(mask)
    h, w = mask.shape
    sy0, sy1 = max(0, -dy), min(h, h - dy)
    sx0, sx1 = max(0, -dx), min(w, w - dx)
    if sy1 > sy0 and sx1 > sx0:
        out[sy0 + dy:sy1 + dy, sx0 + dx:sx1 + dx] = mask[sy0:sy1, sx0:sx1]
    return out


def dilate(mask, r, shape: str = "square") -> np.ndarray:
    """Grow ``mask`` by a square or disc element of radius ``r`` (pixels)."""
    work = np.asarray(mask, dtype=bool)
    out = np.zeros_like(work)
    for dy, dx in _offsets(r, shape):
        out |= _shift(work, dy, dx)
    return out


def erode(mask, r, shape: str = "square") -> np.ndarray:
    """Shrink ``mask`` by a square or disc element of radius ``r`` (pixels)."""
    return ~dilate(~np.asarray(mask, dtype=bool), r, shape)


def open_mask(mask, r, shape: str = "square") -> np.ndarray:
    """An opening: erode then dilate, which removes specks and thin spikes."""
    return dilate(erode(mask, r, shape), r, shape)


def close_mask(mask, r, shape: str = "square") -> np.ndarray:
    """A closing: dilate then erode, which fills pinholes and narrow gaps."""
    return erode(dilate(mask, r, shape), r, shape)


# --- Components ------------------------------------------------------------


def components(mask, connectivity: int = 4):
    """Label connected components; return ``(labels, count)``, 0 is background."""
    work = np.asarray(mask, dtype=bool)
    if connectivity == 4:
        steps = ((-1, 0), (1, 0), (0, -1), (0, 1))
    elif connectivity == 8:
        steps = ((-1, -1), (-1, 0), (-1, 1), (0, -1),
                 (0, 1), (1, -1), (1, 0), (1, 1))
    else:
        raise ValueError("connectivity must be 4 or 8")
    labels = np.zeros(work.shape, dtype=np.int32)
    height, width = work.shape
    count = 0
    for row, col in zip(*np.nonzero(work)):
        if labels[row, col]:
            continue
        count += 1
        labels[row, col] = count
        stack = [(row, col)]
        while stack:
            current_row, current_col = stack.pop()
            for dr, dc in steps:
                rr, cc = current_row + dr, current_col + dc
                if 0 <= rr < height and 0 <= cc < width and work[rr, cc] and labels[rr, cc] == 0:
                    labels[rr, cc] = count
                    stack.append((rr, cc))
    return labels, count


# --- Contours --------------------------------------------------------------

# Marching-squares segments per corner code (bit 8 = top-left, 4 = top-right,
# 2 = bottom-right, 1 = bottom-left). Edge midpoints are E0 top, E1 right,
# E2 bottom, E3 left. Each segment is written so the foreground lies on its
# right in image coordinates (row increasing downward).
# Marching-squares segments per corner code (bit 8 = top-left, 4 = top-right,
# 2 = bottom-right, 1 = bottom-left). Edge midpoints are E0 top, E1 right,
# E2 bottom, E3 left. Each segment is written so the foreground lies on its
# right, so outer loops come out clockwise in image coordinates (row down)
# and their signed area is positive.
_CASES = {
    1: (("E3", "E2"),),          # bottom-left
    2: (("E2", "E1"),),          # bottom-right
    3: (("E3", "E1"),),          # bottom half
    4: (("E1", "E0"),),          # top-right
    5: (("E1", "E0"), ("E3", "E2")),  # top-right and bottom-left
    6: (("E2", "E0"),),          # right half
    7: (("E3", "E0"),),          # all but top-left
    8: (("E0", "E3"),),          # top-left
    9: (("E0", "E2"),),          # left half
    10: (("E0", "E3"), ("E2", "E1")),  # top-left and bottom-right
    11: (("E0", "E1"),),         # all but top-right
    12: (("E1", "E3"),),         # top half
    13: (("E1", "E2"),),         # all but bottom-right
    14: (("E2", "E3"),),         # all but bottom-left
}


# Edge midpoints in doubled pixel coordinates, as offsets from a cell's
# (2 * column, 2 * row) origin. Doubling makes the half-pixel midpoints exact
# integers, so the segment endpoints can be matched by integer key.
_EDGES = {
    "E0": (-1, -2),
    "E1": (0, -1),
    "E2": (-1, 0),
    "E3": (-2, -1),
}


def _edge_tables():
    """Per slot (0, 1) and corner code: start and end edge offsets."""
    starts = np.zeros((2, 16, 2), dtype=np.int64)
    ends = np.zeros((2, 16, 2), dtype=np.int64)
    present = np.zeros((2, 16), dtype=bool)
    for code, segments in _CASES.items():
        for slot, (start, end) in enumerate(segments):
            starts[slot, code] = _EDGES[start]
            ends[slot, code] = _EDGES[end]
            present[slot, code] = True
    return starts, ends, present


_STARTS, _ENDS, _PRESENT = _edge_tables()


def _loops(mask: np.ndarray) -> list[np.ndarray]:
    """Closed loops of the mask boundary, in pixel coordinates (y up).

    The marching-squares cells are evaluated for the whole grid at once; the
    resulting segments are then chained, each edge midpoint having exactly one
    outgoing segment (a saddle where two do meet is resolved in index order,
    as before).
    """
    work = np.asarray(mask, dtype=bool)
    height, width = work.shape
    padded = np.zeros((height + 2, width + 2), dtype=bool)
    padded[1:-1, 1:-1] = work

    corners = padded.astype(np.uint8)
    code = ((corners[:-1, :-1] << 3) | (corners[:-1, 1:] << 2)
            | (corners[1:, 1:] << 1) | corners[1:, :-1])

    # Every (cell, slot) pair that produces a segment, in the same order the
    # per-cell loop used to append them: cell row-major, then slot.
    cells, slots = [], []
    for slot in range(2):
        rows, cols = np.nonzero(_PRESENT[slot][code])
        cells.append(rows * (width + 1) + cols)
        slots.append(np.full(len(rows), slot, dtype=np.int64))
    cell = np.concatenate(cells)
    slot = np.concatenate(slots)
    if len(cell) == 0:
        return []
    order = np.argsort(cell * 2 + slot, kind="stable")
    cell, slot = cell[order], slot[order]

    cell_cols = cell % (width + 1)
    cell_rows = cell // (width + 1)
    cell_code = code.reshape(-1)[cell]
    start_x = 2 * cell_cols + _STARTS[slot, cell_code, 0]
    start_y = 2 * cell_rows + _STARTS[slot, cell_code, 1]
    end_x = 2 * cell_cols + _ENDS[slot, cell_code, 0]
    end_y = 2 * cell_rows + _ENDS[slot, cell_code, 1]

    # Integer keys for the midpoint keys; +2 keeps them non-negative and the
    # span is wider than the doubled width, so the pair maps injectively.
    span = 2 * width + 4
    start_key = (start_y + 2) * span + (start_x + 2)
    end_key = (end_y + 2) * span + (end_x + 2)

    count = len(start_key)
    by_start = np.argsort(start_key, kind="stable")
    sorted_keys = start_key[by_start]
    left = np.searchsorted(sorted_keys, end_key, side="left")
    right = np.searchsorted(sorted_keys, end_key, side="right")
    found = left < count
    safe = np.clip(left, 0, count - 1)
    found &= sorted_keys[safe] == end_key
    first_next = np.where(found, by_start[np.where(found, safe, 0)], -1)

    # A midpoint with several outgoing segments (a corner-touching saddle) is
    # resolved by taking the lowest-numbered unused one.
    multi = {}
    for group_left, group_right in zip(left, right):
        if group_right - group_left > 1:
            key = int(sorted_keys[group_left])
            multi[key] = [int(k) for k in by_start[group_left:group_right]]

    used = np.zeros(count, dtype=bool)
    xs, ys = start_x.tolist(), start_y.tolist()
    loops = []
    for first in range(count):
        if used[first]:
            continue
        index = first
        start_key_first = start_key[first]
        points = []
        while True:
            used[index] = True
            points.append(index)
            key = end_key[index]
            if key == start_key_first:
                break
            candidates = multi.get(int(key))
            if candidates is None:
                index = int(first_next[index])
                if index < 0:
                    break
            else:
                index = next((k for k in candidates if not used[k]), -1)
                if index < 0:
                    break
        if len(points) >= 3:
            loops.append(np.stack([np.array([xs[p] for p in points], dtype=float),
                                   np.array([ys[p] for p in points], dtype=float)],
                                  axis=1) * 0.5)
    return loops


def contours(mask, holes: bool = False):
    """Outer boundaries of every component, as closed polygons (no repeated end).

    Points are ``(x, y)`` in pixel coordinates with ``y`` up, ordered
    counter-clockwise. With ``holes=True`` the result is ``(outer, holes)``;
    hole loops are clockwise, so their signed area is negative.
    """
    outer, inner = [], []
    for loop in _loops(mask):
        x, y = loop[:, 0], loop[:, 1]
        area = 0.5 * float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))
        flipped = np.stack([loop[:, 0], -loop[:, 1]], axis=1)
        # In a y-down frame outer loops wind one way; flipping y and reversing
        # the order makes outer loops counter-clockwise in the y-up frame.
        flipped = flipped[::-1]
        if area >= 0:
            outer.append(flipped)
        else:
            inner.append(flipped)
    if holes:
        return outer, inner
    return outer


# --- Simplification --------------------------------------------------------


def _douglas_peucker(points: np.ndarray, tol: float) -> np.ndarray:
    if len(points) <= 2:
        return points
    keep = np.zeros(len(points), dtype=bool)
    keep[0] = keep[-1] = True
    stack = [(0, len(points) - 1)]
    while stack:
        first, last = stack.pop()
        if last <= first + 1:
            continue
        start, end = points[first], points[last]
        span = end - start
        length = float(np.hypot(*span))
        if length == 0:
            distance = np.linalg.norm(points[first + 1:last] - start, axis=1)
        else:
            relative = points[first + 1:last] - start
            distance = np.abs(span[0] * relative[:, 1] - span[1] * relative[:, 0]) / length
        if len(distance) == 0:
            continue
        worst = int(np.argmax(distance))
        if distance[worst] > tol:
            middle = first + 1 + worst
            keep[middle] = True
            stack.append((first, middle))
            stack.append((middle, last))
    return points[keep]


def simplify(poly, tol: float) -> np.ndarray:
    """Douglas-Peucker on a closed polygon: drop points within ``tol`` of a chord."""
    points = np.asarray(poly, dtype=float)
    if len(points) <= 3:
        return points.copy()
    distances = np.linalg.norm(points - points[0], axis=1)
    split = int(np.argmax(distances))
    if split in (0, len(points) - 1):
        split = len(points) // 2
    first = _douglas_peucker(points[:split + 1], tol)
    second = _douglas_peucker(np.concatenate([points[split:], points[:1]]), tol)
    return np.concatenate([first[:-1], second[:-1]])


def _area(poly) -> float:
    points = np.asarray(poly, dtype=float)
    x, y = points[:, 0], points[:, 1]
    return 0.5 * float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))


# --- Cutting and rasterising ----------------------------------------------


def cut_by_rays(mask, centre, angles, width):
    """Remove a thin strip of ``width`` pixels along each ray from ``centre``.

    ``angles`` are radians. Each strip runs from the centre outward and on past
    the mask, so a disc with round windows becomes pieces with no holes.
    """
    work = np.asarray(mask, dtype=bool).copy()
    height, width_px = work.shape
    yy, xx = np.indices((height, width_px))
    dx = xx - float(centre[0])
    dy = yy - float(centre[1])
    remove = np.zeros(work.shape, dtype=bool)
    for angle in np.atleast_1d(np.asarray(angles, dtype=float)):
        along = dx * np.cos(angle) + dy * np.sin(angle)
        across = -dx * np.sin(angle) + dy * np.cos(angle)
        remove |= (along >= 0) & (np.abs(across) <= width / 2.0)
    work &= ~remove
    return work


def _fill(mask: np.ndarray, poly: np.ndarray) -> None:
    """Fill one closed polygon into ``mask`` with the even-odd rule."""
    height, width = mask.shape
    xs, ys = poly[:, 0], poly[:, 1]
    previous_x, previous_y = np.roll(xs, -1), np.roll(ys, -1)
    first_row = max(0, int(np.floor(ys.min())))
    last_row = min(height - 1, int(np.ceil(ys.max())))
    for row in range(first_row, last_row + 1):
        yc = float(row)
        crossing = ((ys <= yc) & (previous_y > yc)) | ((previous_y <= yc) & (ys > yc))
        if not crossing.any():
            continue
        x0, y0 = xs[crossing], ys[crossing]
        x1, y1 = previous_x[crossing], previous_y[crossing]
        hits = np.sort(x0 + (yc - y0) * (x1 - x0) / (y1 - y0))
        for k in range(0, len(hits) - 1, 2):
            start = max(0, int(np.ceil(hits[k])))
            stop = min(width, int(np.ceil(hits[k + 1])))
            if stop > start:
                mask[row, start:stop] = True


def _to_units(poly, cx, cy, half):
    points = np.asarray(poly, dtype=float)
    return np.stack([(points[:, 0] - cx) / half, (points[:, 1] + cy) / half], axis=1)


def _to_pixels(poly, cx, cy, half, scale_x, scale_y):
    points = np.asarray(poly, dtype=float)
    x = (points[:, 0] * half + cx) * scale_x
    y = (cy - points[:, 1] * half) * scale_y
    return np.stack([x, y], axis=1)


def _colour(palette, name):
    if isinstance(name, str):
        if name not in palette:
            raise KeyError(f"colour {name!r} is not in the palette")
        return tuple(int(c) for c in palette[name])
    return tuple(int(c) for c in name)


def trace_layers(img, palette, order, background="white", tol: float = 1.5,
                 min_area: float = 30, frame=None) -> Shape:
    """Trace a blueprint into a Shape, with one layer per material in ``order``.

    Match the image to ``palette``, which maps material names to RGB. ``order``
    lists material names from the largest region to the smallest; a region's
    holes become layers of the material underneath them, painted after it, so
    every layer holds only simply connected outlines. Polygons are simplified
    by ``tol`` pixels and regions smaller than ``min_area`` are dropped. The
    ``frame`` (a pixel box ``(x0, y0, x1, y1)``, or the bounding box of
    everything that is not ``background``) maps to ``-1 .. 1`` on its longer
    side, centred, with y up.
    """
    pixels = np.asarray(img)
    if pixels.ndim != 3 or pixels.shape[2] not in (3, 4):
        raise ValueError("images must be (H, W, 3) or (H, W, 4)")
    names = list(palette)
    labels = classify(pixels, palette)
    if isinstance(background, str) and background in palette:
        background_label = names.index(background)
        content = labels != background_label
    else:
        content = np.ones(labels.shape, dtype=bool)

    if frame is None:
        rows, cols = np.nonzero(content)
        if len(rows):
            x0, x1 = float(cols.min()) - 0.5, float(cols.max()) + 0.5
            y0, y1 = float(rows.min()) - 0.5, float(rows.max()) + 0.5
        else:
            x0, x1 = 0.0, float(labels.shape[1])
            y0, y1 = 0.0, float(labels.shape[0])
    else:
        x0, y0, x1, y1 = (float(value) for value in frame)
    cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
    half = max(x1 - x0, y1 - y0) / 2.0 or 1.0

    layers: list[Layer] = []
    for name in order:
        if name not in palette:
            raise KeyError(f"material {name!r} is not in the palette")
        mask = labels == names.index(name)
        if int(mask.sum()) < min_area:
            continue
        outer, holes = contours(mask, holes=True)
        polygons = []
        for loop in outer:
            simple = simplify(loop, tol)
            if abs(_area(simple)) >= min_area:
                polygons.append(_to_units(simple, cx, cy, half))
        if polygons:
            layers.append(Layer(name, polygons))
        for loop in holes:
            simple = simplify(loop, tol)
            if abs(_area(simple)) < min_area:
                continue
            polygon = _to_units(simple, cx, cy, half)
            centre = polygon.mean(axis=0)
            col = int(round(centre[0] * half + cx))
            row = int(round(cy - centre[1] * half))
            col = min(max(col, 0), labels.shape[1] - 1)
            row = min(max(row, 0), labels.shape[0] - 1)
            underneath = names[int(labels[row, col])]
            if underneath == name:
                underneath = background if isinstance(background, str) else "white"
            layers.append(Layer(underneath, [polygon]))

    return Shape(layers=layers, anchor=(0.0, 0.0),
                 meta={"frame": (cx, cy, half), "size": (int(pixels.shape[0]), int(pixels.shape[1]))})


def redraw(shape: Shape, size, palette, background="white") -> np.ndarray:
    """Rasterise a Shape back to an RGB image of ``size=(w, h)``.

    Layers paint in order. A shape traced by :func:`trace_layers` is redrawn
    onto its original frame; any other shape is centred in the canvas.
    """
    width, height = int(size[0]), int(size[1])
    canvas = np.empty((height, width, 3), dtype=np.uint8)
    canvas[...] = _colour(palette, background)
    frame = shape.meta.get("frame")
    source_size = shape.meta.get("size")
    if frame is not None and source_size is not None:
        cx, cy, half = frame
        scale_x = width / source_size[1]
        scale_y = height / source_size[0]
    else:
        cx, cy = width / 2.0, height / 2.0
        half = min(width, height) / 2.0
        scale_x = scale_y = 1.0
    for layer in shape.layers:
        colour = _colour(palette, layer.material)
        mask = np.zeros((height, width), dtype=bool)
        for polygon in layer.polygons:
            _fill(mask, _to_pixels(polygon, cx, cy, half, scale_x, scale_y))
        canvas[mask] = colour
    return canvas


def iou(a, b) -> float:
    """Intersection over union of two boolean masks; two empty masks score 1.0."""
    first = np.asarray(a, dtype=bool)
    second = np.asarray(b, dtype=bool)
    union = np.logical_or(first, second).sum()
    if union == 0:
        return 1.0
    return float(np.logical_and(first, second).sum() / union)
