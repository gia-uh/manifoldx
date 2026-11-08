"""The district's survey and its tile builder, importable by name."""

import numpy as np

from manifoldx.streaming import MeshData

CELL = 20.0
SEGMENTS = 4
PLAN = {}  # (i, j) -> {"height": metres, "tint": linear rgb}, filled by the survey


def district_rows(j_values, i_values):
    """One row of the district's plan per yield; the work stays off the frame."""
    for j in j_values:
        row = {}
        for i in i_values:
            sample = np.random.default_rng(1_000 * (i + 10) + (j + 10)).random((64, 64))
            inside_plaza = i in (-1, 0) and j in (-1, 0)
            row[(i, j)] = {
                "height": 0.0 if inside_plaza else 6.0 + 10.0 * float(sample.mean()),
                "tint": tuple(float(c) for c in sample.mean(axis=0)[:3]),
            }
        yield row


def _box(cx, cz, width, depth, height, tint):
    x0, x1 = cx - width / 2, cx + width / 2
    z0, z1 = cz - depth / 2, cz + depth / 2
    faces = [
        ([(x0, 0, z1), (x1, 0, z1), (x1, height, z1), (x0, height, z1)], (0, 0, 1)),
        ([(x1, 0, z0), (x0, 0, z0), (x0, height, z0), (x1, height, z0)], (0, 0, -1)),
        ([(x1, 0, z1), (x1, 0, z0), (x1, height, z0), (x1, height, z1)], (1, 0, 0)),
        ([(x0, 0, z0), (x0, 0, z1), (x0, height, z1), (x0, height, z0)], (-1, 0, 0)),
        ([(x0, height, z1), (x1, height, z1), (x1, height, z0), (x0, height, z0)], (0, 1, 0)),
        ([(x0, 0, z0), (x1, 0, z0), (x1, 0, z1), (x0, 0, z1)], (0, -1, 0)),
    ]
    positions, normals, colors, indices = [], [], [], []
    for corners, normal in faces:
        base = len(positions)
        positions.extend(corners)
        normals.extend([normal] * 4)
        colors.extend([tint] * 4)
        indices.extend([base, base + 1, base + 2, base, base + 2, base + 3])
    return positions, normals, colors, indices


def build_cell(i, j):
    """Cell (i, j) as MeshData relative to its corner."""
    n = SEGMENTS + 1
    line = np.linspace(0.0, CELL, n)
    plot = PLAN.get((i, j), {"height": 0.0, "tint": (0.5, 0.5, 0.5)})
    inside_plaza = i in (-1, 0) and j in (-1, 0)
    ground = (0.36, 0.34, 0.30) if inside_plaza else (0.28, 0.26, 0.22)
    positions = [[x, 0.0, z] for z in line for x in line]
    normals = [[0.0, 1.0, 0.0] for _ in positions]
    colors = [ground for _ in positions]
    q = np.arange(SEGMENTS)
    a = (q[:, None] * n + q[None, :]).ravel()
    indices = np.stack([a, a + n, a + n + 1, a, a + n + 1, a + 1], -1).ravel().tolist()
    if plot["height"] > 0:
        tint = (
            0.26 + 0.10 * plot["tint"][0],
            0.22 + 0.08 * plot["tint"][1],
            0.18 + 0.06 * plot["tint"][2],
        )
        p, nr, c, idx = _box(CELL / 2, CELL / 2, CELL - 6, CELL - 6, plot["height"], tint)
        base = len(positions)
        positions += p
        normals += nr
        colors += c
        indices += [base + k for k in idx]
    return MeshData(
        np.asarray(positions, np.float32),
        np.asarray(normals, np.float32),
        np.asarray(colors, np.float32),
        np.asarray(indices, np.uint32),
    )
