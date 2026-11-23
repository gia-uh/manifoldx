"""The Python builder S-b compares against: the same 1 km² cell, block by block.

The algorithm is the contract; ``native/src/lib.rs`` implements it in Rust with
the same order of operations and the same f32 values. The spike compares the
bytes, so a change here must be mirrored there.
"""

from __future__ import annotations

import numpy as np

BLOCK = 25.0
ROAD = 8.0
HALF = ROAD / 2.0
INSET = HALF + 2.0

ROW_DTYPE = np.dtype([("x", "<f4"), ("z", "<f4"), ("kind", "<u4")])


def _quad(positions, normals, indices, verts, normal):
    base = len(positions) // 3
    for vertex in verts:
        positions.extend(vertex)
        normals.extend(normal)
    indices.extend((base, base + 1, base + 2, base, base + 2, base + 3))


def build_cell(x: float = 0.0, y: float = 0.0, size: float = 1000.0, lod: int = 0, seed: int = 0) -> dict:
    """The cell the native builder makes, as numpy arrays and packed entity rows."""
    n = int(size // BLOCK)
    y_road = 0.02

    rp, rn, ri = [], [], []
    for i in range(n + 1):
        z = i * BLOCK
        _quad(
            rp, rn, ri,
            [(0.0, y_road, z - HALF), (size, y_road, z - HALF),
             (size, y_road, z + HALF), (0.0, y_road, z + HALF)],
            (0.0, 1.0, 0.0),
        )
        xx = i * BLOCK
        _quad(
            rp, rn, ri,
            [(xx - HALF, y_road, 0.0), (xx + HALF, y_road, 0.0),
             (xx + HALF, y_road, size), (xx - HALF, y_road, size)],
            (0.0, 1.0, 0.0),
        )

    bp, bn, bi = [], [], []
    for i in range(n):
        for j in range(n):
            h = (i * 73_856_093 + j * 19_349_663 + seed) & 0xFFFFFFFFFFFFFFFF
            if h % 11 == 0:
                continue
            height = 8.0 + float(h % 32)
            x0 = i * BLOCK + INSET
            x1 = (i + 1) * BLOCK - INSET
            z0 = j * BLOCK + INSET
            z1 = (j + 1) * BLOCK - INSET

            _quad(bp, bn, bi, [(x0, 0.0, z0), (x1, 0.0, z0), (x1, 0.0, z1), (x0, 0.0, z1)], (0.0, -1.0, 0.0))
            _quad(bp, bn, bi, [(x0, height, z0), (x1, height, z0), (x1, height, z1), (x0, height, z1)], (0.0, 1.0, 0.0))
            _quad(bp, bn, bi, [(x0, 0.0, z0), (x0, height, z0), (x0, height, z1), (x0, 0.0, z1)], (-1.0, 0.0, 0.0))
            _quad(bp, bn, bi, [(x1, 0.0, z0), (x1, height, z0), (x1, height, z1), (x1, 0.0, z1)], (1.0, 0.0, 0.0))
            _quad(bp, bn, bi, [(x0, 0.0, z0), (x1, 0.0, z0), (x1, height, z0), (x0, height, z0)], (0.0, 0.0, -1.0))
            _quad(bp, bn, bi, [(x0, 0.0, z1), (x1, 0.0, z1), (x1, height, z1), (x0, height, z1)], (0.0, 0.0, 1.0))

    rows = []
    for i in range(n + 1):
        for j in range(n + 1):
            if (i + j + seed) % 3 != 0:
                continue
            rows.append((i * BLOCK, j * BLOCK, 1))
    row_array = np.empty(len(rows), dtype=ROW_DTYPE)
    for k, row in enumerate(rows):
        row_array[k] = row

    return {
        "meshes": [
            {
                "positions": np.asarray(rp, dtype=np.float32).reshape(-1, 3),
                "normals": np.asarray(rn, dtype=np.float32).reshape(-1, 3),
                "indices": np.asarray(ri, dtype=np.uint32),
                "material": 0,
            },
            {
                "positions": np.asarray(bp, dtype=np.float32).reshape(-1, 3),
                "normals": np.asarray(bn, dtype=np.float32).reshape(-1, 3),
                "indices": np.asarray(bi, dtype=np.uint32),
                "material": 1,
            },
        ],
        "rows": row_array.tobytes(),
        "row_bytes": row_array.dtype.itemsize,
        "archetype": 0,
    }


def build_cell_numpy(size: float = 1000.0, seed: int = 0) -> dict:
    """The same cell, vectorized: the floor of what numpy can do with a regular grid.

    This is not the builder the current streamer uses; it is the best case a Python
    author could reach without leaving numpy, kept so S-b reports both speed-ups.
    """
    n = int(size // BLOCK)

    i = np.arange(n + 1, dtype=np.float32)
    z = i * BLOCK
    zeros = np.zeros_like(z)
    full = np.full_like(z, size)

    horizontal = np.stack([
        np.stack([zeros, np.full_like(z, 0.02), z - HALF], axis=1),
        np.stack([full, np.full_like(z, 0.02), z - HALF], axis=1),
        np.stack([full, np.full_like(z, 0.02), z + HALF], axis=1),
        np.stack([zeros, np.full_like(z, 0.02), z + HALF], axis=1),
    ], axis=1)
    x = i * BLOCK
    vertical = np.stack([
        np.stack([x - HALF, np.full_like(z, 0.02), zeros], axis=1),
        np.stack([x + HALF, np.full_like(z, 0.02), zeros], axis=1),
        np.stack([x + HALF, np.full_like(z, 0.02), full], axis=1),
        np.stack([x - HALF, np.full_like(z, 0.02), full], axis=1),
    ], axis=1)
    road_quads = np.stack([horizontal, vertical], axis=1)
    rp = road_quads.reshape(-1, 3)
    rn = np.tile(np.array([0.0, 1.0, 0.0], np.float32), (len(rp), 1))
    base = (np.arange(len(rp) // 4, dtype=np.uint32) * 4)[:, None]
    offs = np.array([0, 1, 2, 0, 2, 3], np.uint32)
    ri = (base + offs).reshape(-1)

    ii, jj = np.meshgrid(np.arange(n, dtype=np.uint64), np.arange(n, dtype=np.uint64), indexing="ij")
    h = (ii * np.uint64(73_856_093) + jj * np.uint64(19_349_663) + np.uint64(seed)) & np.uint64(0xFFFFFFFFFFFFFFFF)
    keep = (h % np.uint64(11)) != 0
    ii, jj, h = ii[keep], jj[keep], h[keep]

    height = (8.0 + (h % np.uint64(32)).astype(np.float32)).astype(np.float32)
    x0 = (ii.astype(np.float32) * BLOCK + INSET).astype(np.float32)
    x1 = ((ii + 1).astype(np.float32) * BLOCK - INSET).astype(np.float32)
    z0 = (jj.astype(np.float32) * BLOCK + INSET).astype(np.float32)
    z1 = ((jj + 1).astype(np.float32) * BLOCK - INSET).astype(np.float32)
    zero = np.zeros_like(x0)

    def vertex(px, py, pz):
        return np.stack([px, py, pz], axis=1)

    faces = np.stack([
        np.stack([vertex(x0, zero, z0), vertex(x1, zero, z0), vertex(x1, zero, z1), vertex(x0, zero, z1)], axis=1),
        np.stack([vertex(x0, height, z0), vertex(x1, height, z0), vertex(x1, height, z1), vertex(x0, height, z1)], axis=1),
        np.stack([vertex(x0, zero, z0), vertex(x0, height, z0), vertex(x0, height, z1), vertex(x0, zero, z1)], axis=1),
        np.stack([vertex(x1, zero, z0), vertex(x1, height, z0), vertex(x1, height, z1), vertex(x1, zero, z1)], axis=1),
        np.stack([vertex(x0, zero, z0), vertex(x1, zero, z0), vertex(x1, height, z0), vertex(x0, height, z0)], axis=1),
        np.stack([vertex(x0, zero, z1), vertex(x1, zero, z1), vertex(x1, height, z1), vertex(x0, height, z1)], axis=1),
    ], axis=1)
    bp = faces.reshape(-1, 3).astype(np.float32)
    normals = np.array([
        [0.0, -1.0, 0.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 0.0, 1.0],
    ], np.float32)
    bn = np.broadcast_to(normals[:, None, :], (len(bp) // 24, 6, 4, 3)).reshape(-1, 3)
    base = (np.arange(len(bp) // 24, dtype=np.uint32) * 24)[:, None, None]
    face_offsets = (np.arange(6, dtype=np.uint32) * 4)[None, :, None]
    bi = (base + face_offsets + offs[None, None, :]).reshape(-1)

    li, lj = np.meshgrid(np.arange(n + 1), np.arange(n + 1), indexing="ij")
    mask = ((li + lj + seed) % 3) == 0
    rows = np.empty(int(mask.sum()), dtype=ROW_DTYPE)
    rows["x"] = (li[mask] * BLOCK).astype(np.float32)
    rows["z"] = (lj[mask] * BLOCK).astype(np.float32)
    rows["kind"] = 1

    return {
        "meshes": [
            {"positions": rp, "normals": rn, "indices": ri, "material": 0},
            {"positions": bp, "normals": bn, "indices": bi, "material": 1},
        ],
        "rows": rows.tobytes(),
        "row_bytes": rows.dtype.itemsize,
        "archetype": 0,
    }
