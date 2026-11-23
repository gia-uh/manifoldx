"""S-b2: an irregular cell, in Python.

A street grid with a ninth of its edges missing, BFS distances from the centre,
four candidate lots per block (kept by the block's hash), building heights that
read the BFS distance, and lamp rows that carry it. The same algorithm is in
``native/src/lib.rs`` (``build_cell_irregular``); the spike compares the bytes.

``build_cell_irregular`` is the current shape: Python loops and lists.
``build_cell_irregular_numpy`` vectorizes the geometry and keeps the graph and the
BFS in Python, the best a numpy author can do without a graph library.
"""

from __future__ import annotations

from collections import deque

import numpy as np

BLOCK = 25.0
HALF = 4.0
MASK64 = 0xFFFFFFFFFFFFFFFF
MAX_DIST = 0xFFFFFFFF

ROW_DTYPE = np.dtype([("x", "<f4"), ("z", "<f4"), ("kind", "<u4"), ("dist", "<u4")])


def _hash_edge(i, j, direction, seed):
    return (i * 73_856_093 + j * 19_349_663 + direction * 83_492_791 + seed) & MASK64


def _hash_block(i, j, seed):
    return (i * 83_492_791 + j * 73_856_093 + seed) & MASK64


def _graph(n, seed):
    nodes = n + 1
    adjacency = [[] for _ in range(nodes * nodes)]
    horizontal, vertical = [], []
    for i in range(nodes):
        for j in range(nodes):
            if i < n and _hash_edge(i, j, 0, seed) % 9 != 0:
                a, b = i * nodes + j, (i + 1) * nodes + j
                adjacency[a].append(b)
                adjacency[b].append(a)
                horizontal.append((i, j))
            if j < n and _hash_edge(i, j, 1, seed) % 9 != 0:
                a, b = i * nodes + j, i * nodes + (j + 1)
                adjacency[a].append(b)
                adjacency[b].append(a)
                vertical.append((i, j))
    return adjacency, horizontal, vertical


def _distances(adjacency, n):
    nodes = n + 1
    dist = [MAX_DIST] * (nodes * nodes)
    start = (n // 2) * nodes + n // 2
    dist[start] = 0
    queue = deque([start])
    while queue:
        node = queue.popleft()
        d = dist[node]
        for neighbour in adjacency[node]:
            if dist[neighbour] == MAX_DIST:
                dist[neighbour] = d + 1
                queue.append(neighbour)
    return dist


def _quad(positions, normals, indices, verts, normal):
    base = len(positions) // 3
    for vertex in verts:
        positions.extend(vertex)
        normals.extend(normal)
    indices.extend((base, base + 1, base + 2, base, base + 2, base + 3))


def _box(positions, normals, indices, x0, x1, z0, z1, height):
    _quad(positions, normals, indices, [(x0, 0.0, z0), (x1, 0.0, z0), (x1, 0.0, z1), (x0, 0.0, z1)], (0.0, -1.0, 0.0))
    _quad(positions, normals, indices, [(x0, height, z0), (x1, height, z0), (x1, height, z1), (x0, height, z1)], (0.0, 1.0, 0.0))
    _quad(positions, normals, indices, [(x0, 0.0, z0), (x0, height, z0), (x0, height, z1), (x0, 0.0, z1)], (-1.0, 0.0, 0.0))
    _quad(positions, normals, indices, [(x1, 0.0, z0), (x1, height, z0), (x1, height, z1), (x1, 0.0, z1)], (1.0, 0.0, 0.0))
    _quad(positions, normals, indices, [(x0, 0.0, z0), (x1, 0.0, z0), (x1, height, z0), (x0, height, z0)], (0.0, 0.0, -1.0))
    _quad(positions, normals, indices, [(x0, 0.0, z1), (x1, 0.0, z1), (x1, height, z1), (x0, height, z1)], (0.0, 0.0, 1.0))


def build_cell_irregular(x: float = 0.0, y: float = 0.0, size: float = 1000.0, lod: int = 0, seed: int = 0) -> dict:
    n = int(size // BLOCK)
    nodes = n + 1
    adjacency, horizontal, vertical = _graph(n, seed)
    dist = _distances(adjacency, n)
    y_road = 0.02

    rp, rn, ri = [], [], []
    for i, j in horizontal:
        x0, x1, z = i * BLOCK, (i + 1) * BLOCK, j * BLOCK
        _quad(rp, rn, ri, [(x0, y_road, z - HALF), (x1, y_road, z - HALF),
                           (x1, y_road, z + HALF), (x0, y_road, z + HALF)], (0.0, 1.0, 0.0))
    for i, j in vertical:
        xx, z0, z1 = i * BLOCK, j * BLOCK, (j + 1) * BLOCK
        _quad(rp, rn, ri, [(xx - HALF, y_road, z0), (xx + HALF, y_road, z0),
                           (xx + HALF, y_road, z1), (xx - HALF, y_road, z1)], (0.0, 1.0, 0.0))

    bp, bn, bi = [], [], []
    for k in range(4):
        for i in range(n):
            for j in range(n):
                h = _hash_block(i, j, seed)
                if h % 13 == 0 or (h >> (k + 8)) & 1 == 0:
                    continue
                d = dist[i * nodes + j]
                d = 0 if d == MAX_DIST else d
                height = 4.0 + float((d + ((h >> (4 * k)) & 31)) % 28)
                qx, qz = k % 2, k // 2
                x0 = i * BLOCK + qx * 12.5 + 1.5
                x1 = x0 + 9.5
                z0 = j * BLOCK + qz * 12.5 + 1.5
                z1 = z0 + 9.5
                _box(bp, bn, bi, x0, x1, z0, z1, height)

    rows = []
    for i in range(nodes):
        for j in range(nodes):
            d = dist[i * nodes + j]
            if d == MAX_DIST or d % 3 != 0:
                continue
            rows.append((i * BLOCK, j * BLOCK, 1, d))
    row_array = np.empty(len(rows), dtype=ROW_DTYPE)
    for k, row in enumerate(rows):
        row_array[k] = row

    return {
        "meshes": [
            {"positions": np.asarray(rp, np.float32).reshape(-1, 3),
             "normals": np.asarray(rn, np.float32).reshape(-1, 3),
             "indices": np.asarray(ri, np.uint32), "material": 0},
            {"positions": np.asarray(bp, np.float32).reshape(-1, 3),
             "normals": np.asarray(bn, np.float32).reshape(-1, 3),
             "indices": np.asarray(bi, np.uint32), "material": 1},
        ],
        "rows": row_array.tobytes(),
        "row_bytes": row_array.dtype.itemsize,
        "archetype": 0,
    }


def _hash_edge_vec(i, j, direction, seed):
    return (i.astype(np.uint64) * np.uint64(73_856_093)
            + j.astype(np.uint64) * np.uint64(19_349_663)
            + np.uint64(direction * 83_492_791 + seed)) & np.uint64(MASK64)


def _hash_block_vec(i, j, seed):
    return (i.astype(np.uint64) * np.uint64(83_492_791)
            + j.astype(np.uint64) * np.uint64(73_856_093)
            + np.uint64(seed)) & np.uint64(MASK64)


def _box_faces(x0, x1, z0, z1, height):
    zero = np.zeros_like(x0)

    def v(px, py, pz):
        return np.stack([px, py, pz], axis=1)

    return np.stack([
        np.stack([v(x0, zero, z0), v(x1, zero, z0), v(x1, zero, z1), v(x0, zero, z1)], axis=1),
        np.stack([v(x0, height, z0), v(x1, height, z0), v(x1, height, z1), v(x0, height, z1)], axis=1),
        np.stack([v(x0, zero, z0), v(x0, height, z0), v(x0, height, z1), v(x0, zero, z1)], axis=1),
        np.stack([v(x1, zero, z0), v(x1, height, z0), v(x1, height, z1), v(x1, zero, z1)], axis=1),
        np.stack([v(x0, zero, z0), v(x1, zero, z0), v(x1, height, z0), v(x0, height, z0)], axis=1),
        np.stack([v(x0, zero, z1), v(x1, zero, z1), v(x1, height, z1), v(x0, height, z1)], axis=1),
    ], axis=1)


def build_cell_irregular_numpy(size: float = 1000.0, seed: int = 0) -> dict:
    n = int(size // BLOCK)
    nodes = n + 1
    adjacency, horizontal, vertical = _graph(n, seed)
    dist = _distances(adjacency, n)
    y_road = np.float32(0.02)

    # Roads: all horizontal edges row-major, then all vertical ones.
    hi = np.arange(n, dtype=np.float32)[:, None]
    hj = np.arange(nodes, dtype=np.float32)[None, :]
    hmask = (_hash_edge_vec(np.arange(n)[:, None], np.arange(nodes)[None, :], 0, seed) % np.uint64(9)) != 0
    hi, hj = np.nonzero(hmask)
    hx0, hx1, hz = hi * np.float32(BLOCK), (hi + 1) * np.float32(BLOCK), hj * np.float32(BLOCK)
    hquads = np.stack([
        np.stack([hx0, np.full_like(hx0, y_road), hz - HALF], axis=1),
        np.stack([hx1, np.full_like(hx0, y_road), hz - HALF], axis=1),
        np.stack([hx1, np.full_like(hx0, y_road), hz + HALF], axis=1),
        np.stack([hx0, np.full_like(hx0, y_road), hz + HALF], axis=1),
    ], axis=1)

    vi = np.arange(nodes, dtype=np.float32)[:, None]
    vj = np.arange(n, dtype=np.float32)[None, :]
    vmask = (_hash_edge_vec(np.arange(nodes)[:, None], np.arange(n)[None, :], 1, seed) % np.uint64(9)) != 0
    vi, vj = np.nonzero(vmask)
    vx, vz0, vz1 = vi * np.float32(BLOCK), vj * np.float32(BLOCK), (vj + 1) * np.float32(BLOCK)
    vquads = np.stack([
        np.stack([vx - HALF, np.full_like(vx, y_road), vz0], axis=1),
        np.stack([vx + HALF, np.full_like(vx, y_road), vz0], axis=1),
        np.stack([vx + HALF, np.full_like(vx, y_road), vz1], axis=1),
        np.stack([vx - HALF, np.full_like(vx, y_road), vz1], axis=1),
    ], axis=1)

    quads = np.concatenate([hquads, vquads], axis=0)
    rp = quads.reshape(-1, 3).astype(np.float32)
    rn = np.tile(np.array([0.0, 1.0, 0.0], np.float32), (len(rp), 1))
    base = (np.arange(len(quads), dtype=np.uint32) * 4)[:, None]
    offs = np.array([0, 1, 2, 0, 2, 3], np.uint32)
    ri = (base + offs).reshape(-1)

    # Lots: four per block, k-major, exactly the loop's order.
    ii, jj = np.meshgrid(np.arange(n, dtype=np.uint64), np.arange(n, dtype=np.uint64), indexing="ij")
    h = _hash_block_vec(ii, jj, seed)
    park = (h % np.uint64(13)) == 0
    dist_grid = np.asarray(dist, dtype=np.uint32).reshape(nodes, nodes)
    normals = np.array([
        [0.0, -1.0, 0.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 0.0, 1.0],
    ], np.float32)

    pieces = []
    vertex_offset = 0
    for k in range(4):
        keep = (~park) & (((h >> np.uint64(k + 8)) & np.uint64(1)) == 1)
        bi_, bj_ = ii[keep], jj[keep]
        bh = h[keep]
        d = dist_grid[bi_, bj_]
        d = np.where(d == np.uint32(MAX_DIST), np.uint32(0), d)
        height = (4.0 + ((d + ((bh >> np.uint64(4 * k)) & np.uint64(31))).astype(np.float32) % 28)).astype(np.float32)
        qx, qz = k % 2, k // 2
        x0 = (bi_.astype(np.float32) * BLOCK + qx * 12.5 + 1.5).astype(np.float32)
        x1 = (x0 + 9.5).astype(np.float32)
        z0 = (bj_.astype(np.float32) * BLOCK + qz * 12.5 + 1.5).astype(np.float32)
        z1 = (z0 + 9.5).astype(np.float32)
        faces = _box_faces(x0, x1, z0, z1, height)
        p = faces.reshape(-1, 3).astype(np.float32)
        nm = np.broadcast_to(normals[:, None, :], (len(p) // 24, 6, 4, 3)).reshape(-1, 3)
        base = (np.arange(len(p) // 24, dtype=np.uint32) * 24)[:, None, None]
        face_offsets = (np.arange(6, dtype=np.uint32) * 4)[None, :, None]
        idx = (base + face_offsets + offs[None, None, :]).reshape(-1) + np.uint32(vertex_offset)
        pieces.append((p, nm, idx))
        vertex_offset += len(p)

    bp = np.concatenate([piece[0] for piece in pieces]) if pieces else np.empty((0, 3), np.float32)
    bn = np.concatenate([piece[1] for piece in pieces]) if pieces else np.empty((0, 3), np.float32)
    bi = np.concatenate([piece[2] for piece in pieces]) if pieces else np.empty(0, np.uint32)

    mask = (dist_grid != np.uint32(MAX_DIST)) & ((dist_grid % np.uint32(3)) == 0)
    li, lj = np.nonzero(mask)
    rows = np.empty(len(li), dtype=ROW_DTYPE)
    rows["x"] = (li * BLOCK).astype(np.float32)
    rows["z"] = (lj * BLOCK).astype(np.float32)
    rows["kind"] = 1
    rows["dist"] = dist_grid[li, lj]

    return {
        "meshes": [
            {"positions": rp, "normals": rn, "indices": ri, "material": 0},
            {"positions": bp, "normals": bn, "indices": bi, "material": 1},
        ],
        "rows": rows.tobytes(),
        "row_bytes": rows.dtype.itemsize,
        "archetype": 0,
    }
