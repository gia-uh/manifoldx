"""Procedural primitive generators returning modeling.Mesh values."""

from __future__ import annotations

import numpy as np

from manifoldx.modeling.mesh import Mesh


def icosphere(subdivisions: int = 2, radius: float = 1.0) -> Mesh:
    """Geodesic sphere from a subdivided icosahedron (uniform-ish triangles)."""
    t = (1.0 + 5.0**0.5) / 2.0
    verts = np.array(
        [
            [-1, t, 0], [1, t, 0], [-1, -t, 0], [1, -t, 0],
            [0, -1, t], [0, 1, t], [0, -1, -t], [0, 1, -t],
            [t, 0, -1], [t, 0, 1], [-t, 0, -1], [-t, 0, 1],
        ],
        dtype=np.float64,
    )
    faces = np.array(
        [
            [0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11],
            [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
            [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9],
            [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1],
        ],
        dtype=np.int64,
    )

    for _ in range(subdivisions):
        verts, faces = _subdivide_midpoint(verts, faces)

    verts = verts / np.linalg.norm(verts, axis=1, keepdims=True) * radius
    return Mesh(
        positions=verts.astype(np.float32),
        faces=faces.astype(np.uint32),
    )


def _subdivide_midpoint(verts: np.ndarray, faces: np.ndarray):
    """Split each triangle into 4 via edge midpoints, sharing midpoints across edges."""
    verts = list(map(tuple, verts))
    cache: dict[tuple[int, int], int] = {}

    def midpoint(a: int, b: int) -> int:
        key = (a, b) if a < b else (b, a)
        if key in cache:
            return cache[key]
        va, vb = np.asarray(verts[a]), np.asarray(verts[b])
        verts.append(tuple((va + vb) / 2.0))
        cache[key] = len(verts) - 1
        return cache[key]

    new_faces = []
    for a, b, c in faces:
        ab, bc, ca = midpoint(a, b), midpoint(b, c), midpoint(c, a)
        new_faces += [[a, ab, ca], [b, bc, ab], [c, ca, bc], [ab, bc, ca]]

    return np.array(verts, dtype=np.float64), np.array(new_faces, dtype=np.int64)


def box(width: float = 1.0, height: float = 1.0, depth: float = 1.0) -> Mesh:
    hx, hy, hz = width / 2, height / 2, depth / 2
    corners = np.array(
        [[-hx, -hy, -hz], [hx, -hy, -hz], [hx, hy, -hz], [-hx, hy, -hz],
         [-hx, -hy, hz], [hx, -hy, hz], [hx, hy, hz], [-hx, hy, hz]],
        dtype=np.float32,
    )
    quads = [
        (0, 1, 2, 3), (5, 4, 7, 6), (4, 0, 3, 7),
        (1, 5, 6, 2), (3, 2, 6, 7), (4, 5, 1, 0),
    ]
    faces = []
    for a, b, c, d in quads:
        faces += [[a, c, b], [a, d, c]]   # CCW from outside (outward normals)
    return Mesh(positions=corners, faces=np.array(faces, dtype=np.uint32))


def box_between(a, b) -> Mesh:
    """An axis-aligned box spanning two opposite corners, given as ``(x, y, z)`` points."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    low, high = np.minimum(a, b), np.maximum(a, b)
    size = high - low
    return box(float(size[0]), float(size[1]), float(size[2])).translate((low + high) / 2)


def plane(width: float = 1.0, depth: float = 1.0, segments: int = 1) -> Mesh:
    n = segments + 1
    xs = np.linspace(-width / 2, width / 2, n)
    zs = np.linspace(-depth / 2, depth / 2, n)
    gx, gz = np.meshgrid(xs, zs)
    positions = np.stack([gx.ravel(), np.zeros(n * n), gz.ravel()], axis=1).astype(np.float32)
    faces = []
    for i in range(segments):
        for j in range(segments):
            a = i * n + j
            b = a + 1
            c = a + n
            d = c + 1
            faces += [[a, c, b], [b, c, d]]
    return Mesh(positions=positions, faces=np.array(faces, dtype=np.uint32))


def cylinder(radius: float = 1.0, height: float = 2.0, segments: int = 32) -> Mesh:
    ang = np.linspace(0, 2 * np.pi, segments, endpoint=False)
    ring = np.stack([np.cos(ang) * radius, np.sin(ang) * radius], axis=1)
    top = np.column_stack([ring[:, 0], np.full(segments, height / 2), ring[:, 1]])
    bot = np.column_stack([ring[:, 0], np.full(segments, -height / 2), ring[:, 1]])
    center_top = [0, height / 2, 0]
    center_bot = [0, -height / 2, 0]
    positions = np.vstack([top, bot, center_top, center_bot]).astype(np.float32)
    ct, cb = 2 * segments, 2 * segments + 1
    faces = []
    for i in range(segments):
        n = (i + 1) % segments
        faces += [[i, n, segments + i], [n, segments + n, segments + i]]  # side
        faces += [[ct, n, i]]                                             # top cap
        faces += [[cb, segments + i, segments + n]]                       # bottom cap
    return Mesh(positions=positions, faces=np.array(faces, dtype=np.uint32))


def torus(major: float = 1.0, minor: float = 0.35,
          major_segments: int = 32, minor_segments: int = 16) -> Mesh:
    u = np.linspace(0, 2 * np.pi, major_segments, endpoint=False)
    v = np.linspace(0, 2 * np.pi, minor_segments, endpoint=False)
    uu, vv = np.meshgrid(u, v, indexing="ij")
    x = (major + minor * np.cos(vv)) * np.cos(uu)
    y = minor * np.sin(vv)
    z = (major + minor * np.cos(vv)) * np.sin(uu)
    positions = np.stack([x.ravel(), y.ravel(), z.ravel()], axis=1).astype(np.float32)
    faces = []
    for i in range(major_segments):
        for j in range(minor_segments):
            a = i * minor_segments + j
            b = ((i + 1) % major_segments) * minor_segments + j
            c = i * minor_segments + (j + 1) % minor_segments
            d = ((i + 1) % major_segments) * minor_segments + (j + 1) % minor_segments
            faces += [[a, c, b], [b, c, d]]   # CCW from outside (outward normals)
    return Mesh(positions=positions, faces=np.array(faces, dtype=np.uint32))
