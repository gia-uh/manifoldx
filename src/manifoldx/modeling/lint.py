"""Deterministic checks of a model built from parts: is each mesh closed, which parts touch, what floats.

An agent that builds a model from drawings cannot see it; these checks tell it, in numbers, what came out wrong:
a part that floats away from the rest, a mesh with holes or inside out. numpy only.
"""

from __future__ import annotations

import numpy as np

from manifoldx.modeling.mesh import Mesh


def mesh_report(mesh: Mesh) -> dict:
    """Open edges (on one face only), non-manifold edges (on three or more), degenerate faces and the volume.

    A closed mesh has no open edges; a negative volume means its faces look inward. Vertices at the same place
    are welded first, so the seams of a texture atlas (a vertex split to carry two UVs) are not counted as holes.
    """
    p, inverse = np.unique(np.round(np.asarray(mesh.positions, float), 6), axis=0, return_inverse=True)
    f = inverse.reshape(-1)[np.asarray(mesh.faces, np.int64)]
    edges = np.sort(np.concatenate([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]]), axis=1)
    _, counts = np.unique(edges, axis=0, return_counts=True)
    a, b, c = p[f[:, 0]], p[f[:, 1]], p[f[:, 2]]
    area = np.linalg.norm(np.cross(b - a, c - a), axis=1) / 2
    volume = float(np.einsum("ij,ij->i", a, np.cross(b, c)).sum() / 6)
    return {"faces": int(len(f)), "open_edges": int((counts == 1).sum()), "nonmanifold_edges": int((counts > 2).sum()),
            "degenerate_faces": int((area < 1e-12).sum()), "volume": volume}


def winding(points, mesh: Mesh) -> np.ndarray:
    """Generalised winding number of each point with respect to a mesh: about 1 inside a closed mesh, 0 outside."""
    pts = np.asarray(points, float)
    p = np.asarray(mesh.positions, float)
    f = np.asarray(mesh.faces, np.int64)
    out = np.zeros(len(pts))
    for i in range(0, len(pts), 256):
        q = pts[i:i + 256, None, :]
        a, b, c = p[f[:, 0]] - q, p[f[:, 1]] - q, p[f[:, 2]] - q
        la, lb, lc = (np.linalg.norm(v, axis=2) for v in (a, b, c))
        num = np.einsum("ijk,ijk->ij", a, np.cross(b, c))
        den = (la * lb * lc + np.einsum("ijk,ijk->ij", a, b) * lc + np.einsum("ijk,ijk->ij", b, c) * la
               + np.einsum("ijk,ijk->ij", c, a) * lb)
        out[i:i + 256] = 2 * np.arctan2(num, den).sum(axis=1) / (4 * np.pi)
    return out


def gap(a: Mesh, b: Mesh, sample=400) -> float:
    """Distance between two meshes, 0 when one has a vertex inside the other.

    Measured between vertices, so it is exact to the size of the meshes' edges; enough to tell touching from
    floating.
    """
    pa, pb = np.asarray(a.positions, float), np.asarray(b.positions, float)
    for x, y in ((pa, b), (pb, a)):
        pick = x[np.linspace(0, len(x) - 1, min(sample, len(x))).astype(int)]
        if (np.abs(winding(pick, y)) > 0.5).any():
            return 0.0
    best = np.inf
    for i in range(0, len(pa), 1024):
        d = np.linalg.norm(pa[i:i + 1024, None, :] - pb[None, :, :], axis=2)
        best = min(best, float(d.min()))
    return best


def part_groups(parts: dict, tol: float) -> dict:
    """Which parts touch (gap at most `tol`), the connected groups they form and each part's nearest neighbour.

    Returns ``{"gaps": {(a, b): metres}, "groups": [[names], ...] largest first, "nearest": {name: (other, metres)}}``.
    A model meant to be one object should come out as one group; a second group is a part that floats.
    """
    names = list(parts)
    gaps = {}
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            gaps[(a, b)] = gap(parts[a], parts[b])
    parent = {n: n for n in names}

    def root(n):
        while parent[n] != n:
            parent[n] = parent[parent[n]]
            n = parent[n]
        return n

    for (a, b), g in gaps.items():
        if g <= tol:
            parent[root(a)] = root(b)
    groups = {}
    for n in names:
        groups.setdefault(root(n), []).append(n)
    nearest = {}
    for n in names:
        others = [(g, b if a == n else a) for (a, b), g in gaps.items() if n in (a, b)]
        if others:
            g, o = min(others)
            nearest[n] = (o, g)
    return {"gaps": gaps, "groups": sorted(groups.values(), key=len, reverse=True), "nearest": nearest}
