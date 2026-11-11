"""Deterministic checks of a model built from parts: is each mesh closed, which parts touch, what floats.

An agent that builds a model from drawings cannot see it; these checks tell it, in numbers, what came out wrong:
a part that floats away from the rest, a mesh with holes or inside out. numpy only.

The checks are pure functions returning :class:`Finding` values: a code, a level, where it is, the number behind
it, a message and a hint naming what could change it. They read no file and hold no state.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from manifoldx.modeling.mesh import Mesh
from manifoldx.modeling.trace import iou
from manifoldx.modeling.views import Atlas, Island, template

# Level order, worst first; a report sorts by it.
LEVELS = ("error", "warn", "info")

_HINTS = {
    "A-OPEN": "a builder bug: report it",
    "A-FLIP": "a builder bug: report it",
    "A-FLOAT": ("attach to the nearest part; for a tube, anchor; check the part's text if the drawings "
                "disagree on where it starts"),
    "A-TRIS": "fewer rings or segments; --lod low",
    "A-SEEN": "another builder; power; leave out a view that disagrees (views)",
    "A-ROWS": "the blueprints disagree on what the part is: say it in the part's text, or fill=\"off\"",
    "A-SHEET": "draw --again VIEW",
    "A-KEEP": "draw --again, or a note for that view",
    "A-STRAY": "none, or keep=\"all\"",
    "A-CODE": "paint --again VIEW",
    "A-OUTLINE": "a note for that view; another model",
}


@dataclass(frozen=True)
class Finding:
    """One thing wrong with a model: what, how bad, where, the number behind it, and what could change it."""

    code: str                       # A-FLOAT, A-OPEN, ...
    level: str                      # "error", "warn" or "info"
    part: str | None = None         # the part's name, when the finding is about one
    view: str | None = None         # the view's name, when it is about one
    value: float | int | None = None  # gap in metres, triangle count, IoU, ...
    message: str = ""               # what came out wrong, in words
    hint: str = ""                  # the options that could fix it


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


def closest_offset(a: Mesh, b: Mesh, sample=400) -> np.ndarray:
    """The vector that moves ``a`` the shortest way to touch ``b``, in metres.

    Zero when the two already touch or one has a vertex inside the other, as :func:`gap` counts it; otherwise
    the step from the nearest vertex of ``a`` to the nearest vertex of ``b``. Measured between vertices, so it
    is exact to the size of the meshes' edges.
    """
    pa, pb = np.asarray(a.positions, float), np.asarray(b.positions, float)
    if len(pa) == 0 or len(pb) == 0:
        return np.zeros(3)
    for x, y in ((pa, b), (pb, a)):
        pick = x[np.linspace(0, len(x) - 1, min(sample, len(x))).astype(int)]
        if (np.abs(winding(pick, y)) > 0.5).any():
            return np.zeros(3)
    best, offset = np.inf, np.zeros(3)
    for i in range(0, len(pa), 1024):
        chunk = pa[i:i + 1024]
        d = np.linalg.norm(chunk[:, None, :] - pb[None, :, :], axis=2)
        row, col = np.unravel_index(int(np.argmin(d)), d.shape)
        if d[row, col] < best:
            best, offset = float(d[row, col]), pb[col] - chunk[row]
    return offset


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


# --- Findings: the checks an agent reads ----------------------------------------------------------------


def check_meshes(parts: dict) -> list[Finding]:
    """A-OPEN for a mesh with holes and A-FLIP for one whose faces look inward, both an error."""
    findings = []
    for name, mesh in parts.items():
        report = mesh_report(mesh)
        if report["open_edges"]:
            findings.append(Finding(
                "A-OPEN", "error", part=name, value=report["open_edges"],
                message=f"mesh has {report['open_edges']} open edges (holes)", hint=_HINTS["A-OPEN"]))
        if report["volume"] < 0:
            findings.append(Finding(
                "A-FLIP", "error", part=name, value=round(report["volume"], 6),
                message="mesh is inside out (negative volume)", hint=_HINTS["A-FLIP"]))
    return findings


def check_contact(parts: dict, tol: float) -> list[Finding]:
    """A-FLOAT, an error, for every part outside the largest group of touching parts.

    ``tol`` is the gap in metres under which two parts count as touching. ``value`` is the part's distance to
    its nearest neighbour and the message names that neighbour.
    """
    if len(parts) < 2:
        return []
    groups = part_groups(parts, tol=float(tol))
    main = groups["groups"][0]
    findings = []
    for group in groups["groups"][1:]:
        for name in group:
            other, distance = groups["nearest"][name]
            findings.append(Finding(
                "A-FLOAT", "error", part=name, value=float(distance),
                message=f"floats: not connected to {', '.join(main[:3])}; nearest part {other}",
                hint=_HINTS["A-FLOAT"]))
    return findings


def check_triangles(parts: dict, max_tris: int) -> list[Finding]:
    """A-TRIS, a warning, for every part with more triangles than ``max_tris``."""
    findings = []
    for name, mesh in parts.items():
        count = int(len(mesh.faces))
        if count > max_tris:
            findings.append(Finding(
                "A-TRIS", "warn", part=name, value=count,
                message=f"{count} triangles, more than the {int(max_tris)} allowed", hint=_HINTS["A-TRIS"]))
    return findings


def check_seen(parts: dict, view: str, drawn_masks: dict, frame) -> list[Finding]:
    """A-SEEN: what the camera sees of each part against the mask a drawing gives it, in one view.

    ``parts`` maps a name to a mesh, or to a sequence of meshes when a part has several instances that the
    drawing shows as one region (a chair's four legs, a pair of arms). ``drawn_masks`` maps a name to a bool
    mask (the drawing's silhouette of that part) and ``frame`` is ``(u_min, u_max, v_min, v_max)``, the metres
    the mask covers. A part below 0.75 IoU warns, below 0.5 it is an error; the message counts the pixels built
    but not drawn and drawn but not built, and the rows they fall on. Parts with no drawn mask, or an empty
    one, are left alone.
    """
    groups = {name: (list(value) if isinstance(value, (list, tuple)) else [value])
              for name, value in parts.items()}
    drawn = {name: np.asarray(mask, bool) for name, mask in drawn_masks.items() if name in groups}
    drawn = {name: mask for name, mask in drawn.items() if mask.any()}
    if not drawn:
        return []
    shape = next(iter(drawn.values())).shape
    if any(mask.shape != shape for mask in drawn.values()):
        raise ValueError("all drawn masks of one view must have the same shape")
    height, width = shape
    island = Island(view=view, rect=(0, 0, width, height), frame=tuple(float(v) for v in frame))
    flat, colours, group_of = {}, {}, {}
    for index, (name, meshes) in enumerate(groups.items()):
        for k, mesh in enumerate(meshes):
            key = f"{name}#{k}"
            flat[key] = mesh
            colours[key] = (255, 255, 255)  # the region index comes from the key order, not the colour
            group_of[key] = index + 1
    atlas = Atlas(size=(width, height), islands=[island], uvs={}, meshes={},
                  island_of={key: np.zeros(len(mesh.faces), np.int32) for key, mesh in flat.items()},
                  texel=1.0)
    _, regions = template(flat, atlas, colours, 0, whole=True)
    region_numbers = {key: i + 1 for i, key in enumerate(flat)}  # what template painted this key
    findings = []
    for index, name in enumerate(groups):
        if name not in drawn:
            continue
        wanted = [region_numbers[key] for key in flat if group_of[key] == index + 1]
        seen, want = np.isin(regions, wanted), drawn[name]
        overlap = float(iou(seen, want))
        if overlap >= 0.75:
            continue
        extra, missing = int((seen & ~want).sum()), int((want & ~seen).sum())
        rows = np.flatnonzero((seen ^ want).any(axis=1))
        where = (f"; rows {rows.min() / height:.0%}-{rows.max() / height:.0%} from the top"
                 if rows.size else "")
        findings.append(Finding(
            "A-SEEN", "error" if overlap < 0.5 else "warn", part=name, view=view, value=round(overlap, 3),
            message=(f"{view} view: visible part differs from the drawing ({extra} px built but not drawn, "
                     f"{missing} px drawn but not built{where})"),
            hint=_HINTS["A-SEEN"]))
    return findings
