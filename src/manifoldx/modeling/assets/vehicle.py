"""The shape every vehicle builder returns: named parts, a pose and one merged instancing mesh.

A builder (the car today; a bus, a truck or a cycle later) publishes a :class:`Vehicle`: a params object, the rigid
parts it is made of (meshes per material in vehicle coordinates) and the transforms that pose them. :meth:`Vehicle.merged`
is the mesh a crowd needs: every outside piece posed, coloured per vertex from :data:`PALETTE`, and optionally
decimated, so hundreds of vehicles draw as one entity each.

Axes are the ones every vehicle shares: metres, Y up, the front at +X, the right side at +Z, the centre on the ground
at the origin.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from manifoldx.modeling.mesh import Mesh

# The materials a vehicle uses, as (sRGB colour, roughness, metallic); "lamp" and the beacons are drawn unlit when the
# lights are on. A builder may add to them, and a caller overrides any entry per vehicle.
PALETTE = {
    "interior": ("#2a2c30", 0.85, 0.0), "carpet": ("#1a1b1d", 0.95, 0.0), "dash": ("#1f2023", 0.6, 0.0),
    "seat": ("#3a3c41", 0.8, 0.0),
    "paint": ("#8fbccc", 0.32, 0.15), "roof": ("#8fbccc", 0.32, 0.15),
    "black": ("#0b0c0e", 0.15, 0.0), "glass": ("#1d262c", 0.05, 0.0),
    "trim": ("#17181a", 0.75, 0.0), "underbody": ("#101010", 0.9, 0.0), "liner": ("#0e0e0e", 0.95, 0.0),
    "lamp": ("#eef3ff", 0.2, 0.0), "tail": ("#c41a12", 0.25, 0.0), "plate": ("#e3e8ea", 0.6, 0.0),
    "chrome": ("#d6dbe0", 0.1, 1.0), "tyre": ("#141414", 0.9, 0.0), "rim": ("#2a2c30", 0.45, 0.3),
    "spoke": ("#c8cdd2", 0.2, 1.0), "alloy": ("#33363b", 0.3, 0.8), "whitewall": ("#f2f0ea", 0.55, 0.0),
    "beacon_a": ("#1f4fff", 0.25, 0.0), "beacon_b": ("#e01b1b", 0.25, 0.0),
}

# The materials that belong inside the body: left out of a merged instancing mesh unless asked for.
INSIDE = frozenset({"interior", "carpet", "dash", "seat", "liner"})


@dataclass
class Part:
    """A rigid piece of a vehicle: meshes per material in vehicle coordinates, and how it moves."""

    name: str
    kind: str  # "body", "door", "hood", "hatch", "wheel"
    pieces: dict = field(default_factory=dict)  # material name -> Mesh
    pivot: np.ndarray = field(default_factory=lambda: np.zeros(3))
    axis: np.ndarray = field(default_factory=lambda: np.array([0.0, 1.0, 0.0]))
    open_angle: float = 0.0  # degrees at pose value 1, signed
    steer: bool = False


def _rotate(q, v) -> np.ndarray:
    """Rotate points by a quaternion (x, y, z, w), one point or an (N, 3) array."""
    u, w = q[:3], q[3]
    return v + 2 * np.cross(u, np.cross(u, v) + w * v)


def _colour(value) -> np.ndarray:
    """A palette colour as sRGB floats in 0..1: a hex string, or a float triple."""
    if isinstance(value, str):
        s = value.lstrip("#")
        return np.array([int(s[k:k + 2], 16) for k in (0, 2, 4)], np.float64) / 255.0
    return np.asarray(value[:3], np.float64)


def _nearest_colours(points, source, colours, chunk: int = 512) -> np.ndarray:
    """Each point's colour, taken from the nearest source vertex, in chunks so memory stays bounded."""
    pts = np.asarray(points, np.float64)
    src = np.asarray(source, np.float32)
    src2 = (src * src).sum(1)
    out = np.empty((len(pts), np.asarray(colours).shape[1]), np.float32)
    for start in range(0, len(pts), chunk):
        q = pts[start:start + chunk].astype(np.float32)
        d = (q * q).sum(1)[:, None] + src2[None, :] - 2.0 * (q @ src.T)  # squared distance, no (chunk, N, 3) array
        out[start:start + chunk] = np.asarray(colours)[d.argmin(1)]
    return out


def orient_outward(positions, faces) -> np.ndarray:
    """The faces of every closed component wound alike and turned so they point out.

    The renderer culls back faces, so a closed shell must be wound counter-clockwise seen from outside: a face flipped
    against its neighbours, or a whole shell inside out, lets the camera see into the vehicle. Within each connected
    component whose every edge is shared by exactly two faces, the winding is carried from one face to the next
    across shared edges, and the component is then turned round if its signed volume is negative. Open components,
    such as glass drawn from both sides, are returned as they are.
    """
    F = np.array(faces, dtype=np.int64, copy=True)
    P = np.asarray(positions, float)
    edges = {}
    for i, (a, b, c) in enumerate(F):
        for u, v in ((a, b), (b, c), (c, a)):
            edges.setdefault((min(u, v), max(u, v)), []).append(i)
    neighbours = [[] for _ in range(len(F))]
    open_faces = set()
    for (u, v), fs in edges.items():
        if len(fs) == 2:
            neighbours[fs[0]].append((fs[1], u, v))
            neighbours[fs[1]].append((fs[0], u, v))
        else:
            open_faces.update(fs)
    seen = np.zeros(len(F), bool)
    for start in range(len(F)):
        if seen[start]:
            continue
        seen[start] = True
        component, stack = [start], [start]
        while stack:
            i = stack.pop()
            for j, u, v in neighbours[i]:
                if seen[j]:
                    continue
                seen[j] = True
                # Wound alike, two faces cross their shared edge in opposite directions.
                if _crosses(F[i], u, v) == _crosses(F[j], u, v):
                    F[j] = F[j][::-1]
                component.append(j)
                stack.append(j)
        if open_faces.intersection(component):
            F[component] = np.asarray(faces, dtype=np.int64)[component]
            continue
        a, b, c = P[F[component, 0]], P[F[component, 1]], P[F[component, 2]]
        if np.einsum("ij,ij->i", a, np.cross(b, c)).sum() < 0:
            F[component] = F[component][:, ::-1]
    return F.astype(np.asarray(faces).dtype, copy=False)


def _crosses(face, u, v) -> bool:
    """Whether the face's winding runs from u to v along their shared edge."""
    a, b, c = face
    return (a, b) == (u, v) or (b, c) == (u, v) or (c, a) == (u, v)


@dataclass
class Vehicle:
    """A built vehicle: its measurements and its rigid parts. Every builder returns one of these.

    Every closed piece is wound outward on construction (:func:`orient_outward`), so no builder can leave a face the
    renderer would cull from outside.
    """

    params: object
    parts: list

    def __post_init__(self):
        from dataclasses import replace

        for part in self.parts:
            for mat, mesh in list(part.pieces.items()):
                faces = orient_outward(mesh.positions, mesh.faces)
                if not np.array_equal(faces, mesh.faces):
                    fixed = replace(mesh, faces=faces)
                    part.pieces[mat] = fixed if mesh.normals is None else fixed.recompute_normals()

    def part(self, name: str) -> Part:
        return next(p for p in self.parts if p.name == name)

    @property
    def palette(self) -> dict:
        """Material name -> (sRGB colour, roughness, metallic); the shared palette, plus a builder's own materials."""
        return PALETTE

    def pose(self, **pose) -> dict:
        """Each part's (position, quaternion xyzw) so that vehicle-space vertices land in their posed place."""
        raise NotImplementedError(f"{type(self).__name__} does not say how its parts move")

    @property
    def landmarks(self) -> dict:
        """Named three-dimensional points on the vehicle, for registering a drawing by keypoints."""
        raise NotImplementedError(f"{type(self).__name__} has no landmarks")

    def merged(self, palette: dict | None = None, grid: int | None = None, inside: bool = False, pose=None) -> Mesh:
        """Every piece posed and merged into one mesh, with per-vertex colours from the palette.

        `palette` overrides entries of :attr:`palette`; `pose` is a transform dict (default: the rest pose). The
        interior materials in :data:`INSIDE` are left out unless `inside`. With `grid`, the mesh is decimated and
        each surviving vertex takes the colour of its nearest source vertex. The result is centred on the ground in
        x and z, its nose at +X, with normals recomputed.
        """
        colours_by_material = {**self.palette, **(palette or {})}
        pose = self.pose() if pose is None else pose
        pos, faces, colours, base = [], [], [], 0
        for part in self.parts:
            at, q = pose[part.name]
            at, q = np.asarray(at, np.float64), np.asarray(q, np.float64)
            for mat, piece in part.pieces.items():
                if not inside and mat in INSIDE:
                    continue
                p = _rotate(q, np.asarray(piece.positions, np.float64)) + at
                pos.append(p)
                faces.append(np.asarray(piece.faces) + base)
                colours.append(np.tile(_colour(colours_by_material[mat][0]), (len(p), 1)))
                base += len(p)
        whole = Mesh(positions=np.vstack(pos).astype(np.float32), faces=np.vstack(faces).astype(np.uint32),
                     colors=np.vstack(colours).astype(np.float32))
        small = whole.decimate(grid=grid) if grid else whole
        if grid:  # decimate keeps no colours: carry them over from the nearest source vertex
            small = small.with_colors(_nearest_colours(small.positions, whole.positions, whole.colors))
        p = np.asarray(small.positions, np.float64)
        p[:, [0, 2]] -= (p[:, [0, 2]].max(0) + p[:, [0, 2]].min(0)) / 2
        return small.with_positions(p).recompute_normals()
