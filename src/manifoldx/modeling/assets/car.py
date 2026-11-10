"""A parametric car: named measurements in, an articulated low-poly car out.

The body is a loft of cross-sections along the car. Each section is a fixed template (floor, rocker, shoulder,
belt, glass, roof) scaled by smooth side-profile and plan curves; the curves come from measurements a spec sheet
or a photo gives: length, width, height, wheelbase, overhangs, nose and tail heights, where the hood meets the
windscreen, where the roof ends. Wheel arches are cut by raising the skin above each wheel, so the loft needs no
booleans.

Regions are tags on the loft's faces, in its own coordinates (position along the car, band around the section),
so they follow any shape: glass, pillars, doors, hood and hatch. Small parts that sit on the skin (lamps, grille,
plate, handles) are decals: flat outlines projected onto the body along a view direction, so a lamp style is
just an outline.

Doors, hood and hatch are separate parts with a hinge; wheels spin and the front ones steer. `Car.pose` gives
each part's transform for an opening, a wheel turn and a steering angle.

Axes: metres, Y up, the car's front at +X, its right side at +Z, its centre on the ground at the origin.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from manifoldx.modeling.mesh import Mesh


@dataclass
class CarParams:
    """Every measurement of the car. Heights are from the ground; `_u` values are fractions of the length from
    the front (0) to the rear (1)."""

    # Package: the spec sheet.
    length: float = 4.03
    width: float = 1.81
    height: float = 1.57
    wheelbase: float = 2.66
    front_overhang: float = 0.72
    track: float = 1.55
    wheel_radius: float = 0.318
    tyre_width: float = 0.205
    rim_ratio: float = 0.62  # rim radius over wheel radius
    arch_gap: float = 0.045  # between the tyre and the arch
    # Side profile.
    nose_top: float = 0.74  # top of the nose at the very front
    hood_front: float = 0.87  # the hood just behind the nose
    cowl_u: float = 0.27  # where the hood meets the windscreen
    cowl_height: float = 1.13
    windscreen_u: float = 0.45  # where the windscreen meets the roof
    roof_end_u: float = 0.88  # where the roof meets the hatch
    roof_end_drop: float = 0.05
    tail_top: float = 1.02  # top of the tail at the rear end
    belt_front: float = 1.09  # belt (bottom of the side glass) at the cowl
    belt_rear: float = 1.17  # and at the end of the roof
    sill: float = 0.22  # underside between the wheels
    bumper_bottom_front: float = 0.30
    bumper_bottom_rear: float = 0.36
    hood_crown: float = 0.10  # how far the hood's edge sits below its crown
    # Plan and section.
    nose_half: float = 0.80  # half-width at the front end, over the half-width
    tail_half: float = 0.84
    round_front: float = 0.17  # how far the front corners are pulled back (m)
    round_rear: float = 0.18
    round_power: float = 2.0
    tumblehome: float = 0.30  # how much narrower the roof is than the body
    shoulder: float = 0.95  # belt half-width over the half-width
    # Greenhouse.
    b_pillar_u: float = 0.56
    b_pillar_width: float = 0.07
    c_pillar_u: float = 0.75  # end of the rear door glass; behind it the quarter is black
    black_pillars: bool = True
    roof: str = "paint"  # or "black"
    # Parts.
    headlamp: str = "almond"  # "almond", "round" or "bar"
    headlamp_center: tuple = (0.66, 0.80)  # (lateral, height)
    headlamp_size: tuple = (0.30, 0.11)
    drl_bar: bool = True
    drl_height: float = 0.86
    drl_half_length: float = 0.56
    grille: tuple = (0.62, 0.30, 0.42)  # half-width, bottom, top
    taillamp_center: tuple = (0.69, 1.00)
    taillamp_size: tuple = (0.30, 0.17)
    spoiler: bool = True
    mirrors: bool = True
    spokes: int = 5
    cladding: bool = True  # black lower body
    # Resolution.
    sections: int = 28


@dataclass
class Part:
    """A rigid piece of the car: meshes per material in car coordinates, and how it moves."""

    name: str
    kind: str  # "body", "door", "hood", "hatch", "wheel"
    pieces: dict = field(default_factory=dict)  # material name -> Mesh
    pivot: np.ndarray = field(default_factory=lambda: np.zeros(3))
    axis: np.ndarray = field(default_factory=lambda: np.array([0.0, 1.0, 0.0]))
    open_angle: float = 0.0  # degrees at pose value 1, signed
    steer: bool = False


@dataclass
class Car:
    params: CarParams
    parts: list

    def part(self, name: str) -> Part:
        return next(p for p in self.parts if p.name == name)

    def pose(self, doors=0.0, hood=0.0, hatch=0.0, wheel_turn=0.0, steer=0.0) -> dict:
        """Each part's (position, quaternion xyzw) so that car-space vertices land in their posed place.

        doors, hood, hatch: 0 shut to 1 fully open (doors may be a dict by part name). wheel_turn: radians the
        wheels have rolled. steer: radians the front wheels turn, positive to the left.
        """
        out = {}
        for p in self.parts:
            if p.kind == "door":
                q = _axis_angle(p.axis, np.radians(p.open_angle) * (doors.get(p.name, 0.0) if isinstance(doors, dict) else doors))
            elif p.kind in ("hood", "hatch"):
                q = _axis_angle(p.axis, np.radians(p.open_angle) * (hood if p.kind == "hood" else hatch))
            elif p.kind == "wheel":
                q = _axis_angle(np.array([0.0, 0.0, 1.0]), -wheel_turn)
                if p.steer:
                    q = _quat_mul(_axis_angle(np.array([0.0, 1.0, 0.0]), steer), q)
            else:
                q = np.array([0.0, 0.0, 0.0, 1.0])
            out[p.name] = (p.pivot - _rotate(q, p.pivot), q)
        return out


# --- curves and small geometry helpers ---------------------------------------------------------------------


def _pchip(xk, yk, x):
    """Monotone cubic through (xk, yk): smooth, and it never overshoots between control points."""
    xk, yk, x = np.asarray(xk, float), np.asarray(yk, float), np.asarray(x, float)
    h, d = np.diff(xk), np.diff(yk) / np.diff(xk)
    m = np.zeros(len(xk))
    m[0], m[-1] = d[0], d[-1]
    for k in range(1, len(xk) - 1):
        if d[k - 1] * d[k] > 0:
            w1, w2 = 2 * h[k] + h[k - 1], h[k] + 2 * h[k - 1]
            m[k] = (w1 + w2) / (w1 / d[k - 1] + w2 / d[k])
    i = np.clip(np.searchsorted(xk, x, side="right") - 1, 0, len(xk) - 2)
    t = (x - xk[i]) / h[i]
    h00, h10, h01, h11 = 2 * t**3 - 3 * t**2 + 1, t**3 - 2 * t**2 + t, -2 * t**3 + 3 * t**2, t**3 - t**2
    return h00 * yk[i] + h10 * h[i] * m[i] + h01 * yk[i + 1] + h11 * h[i] * m[i + 1]


def _axis_angle(axis, angle) -> np.ndarray:
    axis = np.asarray(axis, float) / np.linalg.norm(axis)
    return np.concatenate([axis * np.sin(angle / 2), [np.cos(angle / 2)]])


def _quat_mul(a, b) -> np.ndarray:
    (ax, ay, az, aw), (bx, by, bz, bw) = a, b
    return np.array([aw * bx + ax * bw + ay * bz - az * by, aw * by - ax * bz + ay * bw + az * bx,
                     aw * bz + ax * by - ay * bx + az * bw, aw * bw - ax * bx - ay * by - az * bz])


def _rotate(q, v) -> np.ndarray:
    u, w = q[:3], q[3]
    return v + 2 * np.cross(u, np.cross(u, v) + w * v)


def _mesh(verts, tris) -> Mesh:
    return Mesh(positions=np.asarray(verts, np.float32), faces=np.asarray(tris, np.uint32)).recompute_normals()


def _box(center, size, yaw=0.0, pitch=0.0):
    """Vertices and triangles of a box; yaw turns it about Y, pitch about Z."""
    sx, sy, sz = np.asarray(size, float) / 2
    v = np.array([[x, y, z] for x in (-sx, sx) for y in (-sy, sy) for z in (-sz, sz)])
    cy, syw, cp, sp = np.cos(yaw), np.sin(yaw), np.cos(pitch), np.sin(pitch)
    v = v @ np.array([[cp, sp, 0], [-sp, cp, 0], [0, 0, 1]]) @ np.array([[cy, 0, -syw], [0, 1, 0], [syw, 0, cy]])
    q = [(0, 1, 3, 2), (4, 6, 7, 5), (0, 4, 5, 1), (2, 3, 7, 6), (0, 2, 6, 4), (1, 5, 7, 3)]
    return v + center, [t for a, b, c, d in q for t in ((a, b, c), (a, c, d))]


def _revolve(profile, segments, side):
    """A closed (radius, offset) profile turned about the Z axis; offset grows toward the outside (side = +-1)."""
    a = np.linspace(0, 2 * np.pi, segments, endpoint=False)
    n = len(profile)
    v = np.array([[r * np.cos(t), r * np.sin(t), side * o] for t in a for r, o in profile])
    tris = []
    for s in range(segments):
        for k in range(n):
            i0, i1 = s * n + k, s * n + (k + 1) % n
            j0, j1 = ((s + 1) % segments) * n + k, ((s + 1) % segments) * n + (k + 1) % n
            tris += [(i0, j0, j1), (i0, j1, i1)] if side > 0 else [(i0, j1, j0), (i0, i1, j1)]
    return v, tris


def _triangulate(poly):
    """Ear clipping of a simple 2-D polygon; returns index triples."""
    poly = np.asarray(poly, float)
    idx = list(range(len(poly)))
    area = np.sum(poly[:, 0] * np.roll(poly[:, 1], -1) - np.roll(poly[:, 0], -1) * poly[:, 1])
    if area < 0:
        idx.reverse()
    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    tris = []
    while len(idx) > 3:
        for k in range(len(idx)):
            i0, i1, i2 = idx[k - 1], idx[k], idx[(k + 1) % len(idx)]
            a, b, c = poly[i0], poly[i1], poly[i2]
            if cross(a, b, c) <= 1e-12:
                continue
            if any(cross(a, b, poly[j]) >= 0 and cross(b, c, poly[j]) >= 0 and cross(c, a, poly[j]) >= 0
                   for j in idx if j not in (i0, i1, i2)):
                continue
            tris.append((i0, i1, i2))
            idx.pop(k)
            break
        else:
            break
    tris += [(idx[0], idx[k], idx[k + 1]) for k in range(1, len(idx) - 1)]
    return tris


# --- the body ------------------------------------------------------------------------------------------------

# Half a section, from the bottom centre round the side to the roof centre: 12 points, so 11 bands of faces.
# Bands: 0 floor, 1 tuck, 2 rocker, 3 lower side, 4 upper side, 5 shoulder, 6 lower glass, 7 upper glass,
# 8 roof edge, 9 roof, 10 roof centre.
HALF = 12
RING = 2 * HALF - 2


def _section(top, bottom, half, belt, roof):
    body, upper = belt - bottom, top - belt
    return [(0, bottom), (0.62 * half, bottom), (0.86 * half, bottom + 0.035 * body),
            (0.96 * half, bottom + 0.16 * body), (half, bottom + 0.48 * body), (0.985 * half, belt - 0.14 * body),
            (0.95 * half, belt - 0.01 * body), (0.55 * 0.95 * half + 0.45 * roof, belt + 0.45 * upper),
            (roof, top - 0.10 * upper), (0.80 * roof, top - 0.025 * upper), (0.42 * roof, top - 0.004 * upper),
            (0, top)]


def _stations(p: CarParams, axles):
    """Section positions as fractions of the length: denser at the ends, plus a fan over each wheel arch."""
    u = 0.5 - 0.5 * np.cos(np.pi * np.linspace(0, 1, p.sections))
    R = p.wheel_radius + p.arch_gap
    for xa in axles:
        ua = (p.length / 2 - xa) / p.length
        u = np.concatenate([u, ua + R * np.sin(np.linspace(-np.pi / 2, np.pi / 2, 9)) / p.length,
                            [ua - (R + 0.02) / p.length, ua + (R + 0.02) / p.length]])
    u = np.sort(u)
    keep = np.concatenate([[True], np.diff(u) > 0.004])
    return u[keep]


def _profiles(p: CarParams, u):
    half = p.width / 2
    top = _pchip([0, 0.04, p.cowl_u, p.windscreen_u, 0.62, p.roof_end_u, 1],
                 [p.nose_top, p.hood_front, p.cowl_height, p.height - 0.015, p.height,
                  p.height - p.roof_end_drop, p.tail_top], u)
    fo, ro = p.front_overhang / p.length, 1 - (p.front_overhang + p.wheelbase) / p.length
    bottom = _pchip([0, 0.6 * fo, 1 - 0.6 * ro, 1], [p.bumper_bottom_front, p.sill, p.sill, p.bumper_bottom_rear], u)
    halfw = _pchip([0, 0.05, 0.15, 0.85, 0.95, 1], [p.nose_half * half, 0.97 * half, half, half, 0.97 * half,
                                                    p.tail_half * half], u)
    g = np.clip((u - (p.cowl_u - 0.03)) / 0.04, 0, 1)  # 0 over the hood, 1 under the glass
    belt_g = np.interp(u, [p.cowl_u, p.roof_end_u], [p.belt_front, p.belt_rear])
    belt = np.minimum((1 - g) * (top - p.hood_crown) + g * belt_g, top - 0.05)
    roof = (1 - g) * 0.80 * halfw + g * (1 - p.tumblehome) * halfw
    return top, bottom, halfw, belt, roof


def _rounded(x, s, halfw, p: CarParams, zone=0.5):
    """Pull points near each end toward the middle, the more the further they are from the centre line."""
    r = np.clip(np.abs(s) / np.maximum(halfw, 1e-6), 0, 1) ** p.round_power
    out = x.copy()
    for end, amount, sign in ((p.length / 2, p.round_front, -1), (-p.length / 2, p.round_rear, 1)):
        d = (x - end) * sign
        m = (d >= 0) & (d < zone)
        out[m] = end + sign * (amount * r[m] + d[m] * (1 - amount * r[m] / zone))
    return out


def _body_grid(p: CarParams, axles):
    u = _stations(p, axles)
    top, bottom, halfw, belt, roof = _profiles(p, u)
    S = len(u)
    P = np.zeros((S, RING, 3))
    for i in range(S):
        h = _section(top[i], bottom[i], halfw[i], belt[i], roof[i])
        full = h + [(-s, y) for s, y in reversed(h[1:-1])]
        P[i, :, 2] = [s for s, _ in full]
        P[i, :, 1] = [y for _, y in full]
        P[i, :, 0] = p.length / 2 - u[i] * p.length
    P[..., 0] = _rounded(P[..., 0], P[..., 2], halfw[:, None], p)
    # Wheel arches: raise every point that falls inside the arch circle onto it.
    R = p.wheel_radius + p.arch_gap
    raised = np.zeros(P.shape[:2], bool)
    for xa in axles:
        dx = P[..., 0] - xa
        inside = np.abs(dx) < R
        arch = p.wheel_radius + np.sqrt(np.clip(R**2 - dx**2, 0, None))
        m = inside & (P[..., 1] < arch)
        P[..., 1] = np.where(m, arch, P[..., 1])
        raised |= m
    return u, P, raised, halfw


def _band(j):
    """The band (0-10) of the face that starts at ring index j and goes to j + 1."""
    def hj(k):
        return k if k < HALF else RING - k
    return min(hj(j), hj((j + 1) % RING))


def _body_faces(p: CarParams, u, P, raised):
    """Triangles of the loft, each with a material and a part."""
    S = len(u)
    verts = P.reshape(-1, 3)
    def vid(i, j):
        return i * RING + j % RING
    tris, mats, parts = [], [], []
    for i in range(S - 1):
        um = (u[i] + u[i + 1]) / 2
        for j in range(RING):
            b = _band(j)
            side = "r" if P[i, j % RING, 2] + P[i, (j + 1) % RING, 2] > 0 else "l"
            mat, part = _region(p, um, b), "body"
            if raised[i, j] and raised[i, (j + 1) % RING] and raised[i + 1, j] and raised[i + 1, (j + 1) % RING]:
                mat = "liner"
            if b in range(2, 8):
                if p.cowl_u + 0.01 <= um < p.b_pillar_u:
                    part = f"door_f{side}"
                elif p.b_pillar_u <= um < p.c_pillar_u:
                    part = f"door_r{side}"
            if 0.035 <= um < p.cowl_u - 0.005 and b >= 6:
                part = "hood"
            if um >= p.roof_end_u - 0.01 and b >= 4:
                part = "hatch"
            a, c, d, e = vid(i, j), vid(i, j + 1), vid(i + 1, j + 1), vid(i + 1, j)
            tris += [(a, c, d), (a, d, e)]
            mats += [mat, mat]
            parts += [part, part]
    # Close the ends with fans; the nose and tail are bumper paint.
    verts = list(verts)
    for i, part in ((0, "body"), (S - 1, "body")):
        verts.append(P[i].mean(0))
        c = len(verts) - 1
        for j in range(RING):
            t = (c, vid(i, j + 1), vid(i, j)) if i == 0 else (c, vid(i, j), vid(i, j + 1))
            tris.append(t)
            mats.append("paint")
            parts.append(part)
    verts = np.array(verts)
    tris = np.array(tris)
    # Outward normals: the enclosed volume must be positive.
    a, b, c = verts[tris[:, 0]], verts[tris[:, 1]], verts[tris[:, 2]]
    if np.einsum("ij,ij->i", a, np.cross(b, c)).sum() < 0:
        tris = tris[:, ::-1]
    return verts, tris, np.array(mats), np.array(parts)


def _region(p: CarParams, u, b):
    """The material of a face at length fraction u in band b."""
    if b <= 1:
        return "underbody"
    if b == 2 and p.cladding:
        return "trim"
    if u < p.cowl_u - 0.01 or b <= 5:
        return "paint"
    roof = "black" if p.roof == "black" else "paint"
    pillar = "black" if p.black_pillars else "paint"
    if u < p.windscreen_u:
        return "glass" if b >= 8 else pillar
    if u < p.roof_end_u:
        if b >= 8:
            return roof
        if u >= p.c_pillar_u or abs(u - p.b_pillar_u) < p.b_pillar_width / 2 / p.length:
            return pillar
        return "glass"
    if u < 0.985 and b >= 7:
        return "glass"
    return pillar if b >= 6 else "paint"


# --- decals --------------------------------------------------------------------------------------------------


def _raycast(origins, direction, verts, tris, batch=128):
    """Distance to the nearest triangle along direction for each origin (inf where it misses), and its index."""
    if len(origins) > batch:
        parts = [_raycast(origins[i:i + batch], direction, verts, tris, batch) for i in range(0, len(origins), batch)]
        return np.concatenate([t for t, _ in parts]), np.concatenate([k for _, k in parts])
    a, b, c = verts[tris[:, 0]], verts[tris[:, 1]], verts[tris[:, 2]]
    e1, e2 = b - a, c - a
    pv = np.cross(direction, e2)
    det = np.einsum("ij,ij->i", e1, pv)
    ok = np.abs(det) > 1e-12
    inv = np.where(ok, 1 / np.where(ok, det, 1), 0)
    tv = origins[:, None, :] - a[None]
    uu = np.einsum("pij,ij->pi", tv, pv) * inv
    qv = np.cross(tv, e1[None])
    vv = (qv @ direction) * inv
    t = np.einsum("pij,ij->pi", qv, e2) * inv
    hit = ok & (uu >= 0) & (vv >= 0) & (uu + vv <= 1) & (t > 0)
    t = np.where(hit, t, np.inf)
    k = t.argmin(1)
    return t[np.arange(len(origins)), k], k


def _decals(p: CarParams, verts, tris, parts):
    """Flat outlines projected onto the skin: (material, view, outline) -> (material, part, vertices, triangles)."""
    W, L = p.width, p.length
    out = []

    def add(mat, view, outline):
        # Triangulate flat, then split each triangle k x k times (about 4 cm apart), so the decal bends with the
        # skin it lands on instead of cutting through it.
        flat = np.asarray(outline, float)
        k = int(np.clip(np.ptp(flat, axis=0).max() / 0.04, 4, 40))
        pts2, tri = [], []
        for t3 in _triangulate(flat):
            a2, b2, c2 = flat[list(t3)]
            base = len(pts2)
            ids = {}
            for i in range(k + 1):
                for j in range(k + 1 - i):
                    ids[i, j] = len(pts2)
                    pts2.append(a2 + (b2 - a2) * i / k + (c2 - a2) * j / k)
            for i in range(k):
                for j in range(k - i):
                    tri.append((ids[i, j], ids[i + 1, j], ids[i, j + 1]))
                    if i + j < k - 1:
                        tri.append((ids[i + 1, j], ids[i + 1, j + 1], ids[i, j + 1]))
        outline = np.array(pts2)
        tri = np.array(tri)
        if view in ("front", "rear"):
            sign = 1 if view == "front" else -1
            d = np.array([-sign, 0.0, 0.0])
            o = np.column_stack([np.full(len(outline), sign * (L / 2 + 1)), outline[:, 1], outline[:, 0]])
        else:  # "left" or "right": outline is (x, height)
            sign = 1 if view == "right" else -1
            d = np.array([0.0, 0.0, -sign])
            o = np.column_stack([outline[:, 0], outline[:, 1], np.full(len(outline), sign * (W / 2 + 1))])
        t, k = _raycast(o, d, verts, tris)
        if not np.isfinite(t).all():
            return
        pts = o + (t - 0.006)[:, None] * d
        owner = parts[k]
        names, counts = np.unique(owner, return_counts=True)
        if np.dot(np.cross(pts[tri[0, 1]] - pts[tri[0, 0]], pts[tri[0, 2]] - pts[tri[0, 0]]), d) > 0:
            tri = tri[:, ::-1]  # face the viewer, not the skin
        out.append((mat, str(names[counts.argmax()]), pts, tri))

    def both(mat, view, outline):
        add(mat, view, outline)
        add(mat, view, [(-a, b) for a, b in reversed(outline)])

    cy, cz = p.headlamp_center
    a, bh = p.headlamp_size[0] / 2, p.headlamp_size[1] / 2
    if p.headlamp == "round":
        eye = [(cy + bh * np.cos(s), cz + bh * np.sin(s)) for s in np.linspace(0, 2 * np.pi, 16, endpoint=False)]
    elif p.headlamp == "bar":
        eye = [(cy - a, cz - bh / 3), (cy + a, cz - bh / 3), (cy + a, cz + bh / 3), (cy - a, cz + bh / 3)]
    else:  # almond: an ellipse drawn to a point on its inner side, toward the middle of the car
        s = np.linspace(0, 2 * np.pi, 20, endpoint=False)
        eye = [(cy + a * np.cos(v), cz + bh * np.sin(v) * (1 - 0.75 * max(0.0, -np.cos(v)))) for v in s]
    both("lamp", "front", eye)
    if p.drl_bar:
        h, w = p.drl_height, p.drl_half_length
        add("lamp", "front", [(-w, h - 0.009), (w, h - 0.009), (w, h + 0.009), (-w, h + 0.009)])
    gw, g0, g1 = p.grille
    add("trim", "front", [(-gw, g0), (gw, g0), (gw * 1.06, g1), (-gw * 1.06, g1)])
    add("plate", "front", [(-0.26, 0.47), (0.26, 0.47), (0.26, 0.59), (-0.26, 0.59)])
    ty, tz = p.taillamp_center
    ta, tb = p.taillamp_size[0] / 2, p.taillamp_size[1] / 2
    both("tail", "rear", [(ty - ta, tz - tb), (ty + ta, tz - tb), (ty + ta, tz + tb), (ty - ta, tz + tb)])
    add("plate", "rear", [(-0.26, 0.52), (0.26, 0.52), (0.26, 0.64), (-0.26, 0.64)])
    for view in ("left", "right"):
        for u0, u1 in ((p.cowl_u, p.b_pillar_u), (p.b_pillar_u, p.c_pillar_u)):
            x = L / 2 - (u0 + 0.62 * (u1 - u0)) * L
            add("chrome", view, [(x - 0.08, 0.93), (x + 0.08, 0.93), (x + 0.08, 0.955), (x - 0.08, 0.955)])
    return out


# --- wheels and small parts ----------------------------------------------------------------------------------


def _wheel(p: CarParams, x, side):
    """Tyre, rim and spokes of one wheel at its hub, as {material: (verts, tris)} in car coordinates."""
    r, w, rr = p.wheel_radius, p.tyre_width, p.wheel_radius * p.rim_ratio
    hub = np.array([x, r, side * (p.track / 2)])
    tyre = _revolve([(rr, -w / 2), (r - 0.035, -w / 2), (r, -w / 2 + 0.035), (r, w / 2 - 0.035),
                     (r - 0.035, w / 2), (rr, w / 2)], 32, side)
    rim = _revolve([(0.0, w / 2 - 0.05), (rr, w / 2 - 0.05), (rr, -w / 2 + 0.03), (0.0, -w / 2 + 0.03)], 32, side)
    spokes_v, spokes_t = [], []
    for k in range(p.spokes):
        ang = 2 * np.pi * k / p.spokes
        c = np.array([np.cos(ang) * rr / 2, np.sin(ang) * rr / 2, side * (w / 2 - 0.035)])
        v, t = _box(c, (rr, 0.05, 0.03))
        rot = np.array([[np.cos(ang), np.sin(ang), 0], [-np.sin(ang), np.cos(ang), 0], [0, 0, 1]])
        v = (v - c) @ rot + c
        spokes_t += [tuple(i + len(spokes_v) * 8 for i in tri) for tri in t]
        spokes_v.append(v)
    out = {"tyre": tyre, "rim": rim, "spoke": (np.concatenate(spokes_v), spokes_t)}
    return {m: (np.asarray(v) + hub, t) for m, (v, t) in out.items()}, hub


# --- the car -------------------------------------------------------------------------------------------------


def build_car(p: CarParams | None = None) -> Car:
    p = p or CarParams()
    x_front_axle = p.length / 2 - p.front_overhang
    axles = (x_front_axle, x_front_axle - p.wheelbase)
    u, P, raised, halfw = _body_grid(p, axles)
    verts, tris, mats, owners = _body_faces(p, u, P, raised)

    parts: dict[str, Part] = {}

    def piece(part_name, mat, v, t):
        part = parts[part_name]
        if mat in part.pieces:
            old = part.pieces[mat]
            n = len(old.positions)
            v = np.concatenate([old.positions, v])
            t = np.concatenate([old.faces, np.asarray(t) + n])
        part.pieces[mat] = _mesh(v, t)

    L = p.length
    def xu(uu):
        return L / 2 - uu * L
    def sections_at(uu):
        return P[np.argmin(np.abs(u - uu))]
    parts["body"] = Part("body", "body")
    for side, s in (("l", -1), ("r", 1)):
        for door, u0 in (("f", p.cowl_u + 0.01), ("r", p.b_pillar_u)):
            ring = sections_at(u0)
            parts[f"door_{door}{side}"] = Part(f"door_{door}{side}", "door",
                                              pivot=np.array([xu(u0), p.belt_front, s * ring[:, 2].max()]),
                                              open_angle=s * 65.0)
    parts["hood"] = Part("hood", "hood", pivot=np.array([xu(p.cowl_u), p.cowl_height - 0.03, 0.0]),
                         axis=np.array([0.0, 0.0, 1.0]), open_angle=50.0)
    parts["hatch"] = Part("hatch", "hatch", pivot=np.array([xu(p.roof_end_u - 0.01), p.height - p.roof_end_drop, 0.0]),
                          axis=np.array([0.0, 0.0, 1.0]), open_angle=-75.0)

    # Body faces, split by part and material; each part keeps only the vertices it uses.
    for name in parts:
        for mat in np.unique(mats[owners == name]):
            t = tris[(owners == name) & (mats == mat)]
            used, inv = np.unique(t, return_inverse=True)
            piece(name, str(mat), verts[used], inv.reshape(-1, 3))
    for mat, owner, v, t in _decals(p, verts, tris, owners):
        piece(owner, mat, v, t)

    if p.mirrors:
        uu = p.cowl_u + 0.035
        for side, s in (("l", -1), ("r", 1)):
            zr = sections_at(uu)[:, 2].max() * s
            v1, t1 = _box([xu(uu), p.belt_front + 0.07, zr + s * 0.10], (0.10, 0.11, 0.15))
            v2, t2 = _box([xu(uu) + 0.02, p.belt_front + 0.04, zr + s * 0.04], (0.05, 0.03, 0.08))
            piece(f"door_f{side}", "paint", v1, t1)
            piece(f"door_f{side}", "black", v2, t2)
    if p.spoiler:
        ring = sections_at(p.roof_end_u)
        v, t = _box([xu(p.roof_end_u) - 0.05, p.height - p.roof_end_drop + 0.005, 0.0],
                    (0.12, 0.025, 2 * 0.9 * ring[:, 2].max() * (1 - p.tumblehome / 2)), pitch=np.radians(-6))
        piece("hatch", "paint", v, t)

    for xa, front in ((axles[0], True), (axles[1], False)):
        for side, s in (("l", -1), ("r", 1)):
            name = f"wheel_{'f' if front else 'r'}{side}"
            meshes, hub = _wheel(p, xa, s)
            parts[name] = Part(name, "wheel", pivot=hub, axis=np.array([0.0, 0.0, 1.0]), steer=front)
            for mat, (v, t) in meshes.items():
                piece(name, mat, v, t)
    return Car(p, [pt for pt in parts.values() if pt.pieces])


# The materials a car uses, as (sRGB colour, roughness, metallic); "lamp" is drawn unlit when the lights are on.
PALETTE = {
    "paint": ("#8fbccc", 0.32, 0.15), "black": ("#0b0c0e", 0.15, 0.0), "glass": ("#1d262c", 0.05, 0.0),
    "trim": ("#17181a", 0.75, 0.0), "underbody": ("#101010", 0.9, 0.0), "liner": ("#0e0e0e", 0.95, 0.0),
    "lamp": ("#eef3ff", 0.2, 0.0), "tail": ("#c41a12", 0.25, 0.0), "plate": ("#e3e8ea", 0.6, 0.0),
    "chrome": ("#d6dbe0", 0.1, 1.0), "tyre": ("#141414", 0.9, 0.0), "rim": ("#2a2c30", 0.45, 0.3),
    "spoke": ("#c8cdd2", 0.2, 1.0),
}


# --- in an engine --------------------------------------------------------------------------------------------


def spawn_car(engine, car: Car, at=(0.0, 0.0, 0.0), yaw=0.0, palette=None, lights=True) -> dict:
    """Spawn every piece of the car; returns {part name: [entity handles]}. With lights, lamps draw unlit."""
    from manifoldx.components import Material, Mesh as MeshComponent, Transform
    from manifoldx.resources import BasicMaterial, StandardMaterial

    pal = {**PALETTE, **(palette or {})}
    handles = {}
    for part in car.parts:
        handles[part.name] = []
        for mat, mesh in part.pieces.items():
            colour, rough, metal = pal[mat]
            m = BasicMaterial(color=colour) if (lights and mat in ("lamp", "tail")) else \
                StandardMaterial(color=colour, roughness=rough, metallic=metal, opacity=0.55 if mat == "glass" else 1.0)
            handles[part.name].append(engine.spawn(MeshComponent(mesh.to_geometry()), Material(m), Transform()))
    place_car(handles, car, at, yaw)
    return handles


def place_car(handles: dict, car: Car, at=(0.0, 0.0, 0.0), yaw=0.0, **pose) -> None:
    """Move a spawned car to `at`, turned by `yaw` radians about Y, posed as in Car.pose."""
    qy = _axis_angle(np.array([0.0, 1.0, 0.0]), yaw)
    for name, (pos, q) in car.pose(**pose).items():
        world_pos = np.asarray(at, float) + _rotate(qy, pos)
        world_rot = _quat_mul(qy, q)
        for h in handles[name]:
            h.transform.pos = tuple(world_pos)
            h.transform.rot = tuple(world_rot)
