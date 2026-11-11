"""A parametric car: named measurements in, an articulated low-poly car out.

The body is a loft of cross-sections along the car. Each section is a fixed template (floor, rocker, shoulder,
belt, glass, roof) scaled by smooth side-profile and plan curves; the curves come from measurements a spec sheet
or a photo gives: length, width, height, wheelbase, overhangs, nose and tail heights, where the hood meets the
windscreen, where the roof ends. Wheel arches are cut by raising the skin above each wheel, so the loft needs no
booleans.

Regions are tags on the loft's faces, in its own coordinates (position along the car, band around the section),
so they follow any shape: glass, pillars, doors, hood and hatch. Small parts that sit on the skin (lamps, grille,
plate, handles) are decals: flat outlines projected onto the body along a view direction, so a lamp style is
just an outline. Lamps, the grille, wheels, door mirrors, the dashboard and the steering wheel take a shape: outlines traced from a flat drawing of the real
part (`assets.lamps`, `assets.grilles`, `assets.wheels`, `assets.mirrors`, `assets.dashboards`,
`assets.steering`, `assets.doors`, `assets.consoles`), or a plain style built from boxes and ellipses.

Doors, hood and hatch are separate parts with a hinge; wheels spin and the front ones steer. `Car.pose` gives
each part's transform for an opening, a wheel turn and a steering angle.

Axes: metres, Y up, the car's front at +X, its right side at +Z, its centre on the ground at the origin.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from manifoldx.modeling.assets.consoles import CONSOLES
from manifoldx.modeling.assets.dashboards import DASHBOARDS
from manifoldx.modeling.assets.doors import DOORS
from manifoldx.modeling.assets.grilles import GRILLES
from manifoldx.modeling.assets.lamps import LAMPS
from manifoldx.modeling.assets.mirrors import MIRRORS
from manifoldx.modeling.assets.steering import STEERING
from manifoldx.modeling.assets.wheels import WHEELS
from manifoldx.modeling._poly import _triangulate
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
    rim_ratio: float = 0.70  # rim radius over wheel radius
    arch_gap: float = 0.045  # between the tyre and the arch
    # Side profile.
    nose_top: float = 0.75  # top of the nose at the very front
    hood_front: float = 0.925  # the hood just behind the nose
    nose_blunt: float = 0.4  # how much of the rise from nose to hood the face makes in its first 4 cm: 0 is a wedge
    cowl_u: float = 0.29  # where the hood meets the windscreen
    cowl_height: float = 1.20
    windscreen_u: float = 0.46  # where the windscreen meets the roof
    roof_peak_u: float = 0.56  # where the roof is highest
    roof_end_u: float = 0.78  # where the roof meets the rear glass
    roof_end_drop: float = 0.08
    hatch_u: float = 0.92  # where the rear glass ends
    hatch_height: float = 1.165
    tail_top: float = 0.91  # top of the tail at the rear end
    belt_front: float = 1.03  # belt (bottom of the side glass) at the cowl
    belt_rear: float = 1.11  # and at the end of the roof
    sill: float = 0.20  # underside between the wheels
    bumper_bottom_front: float = 0.36
    bumper_bottom_rear: float = 0.35
    hood_crown: float = 0.13  # how far the hood's edge sits below its crown
    # Plan and section.
    nose_half: float = 0.76  # half-width at the front end, over the half-width
    tail_half: float = 0.80
    round_front: float = 0.18  # how far the front corners are pulled back (m)
    round_rear: float = 0.19
    round_power: float = 2.1
    nose_bulge: float = 0.02  # how far the middle of the nose stands proud of its edge (m)
    tail_bulge: float = 0.03
    tumblehome: float = 0.31  # how much narrower the roof is than the body
    shoulder: float = 0.95  # belt half-width over the half-width
    # Greenhouse.
    a_pillar_foot: float = 0.40  # how far up the windscreen the side glass starts, 0 at the cowl to 1 at the roof
    b_pillar_u: float = 0.565
    b_pillar_width: float = 0.10
    c_pillar_u: float = 0.69  # end of the rear door glass; behind it the quarter is black
    black_pillars: bool = True
    roof: str = "paint"  # or "black"
    # Parts.
    # A lamp is a shape from `lamps.LAMPS` by name (or a shape of your own), or one of the plain outlines. A shape
    # is placed by its anchor, sized in its own plane and turned `yaw` degrees from straight ahead toward its side
    # of the car, so a lamp on a corner is drawn as it is seen straight on.
    headlamp: str | dict = "wedge"  # a shape, or "almond", "round" or "bar"
    headlamp_center: tuple = (0.505, 0.826)  # (lateral, height) of the lamp's anchor: here, the end of the light bar
    headlamp_size: tuple = (0.36, 0.185)
    headlamp_yaw: float = 30.0
    drl_bar: bool = True
    drl_height: float = 0.826
    drl_half_length: float = 0.505  # to the inner edge of the headlamps, whose own strip carries it on
    grille: tuple = (0.83, 0.37, 0.50)  # half-width, bottom, top
    grille_style: str | dict = "smile"  # a shape, or "bar"
    taillamp: str | dict = "ring"  # a shape, or "pill"
    taillamp_center: tuple = (0.68, 1.03)
    taillamp_size: tuple = (0.33, 0.22)
    taillamp_yaw: float = 35.0
    spoiler: bool = True
    mirrors: bool = True
    mirror: str | dict = "pod"  # a shape, or "box"
    mirror_size: tuple = (0.20, 0.165)  # across and up, stalk included, as seen from ahead of the car
    mirror_yaw: float = 22.0  # degrees its outer end is swept back
    wheel: str | dict = "swirl"  # a shape, or "spokes"
    spokes: int = 5  # of the plain wheel
    dash: str | dict = "deck"  # a shape, or "box"
    steering: str | dict = "tri"  # a shape, or "ring"
    door_card: str | dict | None = "quilt"  # a shape, or None for bare doors
    console: str | dict | None = "tray"  # a shape, or None for no console
    cladding: bool = False  # black lower body
    handles: bool = True  # a chrome bar on each door; leave it out when a texture paints the handles
    door_strip: bool = True  # a black strip with a chrome line low on the doors
    fog_rings: bool = True  # a lit ring at each end of the grille
    # Resolution and shading.
    sections: int = 28
    crease: float = 40.0  # degrees: an edge that turns more than this is shaded sharp


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
            (roof, top - 0.20 * upper), (0.80 * roof, top - 0.06 * upper), (0.42 * roof, top - 0.012 * upper),
            (0, top)]


def _stations(p: CarParams, axles):
    """Section positions as fractions of the length: denser at the ends, plus a fan over each wheel arch and one
    at every edge between regions, so pillars and panel gaps fall on a section."""
    u = 0.5 - 0.5 * np.cos(np.pi * np.linspace(0, 1, p.sections))
    bw = p.b_pillar_width / 2 / p.length
    u = np.concatenate([u, [p.cowl_u - 0.01, p.cowl_u + 0.01, p.windscreen_u, p.b_pillar_u - bw, p.b_pillar_u,
                            p.b_pillar_u + bw, p.c_pillar_u, p.roof_end_u - 0.01, p.hatch_u]])
    R = p.wheel_radius + p.arch_gap
    for xa in axles:
        ua = (p.length / 2 - xa) / p.length
        u = np.concatenate([u, ua + R * np.sin(np.linspace(-np.pi / 2, np.pi / 2, 9)) / p.length,
                            [ua - (R + 0.02) / p.length, ua + (R + 0.02) / p.length]])
    u = np.sort(u)
    edges = np.isin(u, [p.cowl_u - 0.01, p.cowl_u + 0.01, p.windscreen_u, p.b_pillar_u - bw, p.b_pillar_u,
                        p.b_pillar_u + bw, p.c_pillar_u, p.roof_end_u - 0.01, p.hatch_u])
    keep = np.ones(len(u), bool)
    for k in range(1, len(u)):  # drop a section too close to the one kept before it, unless it is a region edge
        last = np.nonzero(keep[:k])[0][-1]
        if u[k] - u[last] < 0.004:
            keep[k if not edges[k] or edges[last] else last] = False
    keep[[0, -1]] = True
    return u[keep]


def _profiles(p: CarParams, u):
    half = p.width / 2
    top = _pchip([0, 0.01, 0.04, p.cowl_u, p.windscreen_u, p.roof_peak_u, p.roof_end_u, p.hatch_u, 1],
                 [p.nose_top, p.nose_top + p.nose_blunt * (p.hood_front - p.nose_top), p.hood_front, p.cowl_height, p.height - 0.015, p.height,
                  p.height - p.roof_end_drop, p.hatch_height, p.tail_top], u)
    fo, ro = p.front_overhang / p.length, 1 - (p.front_overhang + p.wheelbase) / p.length
    bottom = _pchip([0, 0.4 * fo, 1 - 0.6 * ro, 1], [p.bumper_bottom_front, p.sill, p.sill, p.bumper_bottom_rear], u)
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
    # Leave room for the domes that close the ends, so the car keeps its length.
    P[..., 0] = P[..., 0] * (1 - (p.nose_bulge + p.tail_bulge) / p.length) + (p.tail_bulge - p.nose_bulge) / 2
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


def _hj(k):
    """Ring index to its place in the half section, 0 at the bottom centre to HALF - 1 at the roof centre."""
    k %= RING
    return k if k < HALF else RING - k


def _band(j):
    """The band (0-10) of the face that starts at ring index j and goes to j + 1."""
    return min(_hj(j), _hj(j + 1))


def _body_faces(p: CarParams, u, P, raised):
    """Triangles of the loft, each with a material and a part."""
    S = len(u)
    verts = P.reshape(-1, 3)
    def vid(i, j):
        return i * RING + j % RING
    tris, mats, parts, bands = [], [], [], []
    for i in range(S - 1):
        um = (u[i] + u[i + 1]) / 2
        for j in range(RING):
            b = _band(j)
            side = "r" if P[i, j % RING, 2] + P[i, (j + 1) % RING, 2] > 0 else "l"
            # A material per triangle, from its centroid, so a slanted edge like the A-pillar steps half as coarsely.
            h0, h1 = _hj(j), _hj(j + 1)
            # The quad's diagonal is mirrored on the other half of the ring, so both sides of the car match.
            far = j >= HALF - 1
            va, vb = ((2 * h0 + h1) / 3, (h0 + 2 * h1) / 3) if far else ((h0 + 2 * h1) / 3, (2 * h0 + h1) / 3)
            pair = [_region(p, um, b, (2 * u[i] + u[i + 1]) / 3, va), _region(p, um, b, (u[i] + 2 * u[i + 1]) / 3, vb)]
            part = "body"
            if raised[i, j] and raised[i, (j + 1) % RING] and raised[i + 1, j] and raised[i + 1, (j + 1) % RING]:
                pair = ["liner", "liner"]
            if b in range(2, 8):
                if p.cowl_u + 0.01 <= um < p.b_pillar_u:
                    part = f"door_f{side}"
                elif p.b_pillar_u <= um < p.c_pillar_u:
                    part = f"door_r{side}"
            if 0.035 <= um < p.cowl_u - 0.005 and b >= 6:
                part = "hood"
            if um >= p.roof_end_u - 0.01 and b >= 8:
                part = "hatch"
            a, c, d, e = vid(i, j), vid(i, j + 1), vid(i + 1, j + 1), vid(i + 1, j)
            tris += [(a, c, e), (c, d, e)] if far else [(a, c, d), (a, d, e)]
            mats += pair
            parts += [part, part]
            bands += [b, b]
    # Close each end: copies of the end ring shrinking toward its middle, then a fan. Seen from above the end
    # keeps the rounding of the corners (a cone would come to a point), and the bulge pushes its middle out a
    # little, flat in the centre. The nose and tail are bumper paint.
    verts = list(verts)
    U, J = list(np.repeat(u, RING)), list(np.tile(np.arange(RING), S))
    for i, sign, bulge in ((0, 1, p.nose_bulge), (S - 1, -1, p.tail_bulge)):
        ring = P[i]
        c = np.array([ring[:, 0].max() if sign > 0 else ring[:, 0].min(), (ring[:, 1].min() + ring[:, 1].max()) / 2, 0.0])
        outer = [vid(i, j) for j in range(RING)]
        wide = np.abs(ring[:, 2]).max()
        pull = np.abs(c[0] - ring[np.abs(ring[:, 2]).argmax(), 0])  # how far the corners sit behind the middle
        for k in (0.8, 0.6, 0.4, 0.2, 0.0):
            inner = []
            for j in range(RING if k else 1):
                q = c + k * (ring[j] - c)
                q[0] = c[0] - sign * pull * (abs(q[2]) / wide) ** p.round_power + sign * bulge * (1 - k**4)
                verts.append(q)
                U.append(float(i > 0))
                J.append(j)
                inner.append(len(verts) - 1)
            for j in range(RING):
                o0, o1, n0, n1 = outer[j], outer[(j + 1) % RING], inner[j % len(inner)], inner[(j + 1) % len(inner)]
                quad = [(n0, n1, o1), (n0, o1, o0)] if sign > 0 else [(o0, o1, n1), (o0, n1, n0)]
                for t in quad if k else quad[1:] if sign > 0 else quad[:1]:
                    tris.append(t)
                    mats.append("paint")
                    parts.append("body")
                    bands.append(-1)
            outer = inner
    verts = np.array(verts)
    tris = np.array(tris)
    # Outward normals: the enclosed volume must be positive.
    a, b, c = verts[tris[:, 0]], verts[tris[:, 1]], verts[tris[:, 2]]
    if np.einsum("ij,ij->i", a, np.cross(b, c)).sum() < 0:
        tris = tris[:, ::-1]
    return verts, tris, np.array(mats), np.array(parts), np.array(bands), (np.array(U), np.array(J))


def _region(p: CarParams, u, b, ut, v):
    """The material of a triangle whose quad sits at length fraction u in band b; ut and v are the triangle's own
    centroid along the car and around the section (in bands), for the one edge that runs slanted."""
    if b <= 1:
        return "underbody"
    if b == 2 and p.cladding:
        return "trim"
    if u < p.cowl_u - 0.01 or b <= 5:
        return "paint"
    roof = "black" if p.roof == "black" else "paint"
    pillar = "black" if p.black_pillars else "paint"
    if u < p.windscreen_u:
        # Beside the windscreen the side glass sits under the A-pillar, a line that climbs from the belt to the roof.
        t = (ut - p.cowl_u) / (p.windscreen_u - p.cowl_u)
        return "glass" if b >= 8 or v < 6 + 2 * (t - p.a_pillar_foot) / (1 - p.a_pillar_foot) - 0.25 else pillar
    if u < p.roof_end_u:
        if b >= 8:
            return roof
        if u >= p.c_pillar_u or abs(u - p.b_pillar_u) < p.b_pillar_width / 2 / p.length:
            return pillar
        return "glass"
    if u < p.hatch_u:
        return "glass" if b >= 8 else pillar
    return "paint"


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


def _split(outline, metres=1.0, step=0.04, least=4):
    """Triangulate a flat outline, then split each triangle k x k times so its edges are about `step` long
    (`metres` is the size of one unit of the outline). Returns the points and the triangles."""
    flat = np.asarray(outline, float)
    pts2, tri = [], []
    for t3 in _triangulate(flat):
        a2, b2, c2 = flat[list(t3)]
        longest = max(np.linalg.norm((q - r) * metres) for q, r in ((a2, b2), (b2, c2), (c2, a2)))
        k = int(np.clip(longest / step, least, 40))
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
    return np.array(pts2), np.array(tri)


def _span(poly, at):
    """Where the line through `at` on its first axis crosses an outline: the lowest and the highest crossing on
    the second axis, or None when it misses."""
    lo, hi = poly, np.roll(poly, -1, axis=0)
    cross = (lo[:, 0] - at) * (hi[:, 0] - at) < 0
    if not cross.any():
        return None
    y = lo[cross, 1] + (at - lo[cross, 0]) / (hi[cross, 0] - lo[cross, 0]) * (hi[cross, 1] - lo[cross, 1])
    return y.min(), y.max()


def _prism(outline, lo, hi):
    """A flat outline (u, v) made solid between two offsets along w: the face at `hi`, its back at `lo` and the
    wall between them, each with corners of its own so the edges stay sharp. Returns (u, v, w) and triangles."""
    q = np.asarray(outline, float)
    if np.sum(q[:, 0] * np.roll(q[:, 1], -1) - np.roll(q[:, 0], -1) * q[:, 1]) < 0:
        q = q[::-1]
    n = len(q)
    top, back = np.column_stack([q, np.full(n, hi)]), np.column_stack([q, np.full(n, lo)])
    face = _triangulate(q)
    wall = [t for i in range(n) for j in [(i + 1) % n] for t in ((2 * n + i, 3 * n + i, 3 * n + j), (2 * n + i, 3 * n + j, 2 * n + j))]
    return np.concatenate([top, back, top, back]), face + [(n + c, n + b, n + a) for a, b, c in face] + wall


def _drape(outline, metres, origin, across, up, toward, verts, tris, lift, step=0.03):
    """Lay a flat outline on a mesh: split it small, put each point at origin + u * across + v * up and drop it
    along `toward` onto the mesh, `lift` short of it. A point past the mesh's edge misses it and takes its
    triangles with it. Returns the points' own (u, v), the points and the triangles, or None."""
    flat, tri = _split(outline, metres, step=step, least=1)
    o = origin + np.outer(flat[:, 0], across) + np.outer(flat[:, 1], up)
    t, _ = _raycast(o, toward, verts, tris)
    ok = np.isfinite(t)
    tri = (np.cumsum(ok) - 1)[tri[ok[tri].all(1)]]
    if not len(tri):
        return None
    flat, pts = flat[ok], o[ok] + (t[ok] - lift)[:, None] * toward
    if np.dot(np.cross(pts[tri[0, 1]] - pts[tri[0, 0]], pts[tri[0, 2]] - pts[tri[0, 0]]), toward) > 0:
        tri = tri[:, ::-1]
    return flat, pts, tri


def _slabs(outline, width):
    """Cut an outline into pieces no wider than `width` along its first axis. A long thin shape then triangulates
    into compact triangles instead of slivers as long as itself, which split into thousands."""
    poly = np.asarray(outline, float)
    lo, hi = poly[:, 0].min(), poly[:, 0].max()
    n = max(1, int(np.ceil((hi - lo) / width)))

    def clip(pts, bound, sign):  # keep what lies on the side where sign * (x - bound) <= 0
        out = []
        for a, b in zip(pts, np.roll(pts, -1, axis=0)):
            da, db = sign * (a[0] - bound), sign * (b[0] - bound)
            if da <= 0:
                out.append(a)
            if da * db < 0:
                out.append(a + (b - a) * da / (da - db))
        return np.array(out).reshape(-1, 2)

    pieces = [clip(clip(poly, lo + (hi - lo) * i / n, -1), lo + (hi - lo) * (i + 1) / n, 1) for i in range(n)]
    return [q for q in pieces if len(q) >= 3]


def _decals(p: CarParams, verts, tris, parts, corner_uv):
    """Flat outlines projected onto the skin: (material, view, outline) -> (material, part, vertices, triangles)."""
    W, L = p.width, p.length
    out = []

    def add(mat, view, outline, lift=0.006, close=None, frame=None, least=4):
        # The outline is split into small triangles, so the decal bends with the skin it lands on instead of
        # cutting through it. A frame (anchor on the skin, view direction, the
        # outline's own right axis, half-size, the outline's anchor, island) places an outline drawn in units of
        # its half-size as seen along that direction, and maps it whole into the island.
        outline, tri = _split(outline, np.array(frame[3:5]) if frame else 1.0, least=1 if frame else least)
        if frame:
            anchor, d, right, a, bh, au, av, island = frame
            o = anchor + np.outer((outline[:, 0] - au) * a, right) - 1.5 * d
            o[:, 1] += (outline[:, 1] - av) * bh
        elif view in ("front", "rear"):
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
        pts = o + (t - lift)[:, None] * d
        owner = parts[k]
        names, counts = np.unique(owner, return_counts=True)
        # Texture coordinates: those of the skin under each point, by its barycentric weights.
        a, b, c = verts[tris[k, 0]], verts[tris[k, 1]], verts[tris[k, 2]]
        hit = o + t[:, None] * d
        v0, v1, v2 = b - a, c - a, hit - a
        d00, d01, d11 = (v0 * v0).sum(1), (v0 * v1).sum(1), (v1 * v1).sum(1)
        d20, d21 = (v2 * v0).sum(1), (v2 * v1).sum(1)
        den = np.where(np.abs(d00 * d11 - d01**2) > 1e-18, d00 * d11 - d01**2, 1.0)
        wb, wc = (d11 * d20 - d01 * d21) / den, (d00 * d21 - d01 * d20) / den
        uv = (1 - wb - wc)[:, None] * corner_uv[k, 0] + wb[:, None] * corner_uv[k, 1] + wc[:, None] * corner_uv[k, 2]
        if np.dot(np.cross(pts[tri[0, 1]] - pts[tri[0, 0]], pts[tri[0, 2]] - pts[tri[0, 0]]), d) > 0:
            tri = tri[:, ::-1]  # face the viewer, not the skin
        if frame:
            uv = _island(island, ((outline[:, 0] + 1) / 2 + 0.06) / 1.12, ((1 - outline[:, 1]) / 2 + 0.06) / 1.12)
        elif view in ("front", "rear"):  # the same projection the skin under it uses in its view's island
            Wv, Hv = p.width + VIEW_MARGIN[1], p.height + VIEW_MARGIN[2]
            uv = _island(view, (Wv / 2 - sign * pts[:, 2]) / Wv, (Hv - pts[:, 1]) / Hv)
        if close:  # a lamp's own island: its box (lateral 0-1, height 0-1), mirrored for the lamp on the other side
            name, (y0, z0, y1, z1), side = close
            uv = _island(name, (side * pts[:, 2] - y0) / (y1 - y0), (z1 - pts[:, 1]) / (z1 - z0))
        out.append((mat, str(names[counts.argmax()]), pts, tri, uv))

    def both(mat, view, outline, lift=0.006, close=None):
        add(mat, view, outline, lift, close and (*close, 1))
        add(mat, view, [(-a, b) for a, b in reversed(outline)], lift, close and (*close, -1))

    def lens(mat, view, outline, rim=0.72, core=0.34, about=None, island=None):
        """A lamp as a lit rim round a dark lens with a chrome core on top. The lens is the outline shrunk toward
        `about` (its middle by default): off-centre, the rim is wide on the far side and thin on the near one.
        The core is the outline shrunk again, or an outline of its own. With an island, the lamp is drawn there
        in close-up instead of in its view."""
        c = np.mean(outline, axis=0) if about is None else np.asarray(about, float)
        lo, hi = np.min(outline, axis=0), np.max(outline, axis=0)
        pad = 0.06 * (hi - lo)
        close = island and (island, (*(lo - pad), *(hi + pad)))
        both(mat, view, outline, close=close)
        both("black", view, [tuple(c + rim * (np.array(q) - c)) for q in outline], 0.009, close)
        if core:
            both("chrome", view, core if isinstance(core, list) else [tuple(c + core * (np.array(q) - c)) for q in outline],
                 0.012, close)

    def unit(shape, end, centre, size, yaw, island, lit):
        """A lamp from a shape, on both sides of the car: at the front (end 1) or the rear (-1), its anchor where a
        look straight along the car meets the skin at `centre`, facing `yaw` degrees toward its side."""
        ang = np.radians(yaw)
        ahead = np.array([-end, 0.0, 0.0])
        for side in (1, -1):
            start = np.array([[end * (L / 2 + 1), centre[1], side * centre[0]]])
            t0, _ = _raycast(start, ahead, verts, tris)
            if not np.isfinite(t0[0]):
                continue
            d = np.array([-end * np.cos(ang), 0.0, -side * np.sin(ang)])
            right = np.array([-end * np.sin(ang), 0.0, side * np.cos(ang)])
            frame = (start[0] + t0[0] * ahead, d, right, size[0] / 2, size[1] / 2, *shape["anchor"], island)
            for k, (m, outline) in enumerate(shape["layers"]):
                add(lit if m == "lit" else m, "unit", outline, 0.006 + 0.002 * k, frame=frame)

    def pill(cx, cy, a, b):
        return [(cx + a * np.sign(np.cos(v)) * abs(np.cos(v)) ** 0.5, cy + b * np.sign(np.sin(v)) * abs(np.sin(v)) ** 0.5)
                for v in np.linspace(0, 2 * np.pi, 20, endpoint=False)]

    head = LAMPS.get(p.headlamp) if isinstance(p.headlamp, str) else p.headlamp
    if head:
        unit(head, 1, p.headlamp_center, p.headlamp_size, p.headlamp_yaw, "headlamp", "lamp")
    else:
        cy, cz = p.headlamp_center
        a, bh = p.headlamp_size[0] / 2, p.headlamp_size[1] / 2
        if p.headlamp == "round":
            eye = [(cy + bh * np.cos(s), cz + bh * np.sin(s)) for s in np.linspace(0, 2 * np.pi, 16, endpoint=False)]
        elif p.headlamp == "bar":
            eye = [(cy - a, cz - bh / 3), (cy + a, cz - bh / 3), (cy + a, cz + bh / 3), (cy - a, cz + bh / 3)]
        else:  # almond: an ellipse drawn to a point on its inner side, toward the middle of the car
            s = np.linspace(0, 2 * np.pi, 20, endpoint=False)
            eye = [(cy + a * np.cos(v), cz + bh * np.sin(v) * (1 - 0.75 * max(0.0, -np.cos(v)))) for v in s]
        lens("lamp", "front", eye, island="headlamp")
    if p.drl_bar:
        h, w = p.drl_height, p.drl_half_length
        add("lamp", "front", [(-w, h - 0.009), (w, h - 0.009), (w, h + 0.009), (-w, h + 0.009)], 0.010)
    gw, g0, g1 = p.grille
    grille = GRILLES.get(p.grille_style) if isinstance(p.grille_style, str) else p.grille_style
    if grille:
        # The shape is as tall as the grille. Its ends keep their proportions and its middle takes up the length.
        scale, ends = (g1 - g0) / 2 * grille["aspect"], grille["ends"]
        for k, (m, outline) in enumerate(grille["layers"]):
            if m == "lit" and not p.fog_rings:
                continue
            u, v = np.asarray(outline, float).T
            across = np.where(abs(u) < ends, u * (gw - (1 - ends) * scale) / ends, np.sign(u) * (gw - (1 - abs(u)) * scale))
            for slab in _slabs(np.column_stack([across, (g0 + g1) / 2 + v * (g1 - g0) / 2]), 0.12):
                add("lamp" if m == "lit" else "trim", "front", slab, 0.006 + 0.003 * k, least=1)
    else:
        add("trim", "front", [(-gw, g0), (gw, g0), (gw * 1.06, g1), (-gw * 1.06, g1)], 0.003)
        if p.fog_rings:
            lens("lamp", "front", pill(0.74 * gw, (g0 + g1) / 2, 0.20 * gw, 0.36 * (g1 - g0)), rim=0.78, core=0.0)
    add("plate", "front", [(-0.26, 0.47), (0.26, 0.47), (0.26, 0.59), (-0.26, 0.59)])
    tail = LAMPS.get(p.taillamp) if isinstance(p.taillamp, str) else p.taillamp
    if tail:
        unit(tail, -1, p.taillamp_center, p.taillamp_size, p.taillamp_yaw, "taillamp", "tail")
    else:
        lens("tail", "rear", pill(*p.taillamp_center, p.taillamp_size[0] / 2, p.taillamp_size[1] / 2), island="taillamp")
    add("plate", "rear", [(-0.26, 0.52), (0.26, 0.52), (0.26, 0.64), (-0.26, 0.64)])
    for view in ("left", "right"):
        for u0, u1, s0, s1 in ((p.cowl_u, p.b_pillar_u, 0.05, 0.0), (p.b_pillar_u, p.c_pillar_u, 0.0, 0.025)):
            x = L / 2 - (u0 + 0.62 * (u1 - u0)) * L
            if p.handles:
                add("chrome", view, [(x - 0.08, 0.93), (x + 0.08, 0.93), (x + 0.08, 0.955), (x - 0.08, 0.955)])
            if p.door_strip:
                x0, x1 = L / 2 - (u1 + s1) * L, L / 2 - (u0 + s0) * L
                add("trim", view, [(x0, 0.40), (x1, 0.40), (x1, 0.47), (x0, 0.47)])
                add("chrome", view, [(x0, 0.47), (x1, 0.47), (x1, 0.485), (x0, 0.485)])
    return out


# --- wheels and small parts ----------------------------------------------------------------------------------


def _wheel(p: CarParams, x, side):
    """Tyre, rim and spokes of one wheel at its hub, as {material: (verts, tris)} in car coordinates. A shape
    gives the wheel its face: the pieces of the face stand on the dish of the rim as solid metal, and the
    shape's layers lie on them."""
    r, w, rr = p.wheel_radius, p.tyre_width, p.wheel_radius * p.rim_ratio
    hub = np.array([x, r, side * (p.track / 2)])
    tyre = _revolve([(rr, -w / 2), (r - 0.035, -w / 2), (r, -w / 2 + 0.035), (r, w / 2 - 0.035),
                     (r - 0.035, w / 2), (rr, w / 2)], 32, side)
    rim = _revolve([(0.0, w / 2 - 0.05), (rr, w / 2 - 0.05), (rr, -w / 2 + 0.03), (0.0, -w / 2 + 0.03)], 32, side)
    out = {"tyre": [tyre], "rim": [rim]}
    shape = WHEELS.get(p.wheel) if isinstance(p.wheel, str) else p.wheel
    if shape:
        names = {"dark": "alloy", "metal": "spoke"}
        top, dish = w / 2 - 0.012, w / 2 - 0.05

        def flat(outline, offset):  # an outline as seen from outside the car, at an offset along the axle
            q = np.asarray(outline, float) * rr
            if np.sum(q[:, 0] * np.roll(q[:, 1], -1) - np.roll(q[:, 0], -1) * q[:, 1]) < 0:
                q = q[::-1]
            return q, np.column_stack([side * q[:, 0], q[:, 1], np.full(len(q), side * offset)])

        for mat, outline in shape["face"]:
            q, v = flat(outline, top)
            n = len(q)
            wall = np.concatenate([v, flat(q / rr, dish)[1]])  # its own corners, so the face keeps a sharp edge
            sides = [t for i in range(n) for j in [(i + 1) % n] for t in ((i, n + i, n + j), (i, n + j, j))]
            out.setdefault(names.get(mat, mat), []).extend([(v, _triangulate(q)), (wall, sides)])
        for k, (mat, outline) in enumerate(shape["layers"]):
            q, v = flat(outline, top + 0.002 * (k + 1))
            out.setdefault(names.get(mat, mat), []).append((v, _triangulate(q)))
    else:
        for k in range(p.spokes):
            ang = 2 * np.pi * k / p.spokes
            c = np.array([np.cos(ang) * rr / 2, np.sin(ang) * rr / 2, side * (w / 2 - 0.035)])
            v, t = _box(c, (rr, 0.05, 0.03))
            rot = np.array([[np.cos(ang), np.sin(ang), 0], [-np.sin(ang), np.cos(ang), 0], [0, 0, 1]])
            out.setdefault("spoke", []).append(((v - c) @ rot + c, t))
    return {m: [(np.asarray(v) + hub, t) for v, t in pieces] for m, pieces in out.items()}, hub


def _mirror(shape, size, side, yaw=0.0):
    """A door mirror from a shape, as (material, verts, tris, normals, (u, v)) pieces about the foot of its
    stalk, on the car's right (side 1) or left (-1). The mirror is a row of sections from the door outward: each
    is as tall as the front view at that point and as deep as the plan, round ahead and above, flat behind where
    the glass is and squarer below. The cap is the part of each section above the lower edge of the front view's painted region. The
    strip and the glass are laid on the result from ahead and from behind. Last, the whole mirror turns `yaw`
    degrees about its foot, its outer end toward the rear."""
    a, bh = size[0] / 2, size[1] / 2
    layers = {m: np.asarray(o, float) for m, o in reversed(shape["layers"][1:])}
    body, cap, plan = np.asarray(shape["layers"][0][1], float), layers["paint"], np.asarray(shape["plan"], float)
    au, av = shape["anchor"]

    def bend(c, n):  # a superellipse's coordinate: n = 2 is a circle, a large n a box
        return np.sign(c) * np.abs(c) ** (2 / n)

    deep = plan[:, 1].max()
    foot = 0.7 * deep  # how far ahead of the glass the stalk stands
    rings, points = [], 14
    for u in -0.985 * np.cos(np.linspace(0.0, np.pi, 16)):
        v0, v1 = _span(body, u)
        over = _span(cap, u)
        d0, d1 = _span(plan, u) or (foot - 0.25 * deep, foot + 0.25 * deep)
        cd, rd, cv, rv = (d0 + d1) / 2, (d1 - d0) / 2, (v0 + v1) / 2, max((v1 - v0) / 2, 1e-6)
        t = np.clip(((over[0] if over else v1) - cv) / rv, -1, 1)  # where the cap ends, on the section's height
        ahead = np.arcsin(np.sign(t) * abs(t) ** (1.3 if t > 0 else 2.5))
        ang = np.concatenate([np.linspace(ahead, np.pi - ahead, points // 2),
                              np.linspace(np.pi - ahead, 2 * np.pi + ahead, points // 2 + 2)[1:-1]])
        d = cd + rd * np.where(np.cos(ang) > 0, bend(np.cos(ang), 2.4), bend(np.cos(ang), 5.0))
        v = cv + rv * np.where(np.sin(ang) > 0, bend(np.sin(ang), 2.6), bend(np.sin(ang), 5.0))
        rings.append(np.column_stack([(d - foot) * a, (v - av) * bh, np.full(points, (u - au) * a), np.full(points, u), v]))
    grid = np.array(rings)  # (stations, points, 5): position, then the front view's (u, v)
    n = len(grid)
    verts = np.concatenate([grid.reshape(-1, 5), grid[[0, -1]].mean(1)])
    tris, mats = [], []
    for i in range(n - 1):
        for j in range(points):
            p0, p1, q0, q1 = i * points + j, i * points + (j + 1) % points, (i + 1) * points + j, (i + 1) * points + (j + 1) % points
            tris += [(p0, q0, q1), (p0, q1, p1)]
            mats += ["paint" if j < points // 2 - 1 else "black"] * 2
    for end, ring in ((n * points, 0), (n * points + 1, (n - 1) * points)):
        for j in range(points):
            tris.append((end, ring + (j + 1) % points, ring + j) if ring == 0 else (end, ring + j, ring + (j + 1) % points))
            mats.append("black")
    tris, mats = np.array(tris), np.array(mats)
    xyz = verts[:, :3] * np.array([1.0, 1.0, side])
    if side > 0:  # as built the faces look inward; mirrored to the left side they look outward
        tris = tris[:, ::-1]
    normals = _mesh(xyz, tris).normals
    area = np.linalg.norm(np.cross(xyz[tris[:, 1]] - xyz[tris[:, 0]], xyz[tris[:, 2]] - xyz[tris[:, 0]]), axis=1)
    out = [(m, xyz, tris[(mats == m) & (area > 1e-9)], normals, verts[:, 3:]) for m in ("paint", "black")]

    def lay(mat, outline, toward, lift):  # an outline of the front view, dropped on the mirror along x
        d = np.array([float(toward), 0.0, 0.0])
        laid = _drape(outline, np.array([a, bh]), np.array([-toward * 1.0, -av * bh, -side * au * a]),
                      np.array([0.0, 0.0, side * a]), np.array([0.0, bh, 0.0]), d, xyz, tris, lift)
        if laid:
            flat, pts, tri = laid
            out.append((mat, pts, tri, np.tile(-d, (len(pts), 1)), flat))

    if "lit" in layers:
        lay("lit", layers["lit"], -1, 0.002)
    glass = np.asarray(shape["glass"], float)
    lay("frame", glass.mean(0) + 1.1 * (glass - glass.mean(0)), 1, 0.002)
    lay("glass", glass, 1, 0.004)
    c, sn = np.cos(np.radians(yaw)), side * np.sin(np.radians(yaw))
    turn = np.array([[c, 0.0, sn], [0.0, 1.0, 0.0], [-sn, 0.0, c]])  # applied to rows: x gives way as z grows outward
    return [(m, v @ turn, t, nrm @ turn, flat) for m, v, t, nrm, flat in out]


# --- texture atlas ------------------------------------------------------------------------------------------

# Islands of the atlas as (s0, t0, s1, t1), with t growing down the image as in glTF. Each face of the skin goes
# to the view that sees it best and is drawn there as that view shows it: the left side (front at the left), the
# right side (front at the right), the top (front at the left, the car's right side up), the front and the rear.
# So every island is a picture of the car a person, a paint program or an image model reads at a glance, and a
# photo taken from that side lines up with it. The right-hand column of the atlas is free for the cabin's views.
ATLAS = {
    "left": (0.01, 0.01, 0.71, 0.29), "right": (0.01, 0.30, 0.71, 0.58), "top": (0.01, 0.59, 0.51, 0.845),
    "front": (0.72, 0.01, 0.99, 0.22), "rear": (0.72, 0.23, 0.99, 0.44), "wheel": (0.72, 0.45, 0.89, 0.62),
    # A lamp is small on the car and the first thing an eye goes to: each kind gets a close-up of its own, shared by
    # the two sides (the left lamp as seen from outside the car, its outer edge at the right).
    "headlamp": (0.72, 0.63, 0.855, 0.708), "taillamp": (0.86, 0.63, 0.99, 0.705),
    "small": (0.72, 0.87, 0.85, 0.99), "interior": (0.86, 0.87, 0.99, 0.99), "bottom": (0.01, 0.96, 0.35, 0.99),
    # The cabin: the dashboard as the driver sees it, the seats and the bench as seen from the front, the carpet
    # from above.
    "dash": (0.52, 0.59, 0.71, 0.71), "seats": (0.52, 0.72, 0.71, 0.855), "bench": (0.52, 0.865, 0.71, 0.97),
    "carpet": (0.22, 0.86, 0.50, 0.95),
    # A door's inner panel as seen from the cabin (the right door's; the left one is its mirror image), shared by
    # the four doors, and the console between the seats from above, the car's front up.
    "door": (0.72, 0.715, 0.99, 0.86), "console": (0.90, 0.45, 0.99, 0.62),
}
VIEW_MARGIN = (0.1, 0.3, 0.1)  # what a view shows beyond the car's length, width and height (m)


def _island(name, s, t):
    s0, t0, s1, t1 = ATLAS[name]
    return np.column_stack([s0 + np.clip(s, 0, 1) * (s1 - s0), t0 + np.clip(t, 0, 1) * (t1 - t0)])


def _planar_uv(v, island, cell=(0, 1)):
    """Project a part side-on (x, y) into its island, fitted to its own bounds; cell (i, n) keeps it to the i-th
    of n columns, so parts that share an island do not overlap."""
    lo, hi = v[:, :2].min(0), v[:, :2].max(0)
    span = np.maximum(hi - lo, 1e-6).max()
    return _island(island, (cell[0] + 0.96 * (v[:, 0] - lo[0]) / span) / cell[1], (hi[1] - v[:, 1]) / span)


def _corner_normals(verts, tris, crease):
    """A normal per triangle corner: the area-weighted mean of the faces round its vertex that turn less than
    `crease` degrees from the corner's own face. Soft bends shade smooth and sharp ones keep an edge."""
    a, b, c = verts[tris[:, 0]], verts[tris[:, 1]], verts[tris[:, 2]]
    area = np.cross(b - a, c - a)
    unit = area / (np.linalg.norm(area, axis=1, keepdims=True) + 1e-12)
    out = np.zeros((len(tris), 3, 3))
    order = np.argsort(tris.ravel(), kind="stable")
    cuts = np.flatnonzero(np.concatenate([[True], np.diff(tris.ravel()[order]) > 0, [True]]))
    for s0, s1 in zip(cuts[:-1], cuts[1:]):
        face, corner = order[s0:s1] // 3, order[s0:s1] % 3
        near = unit[face] @ unit[face].T > np.cos(np.radians(crease))
        out[face, corner] = near @ area[face]
    return out / (np.linalg.norm(out, axis=2, keepdims=True) + 1e-12)


def _body_uvs(p: CarParams, u, verts, tris, bands, loft_of):
    """Per-corner texture coordinates of the skin, outside and inside. Outside, a face is drawn in the view that
    sees it best; inside, it takes the loft's own coordinates (along the car, round the section)."""
    S = len(u)
    pts = verts[tris]
    x, y, z = pts[..., 0].ravel(), pts[..., 1].ravel(), pts[..., 2].ravel()
    n = np.cross(pts[:, 1] - pts[:, 0], pts[:, 2] - pts[:, 0])
    k = int((bands >= 0).sum())  # the loft's triangles come two to a quad: both go where the quad's normal says
    n[:k] = np.repeat(n[0:k:2] + n[1:k:2], 2, axis=0)
    n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-12
    Lv, Wv, Hv = p.length + VIEW_MARGIN[0], p.width + VIEW_MARGIN[1], p.height + VIEW_MARGIN[2]
    def view(name, s, t):
        return _island(name, s, t).reshape(-1, 3, 2)
    views = {
        "left": view("left", (Lv / 2 - x) / Lv, (Hv - y) / Hv), "right": view("right", (Lv / 2 + x) / Lv, (Hv - y) / Hv),
        "top": view("top", (Lv / 2 - x) / Lv, (Wv / 2 - z) / Wv), "bottom": view("bottom", (Lv / 2 - x) / Lv, (Wv / 2 + z) / Wv),
        "front": view("front", (Wv / 2 - z) / Wv, (Hv - y) / Hv), "rear": view("rear", (Wv / 2 + z) / Wv, (Hv - y) / Hv),
    }
    # Bands say most of it: the floor looks down, the roof bands look up, the flank looks sideways. The two bands
    # between belt and roof are glass beside the cabin, but the shoulder of the hood ahead of it and of the tail
    # behind it, which look up.
    caps = bands < 0
    uc = 0.5 - pts[..., 0].mean(1) / p.length
    up = (bands >= 8) | ((bands >= 6) & ((uc < p.cowl_u - 0.01) | (uc >= p.hatch_u)))
    side = np.where(n[:, 2] < 0, "left", "right")
    where = np.where(bands <= 1, "bottom", np.where(up, "top", side))
    where = np.where((n[:, 0] > 0.55) | (caps & (pts[..., 0].mean(1) > 0)), "front", where)
    where = np.where((n[:, 0] < -0.55) | (caps & (pts[..., 0].mean(1) < 0)), "rear", where)
    out = np.zeros((len(tris), 3, 2))
    for name, uv in views.items():
        out[where == name] = uv[where == name]
    grid = verts[: S * RING].reshape(S, RING, 3)
    seg = np.linalg.norm(np.diff(np.concatenate([grid, grid[:, :1]], 1), axis=1), axis=2).mean(0)
    V = np.concatenate([[0], np.cumsum(seg)]) / seg.sum()  # around the ring, by mean arc length; RING + 1 values
    U, J = loft_of  # each vertex's place along the car and round the section
    loft = np.stack([U[tris], V[J[tris]]], -1)
    wrap = loft[..., 1].max(1) - loft[..., 1].min(1) > 0.5  # the quad that closes the ring: 0 means 1 there
    loft[wrap] = np.where(loft[wrap][..., 1:2] < 0.25, loft[wrap] + [0, 1], loft[wrap])
    inner = _island("interior", loft[..., 0].ravel(), loft[..., 1].ravel()).reshape(-1, 3, 2)
    return out, inner


# --- the car -------------------------------------------------------------------------------------------------


def build_car(p: CarParams | None = None) -> Car:
    p = p or CarParams()
    x_front_axle = p.length / 2 - p.front_overhang
    axles = (x_front_axle, x_front_axle - p.wheelbase)
    u, P, raised, halfw = _body_grid(p, axles)
    verts, tris, mats, owners, bands, loft_of = _body_faces(p, u, P, raised)

    parts: dict[str, Part] = {}

    def piece(part_name, mat, v, t, uv=None, n=None, island="small", cell=(0, 1)):
        """Add triangles to a part's mesh of one material. Without uv, the piece is projected flat into its
        atlas island; without normals, it is smooth-shaded on its own."""
        part = parts[part_name]
        v, t = np.asarray(v, float), np.asarray(t)
        n = _mesh(v, t).normals if n is None else n
        uv = _planar_uv(v, island, cell) if uv is None else uv
        if mat in part.pieces:
            old = part.pieces[mat]
            t = np.concatenate([old.faces, t + len(old.positions)])
            v, n, uv = (np.concatenate([a, b]) for a, b in ((old.positions, v), (old.normals, n), (old.uvs, uv)))
        part.pieces[mat] = Mesh(positions=v.astype(np.float32), faces=t.astype(np.uint32),
                                normals=np.asarray(n, np.float32), uvs=np.asarray(uv, np.float32))

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

    # Body faces, split by part and material, one vertex per triangle corner so each face can take its own
    # place in the texture atlas; normals stay smooth across soft bends, taken from the welded skin.
    corner_uv, inner_uv = _body_uvs(p, u, verts, tris, bands, loft_of)
    smooth = _corner_normals(verts, tris, p.crease)
    for name in parts:
        for mat in np.unique(mats[owners == name]):
            sel = (owners == name) & (mats == mat)
            t = tris[sel]
            k = np.arange(3 * len(t)).reshape(-1, 3)
            piece(name, str(mat), verts[t].reshape(-1, 3), k, corner_uv[sel].reshape(-1, 2), smooth[sel].reshape(-1, 3))
            # The inside: the same triangles reversed, so every panel shows from both sides and an open door
            # shows its inner card. Glass stays glass; the rest is interior trim, in its own atlas island.
            piece(name, "glass" if mat == "glass" else "interior", verts[t].reshape(-1, 3), k[:, ::-1],
                  inner_uv[sel].reshape(-1, 2), -smooth[sel].reshape(-1, 3))
    for mat, owner, v, t, uv in _decals(p, verts, tris, owners, corner_uv):
        piece(owner, mat, v, t, uv)

    card = DOORS.get(p.door_card) if isinstance(p.door_card, str) else p.door_card
    if card:
        # The card is fitted to each door, between the floor and the window sill, and dropped on the inside of the
        # skin from the middle of the cabin. Front and rear doors, left and right, share one picture.
        names = {"dark": "interior", "leather": "seat", "armrest": "dash"}
        y0, y1 = p.sill + 0.10, p.belt_front - 0.02
        for door, u0, u1 in (("f", p.cowl_u + 0.03, p.b_pillar_u - 0.02), ("r", p.b_pillar_u + 0.03, p.c_pillar_u - 0.01)):
            a, bh = (u1 - u0) * L / 2, (y1 - y0) / 2
            for side, s in (("l", -1), ("r", 1)):
                out = np.array([0.0, 0.0, float(s)])
                mid = verts[tris].mean(1)  # only the skin by this door is in a ray's way
                near = tris[(s * mid[:, 2] > 0.3) & (abs(mid[:, 0] - xu((u0 + u1) / 2)) < a + 0.3) & (mid[:, 1] > y0 - 0.3) & (mid[:, 1] < y1 + 0.3)]
                laid = [(m, _drape(slab, np.array([a, bh]), np.array([xu((u0 + u1) / 2), (y0 + y1) / 2, 0.0]), np.array([-a, 0.0, 0.0]),
                                   np.array([0.0, bh, 0.0]), out, verts, near, 0.004 + 0.004 * k, step=0.12))
                        for k, (m, outline) in enumerate(card["layers"]) for slab in _slabs(outline, 0.2 / a)]
                for m, (flat, pts, tri) in ((m, d) for m, d in laid if d):
                    piece(f"door_{door}{side}", names.get(m, m), pts, tri,
                          _island("door", 0.03 + 0.47 * (flat[:, 0] + 1), 0.03 + 0.47 * (1 - flat[:, 1])), np.tile(-out, (len(pts), 1)))

    if p.mirrors:
        uu = p.cowl_u + 0.05
        ring = sections_at(uu)[:HALF]
        shape = MIRRORS.get(p.mirror) if isinstance(p.mirror, str) else p.mirror
        for side, s in (("l", -1), ("r", 1)):
            if shape:
                zr = np.interp(p.belt_front + 0.01, ring[:, 1], ring[:, 2])  # the skin at the stalk's foot
                at = np.array([xu(uu), p.belt_front + 0.01, s * (zr - 0.01)])
                # In the island the shell takes three columns of five, as the front view shows it; frame, glass
                # and strip share the fourth, one above the other.
                names = {"frame": ("black", 0), "glass": ("chrome", 1), "lit": ("black", 2)}  # the strip, unlit
                for mat, v, t, n, flat in _mirror(shape, p.mirror_size, s, p.mirror_yaw):
                    su, tv = 0.02 + 0.48 * (flat[:, 0] + 1), (1 - flat[:, 1]) / 2
                    if mat in names:
                        mat, row = names[mat]
                        uv = _island("small", (3 + su) / 5, (row + tv) / 3)
                    else:
                        uv = _island("small", 3 * su / 5, tv)
                    piece(f"door_f{side}", mat, v + at, t, uv, n)
            else:
                zr = s * np.interp(p.belt_front + 0.05, ring[:, 1], ring[:, 2])  # the skin at the mirror's foot
                v1, t1 = _box([xu(uu), p.belt_front + 0.10, zr + s * 0.13], (0.10, 0.12, 0.17))
                v2, t2 = _box([xu(uu) + 0.01, p.belt_front + 0.06, zr + s * 0.03], (0.05, 0.03, 0.10))
                piece(f"door_f{side}", "paint", v1, t1, cell=(0, 5))
                piece(f"door_f{side}", "black", v2, t2, cell=(1, 5))
    if p.spoiler:
        ring = sections_at(p.roof_end_u)
        v, t = _box([xu(p.roof_end_u) - 0.07, p.height - p.roof_end_drop - 0.03, 0.0],
                    (0.16, 0.03, 2 * 0.62 * ring[:, 2].max() * (1 - p.tumblehome)), pitch=np.radians(-12))
        piece("hatch", "paint", v, t, cell=(4, 5))

    _cabin(p, parts, piece, xu)

    for xa, front in ((axles[0], True), (axles[1], False)):
        for side, s in (("l", -1), ("r", 1)):
            name = f"wheel_{'f' if front else 'r'}{side}"
            meshes, hub = _wheel(p, xa, s)
            parts[name] = Part(name, "wheel", pivot=hub, axis=np.array([0.0, 0.0, 1.0]), steer=front)
            # A wheel is drawn in its island as seen from outside the car, so both sides wear the same picture.
            # The dish of the rim lies behind the face: it goes to a spot on the tyre's tread, which is plain and
            # dark, or the face's picture would show again on what one sees through the windows.
            for mat, pieces in meshes.items():
                for v, t in pieces:
                    a, b = s * (v[:, 0] - hub[0]) / (2.2 * p.wheel_radius), (hub[1] - v[:, 1]) / (2.2 * p.wheel_radius)
                    piece(name, mat, v, t, _island("wheel", *((0.946 + 0.012 * a, 0.5 + 0.012 * b) if mat == "rim" else (a + 0.5, b + 0.5))))
    return Car(p, [pt for pt in parts.values() if pt.pieces])


def _dash(shape, width, deck, rear, front, crease):
    """A dashboard from a shape, as (material, verts, tris, normals, shown) pieces in car coordinates: `width` across,
    its rear edge at the height `deck`, its rear face at `rear` and nothing ahead of `front`. The section is
    swept across the car, and at each point across it is cut to the height the front view has there, up to the
    deck, which gives the footwells. The front view's layers are laid on the result from the cabin; a screen
    stands off it, and may rise above the deck. A piece is `shown` when it faces the cabin, so the front view
    is its picture; the deck's top, the ends and the back of a screen are not."""
    a = width / 2
    bh = a / shape["aspect"]
    top = deck + (1 - shape["deck"]) * bh  # where the front view's upper edge would be
    body = np.asarray(shape["layers"][0][1], float)
    sec = np.asarray(shape["section"], float)
    x = np.minimum(rear + (sec[:, 0].max() - sec[:, 0]) * bh, front)
    n, stations = len(sec), np.linspace(-0.999, 0.999, 25)
    rings = []
    for u in stations:
        v0, v1 = _span(body, u)
        rings.append(np.column_stack([x, top - (1 - np.clip(sec[:, 1], v0, min(v1, shape["deck"]))) * bh, np.full(n, u * a)]))
    verts = np.concatenate(rings)
    tris = []
    for i in range(len(stations) - 1):
        for j in range(n):
            p0, p1, q0, q1 = i * n + j, i * n + (j + 1) % n, (i + 1) * n + j, (i + 1) * n + (j + 1) % n
            tris += [(p0, q0, q1), (p0, q1, p1)]
    cap = _triangulate(sec)
    tris += [(c, b, a_) for a_, b, c in cap] + [((len(stations) - 1) * n + a_, (len(stations) - 1) * n + b, (len(stations) - 1) * n + c) for a_, b, c in cap]
    tris = np.array(tris)
    centre = verts.mean(0)
    out_of = np.einsum("ij,ij->i", np.cross(verts[tris[:, 1]] - verts[tris[:, 0]], verts[tris[:, 2]] - verts[tris[:, 0]]),
                       verts[tris].mean(1) - centre).sum()
    if out_of < 0:
        tris = tris[:, ::-1]
    area = np.linalg.norm(np.cross(verts[tris[:, 1]] - verts[tris[:, 0]], verts[tris[:, 2]] - verts[tris[:, 0]]), axis=1)
    tris = tris[area > 1e-10]
    flat_n = np.cross(verts[tris[:, 1]] - verts[tris[:, 0]], verts[tris[:, 2]] - verts[tris[:, 0]])
    faces_cabin = flat_n[:, 0] < -0.5 * np.linalg.norm(flat_n, axis=1)
    smooth = _corner_normals(verts, tris, crease)
    out = [("dash", verts[tris[sel]].reshape(-1, 3), np.arange(3 * sel.sum()).reshape(-1, 3), smooth[sel].reshape(-1, 3), shown)
           for sel, shown in ((faces_cabin, True), (~faces_cabin, False))]
    names = {"dark": "dash", "leather": "seat", "chrome": "chrome"}
    origin, across, up, ahead = np.array([rear - 1.0, top - bh, 0.0]), np.array([0.0, 0.0, a]), np.array([0.0, bh, 0.0]), np.array([1.0, 0.0, 0.0])
    for i, (m, outline) in enumerate(shape["layers"][1:]):
        if m == "screen":  # a slab of its own, 3 cm off the dashboard's rear face; its face looks into the cabin
            v, f = _prism(np.asarray(outline, float) * [a, bh], 0.0, 0.012)  # (z, y, thickness)
            pts, f, front = np.column_stack([rear - 0.03 - v[:, 2], origin[1] + v[:, 1], v[:, 0]]), np.array(f), len(outline) - 2
            out += [("black", pts, f[:front], None, True), ("black", pts, f[front:], None, False)]
            continue
        laid = _drape(outline, np.array([a, bh]), origin, across, up, ahead, verts, tris, 0.003 + 0.002 * i, step=0.08)
        if laid:
            out.append((names[m], laid[1], laid[2], np.tile(-ahead, (len(laid[1]), 1)), True))
    return out


def _steering(shape, radius, thick=0.03):
    """A steering wheel from a shape, as (material, verts, tris, shown) pieces about its hub, in its own frame: x
    to the driver's right, y up, z toward the driver. The face's pieces are solid; the layers lie on its near
    side. A piece is `shown` when it faces the driver, so the shape's own view is its picture."""
    names = {"rim": "dash", "pad": "seat", "mark": "plate"}
    out = []
    for m, outline in shape["face"]:
        v, t = _prism(np.asarray(outline, float) * radius, -thick / 2, thick / 2)
        t, front = np.array(t), len(outline) - 2
        out += [(names.get(m, m), v, t[:front], True), (names.get(m, m), v, t[front:], False)]
    for k, (m, outline) in enumerate(shape["layers"]):
        q = np.asarray(outline, float) * radius
        if np.sum(q[:, 0] * np.roll(q[:, 1], -1) - np.roll(q[:, 0], -1) * q[:, 1]) < 0:
            q = q[::-1]
        out.append((names.get(m, m), np.column_stack([q, np.full(len(q), thick / 2 + 0.002 * (k + 1))]), np.array(_triangulate(q)), True))
    return out


def _cabin(p: CarParams, parts, piece, xu):
    """Carpet, dashboard, steering wheel, two front seats and a rear bench, placed from the cowl, the B-pillar,
    the belt and the sill, so they move with the car's measurements."""
    parts["cabin"] = Part("cabin", "body")
    xc, xb = xu(p.cowl_u), xu(p.b_pillar_u)
    inner = p.width - 0.30
    floor = p.sill + 0.08
    seat_z = 0.22 * p.width

    wide = inner + 0.1

    def seen(v, island, facing, y0, y1):
        """Texture coordinates of a cabin part as seen along the car: facing +1 looks forward, -1 looks back."""
        return _island(island, (wide / 2 + facing * v[:, 2]) / wide, (y1 - v[:, 1]) / (y1 - y0))

    def box(mat, center, size, pitch=0.0, island="seats", facing=-1, y=(floor, floor + 1.15)):
        v, t = _box(center, size, pitch=pitch)
        piece("cabin", mat, v, t, seen(v, island, facing, *y))

    dash_y = (floor, p.belt_front + 0.30)

    def plain(v):
        """Texture coordinates for what the dashboard's view does not show (a top, a back, a side): a small spot
        in a free corner of the island, where nothing is drawn and the material's own colour stays."""
        return _island("dash", 0.93 + 0.02 * (wide / 2 + v[:, 2]) / wide, 0.93 + 0.02 * (v[:, 1] - floor))

    v, t = _box([(xc + xu(p.roof_end_u)) / 2, floor - 0.02, 0.0], (xc - xu(p.roof_end_u) - 0.2, 0.04, inner))
    piece("cabin", "carpet", v, t, _island("carpet", (xc - v[:, 0]) / (xc - xu(p.roof_end_u)), (wide / 2 - v[:, 2]) / wide))
    dash = DASHBOARDS.get(p.dash) if isinstance(p.dash, str) else p.dash
    if dash:
        for mat, v, t, n, shown in _dash(dash, inner, p.belt_front + 0.03, xc - 0.46, xc - 0.02, p.crease):
            piece("cabin", mat, v, t, seen(v, "dash", 1, *dash_y) if shown else plain(v), n)
    else:
        box("dash", [xc - 0.25, p.belt_front - 0.13, 0.0], (0.42, 0.34, inner), island="dash", facing=1, y=dash_y)
    for z in (-seat_z, seat_z):
        box("seat", [xb + 0.30, floor + 0.24, z], (0.50, 0.12, 0.50))
        box("seat", [xb + 0.02, floor + 0.58, z], (0.12, 0.62, 0.50), pitch=0.25)
        box("seat", [xb - 0.08, floor + 0.98, z], (0.10, 0.18, 0.28), pitch=0.25)
    box("seat", [xb - 0.55, floor + 0.24, 0.0], (0.48, 0.12, inner - 0.08), island="bench", y=(floor, floor + 0.9))
    box("seat", [xb - 0.84, floor + 0.55, 0.0], (0.12, 0.58, inner - 0.08), pitch=0.25, island="bench", y=(floor, floor + 0.9))
    tray = CONSOLES.get(p.console) if isinstance(p.console, str) else p.console
    if tray:
        # The console stands between the front seats: its outline made solid, on a foot, with its layers on top.
        # From above it is its picture; its sides and its foot are plain.
        names = {"dark": "dash", "leather": "seat"}
        half = 0.12
        long = half / tray["aspect"]  # half its length
        cx, top = xb + 0.26, floor + 0.40

        def above(q):
            return _island("console", 0.03 + 0.47 * (q[:, 0] / half + 1), 0.03 + 0.47 * (1 - q[:, 1] / long))

        def aside(v):
            return _island("console", 0.02 + 0.02 * (v[:, 2] / half + 1), 0.01 + 0.01 * (v[:, 0] - cx) / long + 0.01)

        for k, (m, outline) in enumerate(tray["layers"]):
            q = np.asarray(outline, float) * [half, long]
            if k == 0:
                v, f = _prism(q, -0.09, 0.0)  # (z, x, y) about the middle of its top
                pts, f, n = np.column_stack([cx + v[:, 1], top + v[:, 2], v[:, 0]]), np.array(f), len(q) - 2
                piece("cabin", names.get(m, m), pts, f[:n], above(v[:, :2]))
                piece("cabin", names.get(m, m), pts, f[n:], aside(pts))
                continue
            if np.sum(q[:, 0] * np.roll(q[:, 1], -1) - np.roll(q[:, 0], -1) * q[:, 1]) < 0:
                q = q[::-1]
            pts = np.column_stack([cx + q[:, 1], np.full(len(q), top + 0.002 * k), q[:, 0]])
            piece("cabin", names.get(m, m), pts, np.array(_triangulate(q)), above(q))
        v, t = _box([cx, (floor + top - 0.09) / 2, 0.0], (1.3 * long, top - 0.09 - floor, 0.14))
        piece("cabin", "dash", v, t, aside(v))

    # Steering wheel: on the driver's side (left), facing the seat with its top leaning away, on a column that
    # runs down and ahead from its hub.
    lean = np.radians(25)
    to_x = np.array([[0, 0, -1], [0, 1, 0], [1, 0, 0]])  # the wheel's axis (Z) turned to point along -X
    tilt = np.array([[np.cos(lean), -np.sin(lean), 0], [np.sin(lean), np.cos(lean), 0], [0, 0, 1]])
    hub = np.array([xc - 0.62, p.belt_front - 0.05, -seat_z])
    wheel = STEERING.get(p.steering) if isinstance(p.steering, str) else p.steering
    if wheel:
        # The wheel stands in front of the dashboard, so it is not drawn into that view: it has a picture of its
        # own, straight on, in the free part of the island under the dashboard.
        for mat, v, t, shown in _steering(wheel, 0.19):
            uv = _island("dash", 0.25 + 0.101 * v[:, 0] / 0.19, 0.83 - 0.16 * v[:, 1] / 0.19) if shown else plain(v + hub)
            piece("cabin", mat, v @ to_x.T @ tilt + hub, t, uv)
    else:
        v, t = _revolve([(0.17 + 0.02 * np.cos(a), 0.02 * np.sin(a)) for a in np.linspace(0, 2 * np.pi, 8, endpoint=False)],
                        24, 1)
        v = v @ to_x.T @ tilt + hub
        piece("cabin", "dash", v, t, seen(v, "dash", 1, *dash_y))
    box("dash", hub + 0.11 * np.array([np.cos(lean), -np.sin(lean), 0.0]), (0.24, 0.05, 0.05), pitch=-lean, island="dash",
        facing=1, y=dash_y)


# The materials a car uses, as (sRGB colour, roughness, metallic); "lamp" is drawn unlit when the lights are on.
PALETTE = {
    "interior": ("#2a2c30", 0.85, 0.0), "carpet": ("#1a1b1d", 0.95, 0.0), "dash": ("#1f2023", 0.6, 0.0),
    "seat": ("#3a3c41", 0.8, 0.0),
    "paint": ("#8fbccc", 0.32, 0.15), "black": ("#0b0c0e", 0.15, 0.0), "glass": ("#1d262c", 0.05, 0.0),
    "trim": ("#17181a", 0.75, 0.0), "underbody": ("#101010", 0.9, 0.0), "liner": ("#0e0e0e", 0.95, 0.0),
    "lamp": ("#eef3ff", 0.2, 0.0), "tail": ("#c41a12", 0.25, 0.0), "plate": ("#e3e8ea", 0.6, 0.0),
    "chrome": ("#d6dbe0", 0.1, 1.0), "tyre": ("#141414", 0.9, 0.0), "rim": ("#2a2c30", 0.45, 0.3),
    "spoke": ("#c8cdd2", 0.2, 1.0), "alloy": ("#33363b", 0.3, 0.8),
}


# --- in an engine --------------------------------------------------------------------------------------------


def spawn_car(engine, car: Car, at=(0.0, 0.0, 0.0), yaw=0.0, palette=None, lights=True, texture=None) -> dict:
    """Spawn every piece of the car; returns {part name: [entity handles]}. With lights, lamps draw unlit.

    texture: the path of an albedo image laid out as ATLAS; it replaces the palette's colours on everything but
    the glass and the lit lamps.
    """
    from manifoldx.components import Material, Mesh as MeshComponent, Transform
    from manifoldx.resources import BasicMaterial, StandardMaterial
    from manifoldx.textures import load_texture

    pal = {**PALETTE, **(palette or {})}
    albedo = load_texture(engine, texture) if texture else None
    handles = {}
    for part in car.parts:
        handles[part.name] = []
        for mat, mesh in part.pieces.items():
            colour, rough, metal = pal[mat]
            if lights and mat in ("lamp", "tail"):
                m = BasicMaterial(color=colour)
            elif mat == "glass":
                m = StandardMaterial(color=colour, roughness=rough, metallic=metal, opacity=0.55)
            else:
                m = StandardMaterial(color=colour, roughness=rough, metallic=metal, albedo_map=albedo)
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
