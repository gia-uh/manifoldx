"""A parametric truck: named measurements in, an articulated lorry out.

A truck is built around a ladder chassis. The cab is either a bonneted one, with a low nose and a short bonnet ahead
of the cab, or a flat-faced one standing over the engine; either way it is one lofted shell with rounded plan corners
and an arch cut over the front wheels the way the car cuts its own. Behind it the body is a parameter: a closed cargo
box with two rear doors, a flat deck with low sides, a horizontal tank with dished ends and a walkway, an open bin
that tips about its rear lower edge, or a fire engine's lockers, roof ladder and hose reel. Two or three axles carry
single front wheels and twin rear wheels.

The truck reuses the car's wheels (``car._wheel``), its material palette and its beacon colours, and the bus's way of
lofting a rounded box. Doors and the dump bin are parts with hinges, so :meth:`Truck.pose` opens the doors, tips the
bin, rolls the wheels and steers the front axle. Paint areas (``livery``) are polygons in the side view laid on both
flanks, and the cab and body have separate colour slots, ``paint`` and ``body``.

Axes are the shared ones: metres, Y up, the front at +X, the right side at +Z, the centre on the ground at the origin.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np

from manifoldx.modeling._poly import _triangulate
from manifoldx.modeling.assets.car import (
    _axis_angle,
    _box,
    _corner_normals,
    _is_colour,
    _mesh,
    _quat_mul,
    _wheel,
)
from manifoldx.modeling.assets.vehicle import PALETTE, ParamsRanges, Part, Vehicle, _rotate, choice, rng
from manifoldx.modeling.mesh import Mesh

# Materials every truck may use, beyond the shared vehicle palette: a body paint of its own, a wooden deck, a tank's
# metal, locker doors, a hose and a roof ladder.
TRUCK_MATERIALS = {
    "body": ("#e8eaec", 0.45, 0.0),
    "deck": ("#8a5a33", 0.85, 0.0),
    "tank": ("#c9ccd0", 0.25, 0.9),
    "locker": ("#b9bdc2", 0.45, 0.6),
    "hose": ("#7a2a2a", 0.85, 0.0),
    "ladder": ("#b9bec4", 0.3, 0.8),
}


@dataclass
class TruckParams(ParamsRanges):
    """Every measurement of the truck. Heights are metres from the ground; lengths are metres along the truck,
    the front at +X."""

    # Package: the spec sheet.
    length: float = rng(7.8, 4.5, 12.0, step=0.01, unit='m')
    width: float = rng(2.45, 1.9, 3.0, step=0.01, unit='m')
    height: float = rng(3.5, 2.2, 4.5, step=0.01, unit='m')  # overall, over the body or the beacons
    axles: int = choice(2, 2, 3)  # 2 or 3
    wheelbase: float = rng(4.35, 2.0, 6.4, step=0.01, unit='m')  # the front axle to the first rear one
    rear_axle_spacing: float = rng(1.35, 0.5, 3.0, step=0.01, unit='m')  # between the rear axles of a three-axle truck
    front_overhang: float = rng(1.25, 0.5, 3.4, step=0.01, unit='m')
    track: float = rng(2.02, 1.4, 2.8, step=0.01, unit='m')
    wheel_radius: float = rng(0.52, 0.3, 0.8, step=0.005, unit='m')
    tyre_width: float = rng(0.28, 0.15, 0.45, step=0.005, unit='m')
    rim_ratio: float = rng(0.62, 0.35, 0.95, step=0.01, unit='')
    arch_gap: float = rng(0.06, 0.0, 0.2, step=0.005, unit='m')
    twin_rear: bool = True  # two wheels on each side of every rear axle
    # Cab.
    cab: str = choice("over", 'over', 'forward')  # "over" (flat-faced over the engine) or "forward" (a bonnet ahead of the cab)
    cab_length: float = rng(1.85, 1.0, 7.5, step=0.01, unit='m')
    cab_height: float = rng(2.65, 1.5, 4.0, step=0.01, unit='m')
    cab_inset: float = rng(0.10, 0.0, 0.5, step=0.005, unit='m')  # the cab is this much narrower than the body
    cab_bottom: float = rng(0.80, 0.3, 1.5, step=0.01, unit='m')
    bonnet_length: float = rng(1.35, 0.5, 3.0, step=0.01, unit='m')  # forward cab only
    bonnet_height: float = rng(1.35, 0.6, 2.5, step=0.01, unit='m')
    nose_height: float = rng(1.10, 0.4, 2.0, step=0.01, unit='m')
    windscreen_rake: float = rng(0.22, 0.0, 0.6, step=0.005, unit='m')  # how far the screen leans back at the top
    mirrors: bool = True
    # Chassis and fittings.
    chassis_top: float = rng(0.82, 0.4, 1.5, step=0.01, unit='m')
    exhaust: bool = False
    fuel_tank: bool = True
    steps: bool = True
    mudguards: bool = True
    # Body.
    body: str = choice("box", 'box', 'flatbed', 'tanker', 'dump', 'fire')  # "box", "flatbed", "tanker", "dump" or "fire"
    body_gap: float = rng(0.08, 0.0, 1.0, step=0.01, unit='m')  # between the cab and the body
    body_height: float = rng(3.5, 2.0, 4.5, step=0.01, unit='m')  # the roof of a box or fire body
    body_floor: float = rng(1.10, 0.5, 2.0, step=0.01, unit='m')  # the underside of a box, fire body or bin
    deck_height: float = rng(1.15, 0.5, 2.0, step=0.01, unit='m')  # the top of a flatbed's deck
    side_height: float = rng(0.42, 0.0, 1.0, step=0.01, unit='m')  # a flatbed's side boards, above the deck
    tank_radius: float = rng(1.0, 0.3, 1.6, step=0.01, unit='m')
    tank_center: float = rng(2.02, 0.5, 2.8, step=0.01, unit='m')
    tank_walkway: bool = True
    bin_height: float = rng(1.35, 0.3, 3.0, step=0.01, unit='m')  # the dump bin's sides, above its floor
    dump_angle: float = rng(50.0, 0.0, 80.0, step=1.0, unit='deg')  # degrees the bin tips at dump = 1
    lockers: int = rng(3, 0, 8, step=1, unit='')  # roller-door lockers on each flank of a fire body
    ladder: bool = False
    hose_reel: bool = False
    beacons: bool = False
    # Colour slots and paint areas: (material name or sRGB hex, polygon in the side view as (u, height)).
    cab_paint: str = field(default="#8fbccc", metadata={"kind": "colour"})
    body_paint: str = field(default="#e8eaec", metadata={"kind": "colour"})
    livery: tuple = ()
    # Resolution and small parts.
    detail: float = rng(1.0, 0.25, 2.0, step=0.05, unit='x')  # scales the stations, wheel segments, slats and rungs; 1 is the default
    crease: float = rng(38.0, 0.0, 180.0, step=1.0, unit='deg')  # degrees: an edge that turns more than this is shaded sharp
    wheel: str | dict = choice("hubcap", 'spokes', 'hubcap')  # a shape, or "spokes" or "hubcap"
    spokes: int = rng(8, 0, 32, step=1, unit='')
    whitewall: float = rng(0.0, 0.0, 0.2, step=0.005, unit='m')

    def _validate_extra(self) -> None:
        if self.cab_length >= self.length:
            raise ValueError(f"TruckParams.cab_length ({self.cab_length}) must be below length ({self.length})")
        if self.wheelbase + self.front_overhang >= self.length:
            raise ValueError(
                f"TruckParams.wheelbase + front_overhang ({self.wheelbase + self.front_overhang:.2f}) "
                f"must be below length ({self.length:.2f})"
            )
        if self.wheelbase <= 2 * self.wheel_radius:
            raise ValueError(
                f"TruckParams.wheelbase ({self.wheelbase}) overlaps two wheel_radius ({self.wheel_radius}) wheels"
            )


@dataclass
class Truck(Vehicle):
    """A built truck: its measurements and its rigid parts. Vehicle is its shared shape, so a client that poses and
    merges every vehicle alike can take a Truck without knowing its body."""

    params: TruckParams

    @property
    def palette(self) -> dict:
        """The shared palette, the truck's own materials, the cab and body paint slots, and a material for each hex
        paint area."""
        extra = {"paint": (self.params.cab_paint, 0.35, 0.10), "body": (self.params.body_paint, 0.45, 0.0)}
        extra.update(TRUCK_MATERIALS)
        for k, (mat, _) in enumerate(self.params.livery):
            if _is_colour(mat):
                extra[f"livery{k}"] = (mat, 0.35, 0.0)
        return {**PALETTE, **extra}

    @property
    def landmarks(self) -> dict:
        """Named points on the truck, in truck coordinates: the wheel centres, the nose, the tail and the roof."""
        p = self.params
        out = {}
        for part in self.parts:
            if part.kind == "wheel":
                out[part.name] = np.asarray(part.pivot, float)
        out["nose"] = np.array([p.length / 2, p.cab_bottom, 0.0])
        out["tail"] = np.array([-p.length / 2, p.chassis_top, 0.0])
        out["roof"] = np.array([0.0, p.height, 0.0])
        return out

    def pose(self, doors=0.0, dump=0.0, wheel_turn=0.0, steer=0.0) -> dict:
        """Each part's (position, quaternion xyzw) so that truck-space vertices land in their posed place.

        doors: 0 shut to 1 fully open (or a dict by door name). dump: 0 down to 1 fully tipped. wheel_turn:
        radians the wheels have rolled. steer: radians the front wheels turn, positive to the left.
        """
        out = {}
        for p in self.parts:
            if p.kind == "door":
                angle = doors.get(p.name, 0.0) if isinstance(doors, dict) else doors
                q = _axis_angle(p.axis, np.radians(p.open_angle) * angle)
            elif p.kind == "bin":
                q = _axis_angle(p.axis, np.radians(p.open_angle) * dump)
            elif p.kind == "wheel":
                q = _axis_angle(np.array([0.0, 0.0, 1.0]), -wheel_turn)
                if p.steer:
                    q = _quat_mul(_axis_angle(np.array([0.0, 1.0, 0.0]), steer), q)
            else:
                q = np.array([0.0, 0.0, 0.0, 1.0])
            out[p.name] = (p.pivot - _rotate(q, p.pivot), q)
        return out


# --- small geometry helpers ----------------------------------------------------------------------------------


def _outward(verts, tris) -> np.ndarray:
    """The triangles turned so a closed shell's faces point outward (its signed volume is positive)."""
    a, b, c = verts[tris[:, 0]], verts[tris[:, 1]], verts[tris[:, 2]]
    if np.einsum("ij,ij->i", a, np.cross(b, c)).sum() < 0:
        return tris[:, ::-1]
    return tris


def _tube(a, b, radius, segments: int) -> tuple:
    """A closed cylinder between two points, with a flat cap at each end."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    axis = b - a
    length = np.linalg.norm(axis)
    if length < 1e-9:
        raise ValueError("a tube needs two distinct ends")
    n = axis / length
    ref = np.array([0.0, 1.0, 0.0]) if abs(n[1]) < 0.9 else np.array([1.0, 0.0, 0.0])
    e1 = np.cross(ref, n)
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(n, e1)
    angles = np.linspace(0.0, 2 * np.pi, max(6, segments), endpoint=False)
    ring = np.cos(angles)[:, None] * e1 + np.sin(angles)[:, None] * e2
    verts = np.concatenate([a + radius * ring, b + radius * ring, a[None], b[None]])
    k, n0 = len(angles), 2 * len(angles)
    tris = []
    for j in range(k):
        j1 = (j + 1) % k
        tris += [(j, j1, k + j1), (j, k + j1, k + j)]
        tris += [(n0, j1, j), (n0 + 1, k + j, k + j1)]
    tris = np.array(tris)
    return verts, _outward(verts, tris)


def _arch(cx, cy, radius, z0, z1, segments: int, a0=0.10 * np.pi, a1=0.90 * np.pi) -> tuple:
    """A mudguard: a curved band over a wheel, a closed rectangular section swept along the arch."""
    angles = np.linspace(a0, a1, max(5, segments))
    outer, inner = radius + 0.07, radius + 0.01
    verts = []
    for t in angles:
        c, s = np.cos(t), np.sin(t)
        for r, z in ((outer, z0), (outer, z1), (inner, z1), (inner, z0)):
            verts.append([cx + r * c, cy + r * s, z])
    verts = np.array(verts)
    n, tris = len(angles), []
    for i in range(n - 1):
        for j in range(4):
            j1 = (j + 1) % 4
            a, b = i * 4 + j, i * 4 + j1
            c, d = (i + 1) * 4 + j1, (i + 1) * 4 + j
            tris += [(a, b, c), (a, c, d)]
    for i, flip in ((0, False), (n - 1, True)):
        quad = [i * 4 + j for j in range(4)]
        t = np.array([(quad[0], quad[1], quad[2]), (quad[0], quad[2], quad[3])])
        tris += (t[:, ::-1] if flip else t).tolist()
    tris = np.array(tris)
    return verts, _outward(verts, tris)


def _profile(keys):
    """A top line along the truck through (x, height) keys, interpolated linearly between them."""
    pairs = sorted(keys)
    xs = np.array([k[0] for k in pairs], float)
    ys = np.array([k[1] for k in pairs], float)
    return lambda x: float(np.interp(x, xs, ys))


def _shell(x0, x1, half, bottom, top, corner, edge, stations, arch=None, detail=1.0) -> tuple:
    """A closed lofted box: `stations` rings along x, each a flat-bottomed section with softened top edges, its
    plan corners rounded by `corner`. `arch` is (x of a wheel, radius) and raises the skin over that wheel, so a
    wheel sits in a cut-out instead of inside the body."""
    r = min(max(corner, 0.0), 0.45 * abs(x0 - x1), 0.45 * half)
    inner0, inner1 = x0 - r, x1 + r
    xs = list(np.linspace(inner1, inner0, max(6, stations))) if inner0 > inner1 else list(np.linspace(x1, x0, 6))
    k = max(2, int(round(max(6, stations) / 6)))
    arc = np.linspace(0.0, np.pi / 2, k + 1)[1:]
    xs += [x0 - r + r * np.sin(t) for t in arc]
    xs += [x1 + r - r * np.sin(t) for t in arc]
    if arch is not None:
        xa, R = arch
        xs += [xa + R * np.sin(t) for t in np.linspace(-np.pi / 2, np.pi / 2, max(5, int(round(9 * detail))))]
        xs += [xa - (R + 0.02), xa + (R + 0.02)]
    xs = np.unique(np.round(np.clip(xs, x1, x0), 6))

    def width(x):
        if x > inner0:
            d = x - inner0
            return half - r + np.sqrt(max(0.0, r * r - d * d))
        if x < inner1:
            d = x - inner1
            return half - r + np.sqrt(max(0.0, r * r - d * d))
        return half

    rings = []
    for x in xs:
        h, top_y = width(x), float(top(x))
        e = min(edge, max(0.025, 0.30 * (top_y - bottom)))
        pts = [(0.0, bottom), (0.62 * h, bottom), (0.90 * h, bottom + 0.06), (h, bottom + 0.20),
               (h, top_y - e), (0.97 * h, top_y - 0.025), (0.86 * h, top_y), (0.45 * h, top_y), (0.0, top_y)]
        rings.append(pts + [(-z, y) for z, y in reversed(pts[1:-1])])
    P = np.zeros((len(xs), len(rings[0]), 3))
    for i, ring in enumerate(rings):
        P[i, :, 2] = [z for z, _ in ring]
        P[i, :, 1] = [y for _, y in ring]
        P[i, :, 0] = xs[i]
    if arch is not None:
        xa, R = arch
        dx = P[..., 0] - xa
        m = (np.abs(dx) < R) & (P[..., 1] < R)
        arch_y = R + np.sqrt(np.clip(R * R - dx * dx, 0.0, None))
        P[..., 1] = np.where(m, arch_y, P[..., 1])
    S, RING = P.shape[:2]
    verts = [P.reshape(-1, 3)]
    tris = []
    for i in range(S - 1):
        for j in range(RING):
            a, b = i * RING + j, i * RING + (j + 1) % RING
            c, d = (i + 1) * RING + (j + 1) % RING, (i + 1) * RING + j
            tris += [(a, b, d), (b, c, d)]
    for i, sign in ((0, 1.0), (S - 1, -1.0)):
        ring = P[i]
        centre = np.array([ring[:, 0].mean(), (ring[:, 1].min() + ring[:, 1].max()) / 2, 0.0])
        verts.append(centre[None])
        c = sum(len(v) for v in verts) - 1
        for j in range(RING):
            a, b = i * RING + j, i * RING + (j + 1) % RING
            tris.append((c, a, b) if sign > 0 else (c, b, a))
    verts = np.concatenate(verts)
    tris = np.array(tris)
    return verts, _outward(verts, tris)


# --- cab and fittings ----------------------------------------------------------------------------------------


def _cab_outer(p: TruckParams, parts, piece):
    """The cab and, for a bonneted truck, the bonnet ahead of it: one lofted shell."""
    x_front, x_rear = p.length / 2, -p.length / 2
    cab_back = x_front - p.cab_length - (p.bonnet_length if p.cab == "forward" else 0.0)
    return x_front, cab_back


def _build_cab(p: TruckParams, parts, piece):
    """The cab shell, its windows, grille, lamps, bumper, mirrors and two doors that open."""
    x_front = p.length / 2
    cab_back = x_front - p.cab_length - (p.bonnet_length if p.cab == "forward" else 0.0)
    cab_half = (p.width - p.cab_inset) / 2
    if p.cab == "forward":
        cowl = cab_back + p.cab_length
        keys = [(cab_back, p.cab_height), (cowl - 0.02, p.cab_height - 0.01),
                (cowl + p.windscreen_rake, p.bonnet_height), (x_front - 0.12, p.bonnet_height),
                (x_front, p.nose_height)]
    else:
        keys = [(cab_back, p.cab_height), (x_front - 0.06, p.cab_height - 0.01), (x_front, p.cab_height * 0.995)]
    top = _profile(keys)
    R = p.wheel_radius + p.arch_gap
    xa = x_front - p.front_overhang
    verts, tris = _shell(x_front, cab_back, cab_half, p.cab_bottom, top, 0.12, 0.10,
                         max(10, int(round(22 * p.detail))), arch=(xa, R), detail=p.detail)
    piece("body", "paint", verts, tris)

    # Windscreen, with a black frame behind the glass.
    y0, y1 = p.cab_bottom + 0.72, p.cab_height - 0.20
    if p.cab == "forward":
        cowl = cab_back + p.cab_length

        def x_at(y):
            return cowl + p.windscreen_rake * (p.cab_height - y) / max(p.cab_height - p.bonnet_height, 1e-6)

        lean = np.arctan2(p.windscreen_rake, p.cab_height - p.bonnet_height)
        xc = (x_at(y0) + x_at(y1)) / 2 - 0.005
        v, t = _box([xc, (y0 + y1) / 2, 0.0], (0.035, (y1 - y0) / np.cos(lean), 1.76 * cab_half), pitch=lean)
    else:
        xc = x_front - 0.012
        v, t = _box([xc, (y0 + y1) / 2, 0.0], (0.030, y1 - y0, 1.76 * cab_half))
    piece("body", "black", v, t)
    if p.cab == "forward":
        v, t = _box([xc + 0.018, (y0 + y1) / 2, 0.0], (0.020, (y1 - y0) / np.cos(lean) - 0.06, 1.64 * cab_half),
                    pitch=lean)
    else:
        v, t = _box([xc + 0.018, (y0 + y1) / 2, 0.0], (0.020, y1 - y0 - 0.06, 1.64 * cab_half))
    piece("body", "glass", v, t)
    # A centre bar down the screen, as a truck's two-pane windscreen has.
    v, t = _box([xc + 0.028, (y0 + y1) / 2, 0.0], (0.012, y1 - y0 - 0.06, 0.045))
    piece("body", "black", v, t)

    # Grille and headlamps on the front face.
    if p.cab == "over":
        grille = (0.64 * cab_half, 1.02, 1.36)
        lamps = (0.90 * cab_half, 0.90)
    else:
        grille = (0.44 * cab_half, 0.86, max(0.90, p.nose_height - 0.06))
        lamps = (0.70 * cab_half, 0.95)
    gw, g0, g1 = grille
    mid = (g0 + g1) / 2
    v, t = _box([x_front - 0.022, mid, 0.0], (0.012, g1 - g0 + 0.06, 2 * gw + 0.06))
    piece("body", "chrome", v, t)
    v, t = _box([x_front - 0.010, mid, 0.0], (0.018, g1 - g0, 2 * gw))
    piece("body", "trim", v, t)
    bars = max(2, int(round(3 * p.detail)))
    for k in range(bars):
        y = g0 + (g1 - g0) * (k + 0.5) / bars
        v, t = _box([x_front - 0.004, y, 0.0], (0.014, 0.018, 2 * gw - 0.06))
        piece("body", "chrome", v, t)
    lz, ly = lamps
    for side in (-1.0, 1.0):
        v, t = _box([x_front - 0.025, ly, side * lz], (0.040, 0.20, 0.26))
        piece("body", "black", v, t)
        v, t = _box([x_front - 0.006, ly, side * lz], (0.018, 0.15, 0.20))
        piece("body", "lamp", v, t)
    # Front bumper and plate.
    v, t = _box([x_front - 0.06, 0.58, 0.0], (0.12, 0.22, 0.94 * p.width))
    piece("body", "chrome", v, t)
    v, t = _box([x_front - 0.02, 0.58, 0.0], (0.012, 0.11, 0.42))
    piece("body", "plate", v, t)

    if p.mirrors:
        x = x_front - 0.32
        for side in (-1.0, 1.0):
            v, t = _box([x, 2.02, side * (cab_half + 0.055)], (0.045, 0.045, 0.12))
            piece("body", "black", v, t)
            v, t = _box([x, 2.02, side * (cab_half + 0.14)], (0.09, 0.28, 0.055))
            piece("body", "black", v, t)
    _cab_doors(p, parts, piece, x_front, cab_back, cab_half)


def _cab_doors(p: TruckParams, parts, piece, x_front, cab_back, cab_half):
    """A door on each side, hinged at its front edge, with a window and a handle."""
    cab_front = x_front - (p.bonnet_length if p.cab == "forward" else 0.0)
    width = min(0.88, 0.55 * p.cab_length)
    hinge = cab_front - 0.16
    back = max(hinge - width, cab_back + 0.06)
    y0, y1 = p.cab_bottom + 0.20, min(p.cab_height - 0.20, p.cab_bottom + 1.55)
    mid = (hinge + back) / 2
    for side, sign in (("l", -1.0), ("r", 1.0)):
        name = f"door_cab_{side}"
        parts[name] = Part(name, "door", pivot=np.array([hinge, y0, sign * cab_half]),
                           axis=np.array([0.0, 1.0, 0.0]), open_angle=sign * 70.0)
        z = sign * (cab_half + 0.028)
        v, t = _box([mid, (y0 + 1.52) / 2, z], (hinge - back, 1.52 - y0, 0.05))
        piece(name, "paint", v, t)
        v, t = _box([mid, (1.52 + y1) / 2, z], (hinge - back, y1 - 1.52, 0.04))
        piece(name, "black", v, t)
        v, t = _box([mid, (1.60 + y1) / 2, sign * (cab_half + 0.054)],
                    (hinge - back - 0.10, y1 - 1.60 - 0.05, 0.012))
        piece(name, "glass", v, t)
        v, t = _box([back + 0.10, 1.42, sign * (cab_half + 0.060)], (0.14, 0.035, 0.035))
        piece(name, "chrome", v, t)
        v, t = _box([back + 0.03, (1.52 + y1) / 2, sign * (cab_half + 0.030)], (0.035, y1 - 1.52, 0.05))
        piece(name, "paint", v, t)


def _chassis(p: TruckParams, parts, piece):
    """The ladder chassis, with cross members, a fuel tank, the steps and the rear mudguards."""
    x_front, x_rear = p.length / 2, -p.length / 2
    x0, x1 = x_rear + 0.25, x_front - 0.30
    for z in (-0.38, 0.38):
        v, t = _box([(x0 + x1) / 2, p.chassis_top - 0.09, z], (x1 - x0, 0.18, 0.12))
        piece("body", "underbody", v, t)
    count = max(3, int(round(6 * p.detail)))
    for k in range(count):
        x = x0 + (x1 - x0) * (k + 0.5) / count
        v, t = _box([x, p.chassis_top - 0.09, 0.0], (0.10, 0.14, 0.80))
        piece("body", "underbody", v, t)

    if p.fuel_tank:
        x = x_front - p.front_overhang - 0.72 * p.wheelbase
        v, t = _tube((x - 0.48, 0.55, -0.92), (x + 0.48, 0.55, -0.92), 0.27, max(8, int(round(16 * p.detail))))
        piece("body", "chrome", v, t)
        for dx in (-0.28, 0.28):
            v, t = _box([x + dx, 0.55, -0.92], (0.045, 0.60, 0.60))
            piece("body", "trim", v, t)

    if p.steps:
        cab_front = x_front - (p.bonnet_length if p.cab == "forward" else 0.0)
        for side in (-1.0, 1.0):
            for y in (0.42, 0.66):
                v, t = _box([cab_front - 0.34, y, side * (p.width / 2 - 0.16)], (0.46, 0.055, 0.32))
                piece("body", "trim", v, t)

    for xa in _rear_axles(p):
        for side in (-1.0, 1.0):
            v, t = _tube((xa, p.wheel_radius, -p.track / 2), (xa, p.wheel_radius, p.track / 2), 0.075,
                         max(8, int(round(14 * p.detail))))
            piece("body", "underbody", v, t)
        if p.mudguards:
            for side in (-1.0, 1.0):
                z0, z1 = sorted((side * (p.track / 2 - 0.35), side * (p.track / 2 + 0.36)))
                v, t = _arch(xa, p.wheel_radius, p.wheel_radius + 0.04, z0, z1,
                             max(5, int(round(9 * p.detail))))
                piece("body", "trim", v, t)

    if p.exhaust:
        cab_back = x_front - p.cab_length - (p.bonnet_length if p.cab == "forward" else 0.0)
        x = cab_back + 0.10
        z = (p.width - p.cab_inset) / 2 - 0.10
        v, t = _tube((x, 0.95, z), (x, p.cab_height + 0.20, z), 0.055, max(8, int(round(14 * p.detail))))
        piece("body", "chrome", v, t)


def _rear_axles(p: TruckParams) -> list:
    front = p.length / 2 - p.front_overhang
    return [front - p.wheelbase - k * p.rear_axle_spacing for k in range(p.axles - 1)]


# --- bodies --------------------------------------------------------------------------------------------------


def _body_span(p: TruckParams) -> tuple:
    """The x the body runs between: from behind the cab to the rear end."""
    x_front, x_rear = p.length / 2, -p.length / 2
    cab_back = x_front - p.cab_length - (p.bonnet_length if p.cab == "forward" else 0.0)
    return cab_back - p.body_gap, x_rear


def _body_box(p: TruckParams, parts, piece):
    """A closed cargo box with two rear doors, each hinged at its outer edge."""
    x0, x_rear = _body_span(p)
    x1 = x_rear + 0.05
    half = p.width / 2
    verts, tris = _shell(x0, x1, half, p.body_floor, lambda x: p.body_height, 0.07, 0.08,
                         max(8, int(round(14 * p.detail))))
    piece("body", "body", verts, tris)
    # Rear doors, their hinges, handles and ribs.
    for side, sign in (("l", -1.0), ("r", 1.0)):
        name = f"door_rear_{side}"
        leaf = half - 0.03
        parts[name] = Part(name, "door", pivot=np.array([x1, p.body_floor + 0.04, sign * half]),
                           axis=np.array([0.0, 1.0, 0.0]), open_angle=sign * 105.0)
        centre_z = sign * (leaf / 2 + 0.02)
        x_face = x1 - 0.025
        v, t = _box([x_face, (p.body_floor + p.body_height) / 2, centre_z],
                    (0.05, p.body_height - p.body_floor - 0.08, leaf))
        piece(name, "body", v, t)
        for k in range(4):
            y = p.body_floor + 0.25 + (p.body_height - p.body_floor - 0.55) * k / 3
            v, t = _box([x_face - 0.030, y, centre_z], (0.012, 0.035, leaf - 0.14))
            piece(name, "body", v, t)
        v, t = _box([x_face - 0.045, p.body_floor + 0.55, centre_z - sign * 0.22], (0.035, 0.035, 0.30))
        piece(name, "chrome", v, t)
        # A dark seam around the leaf, so the two rear doors read as doors in flat paint.
        h = p.body_height - p.body_floor
        for z, w in ((sign * 0.015, 0.03), (sign * (half - 0.005), 0.03)):
            v, t = _box([x_face - 0.052, (p.body_floor + p.body_height) / 2, z], (0.012, h - 0.06, w))
            piece(name, "trim", v, t)
        for y in (p.body_floor + 0.055, p.body_height - 0.045):
            v, t = _box([x_face - 0.052, y, centre_z], (0.012, 0.05, leaf - 0.02))
            piece(name, "trim", v, t)
        for y in (p.body_floor + 0.35, (p.body_floor + p.body_height) / 2, p.body_height - 0.35):
            v, t = _box([x_face - 0.060, y, sign * (half - 0.04)], (0.025, 0.12, 0.06))
            piece(name, "chrome", v, t)


def _body_flatbed(p: TruckParams, parts, piece):
    """A wooden deck with low painted sides, a headboard and a tailgate."""
    x0, x_rear = _body_span(p)
    x1 = x_rear + 0.04
    half = p.width / 2
    length = x0 - x1
    v, t = _box([(x0 + x1) / 2, p.deck_height - 0.05, 0.0], (length, 0.10, p.width))
    piece("body", "deck", v, t)
    planks = max(3, int(round(8 * p.detail)))
    for k in range(planks):
        y = p.deck_height + 0.002
        x = x1 + length * (k + 0.5) / planks
        v, t = _box([x, y, 0.0], (0.012, 0.004, p.width - 0.06))
        piece("body", "trim", v, t)
    for side in (-1.0, 1.0):
        v, t = _box([(x0 + x1) / 2, p.deck_height + p.side_height / 2, side * (half - 0.03)],
                    (length, p.side_height, 0.06))
        piece("body", "body", v, t)
        pockets = max(2, int(round(4 * p.detail)))
        for k in range(pockets):
            x = x1 + length * (k + 0.5) / pockets
            v, t = _box([x, p.deck_height + p.side_height + 0.02, side * (half + 0.005)], (0.08, 0.10, 0.05))
            piece("body", "trim", v, t)
    v, t = _box([x0 - 0.04, p.deck_height + 0.45, 0.0], (0.07, 0.90, p.width))
    piece("body", "body", v, t)
    v, t = _box([x1 + 0.04, p.deck_height + p.side_height / 2, 0.0], (0.06, p.side_height, p.width - 0.02))
    piece("body", "body", v, t)


def _body_tanker(p: TruckParams, parts, piece):
    """A horizontal tank with dished ends, its saddles and a walkway along the top."""
    x0, x1 = _body_span(p)
    r = p.tank_radius
    cy = p.tank_center
    segments = max(12, int(round(28 * p.detail)))
    v, t = _tube((x1 + 0.28, cy, 0.0), (x0 - 0.28, cy, 0.0), r, segments)
    piece("body", "tank", v, t)
    for x, sign in ((x1 + 0.28, -1.0), (x0 - 0.28, 1.0)):
        v, t = _tube((x, cy, 0.0), (x + sign * 0.28, cy, 0.0), r * 0.94, segments)
        piece("body", "tank", v, t)
        v, t = _tube((x + sign * 0.20, cy, 0.0), (x + sign * 0.29, cy, 0.0), r * 0.72, segments)
        piece("body", "chrome", v, t)
    for x in (x1 + 0.9, x0 - 1.0):
        v, t = _box([x, (p.chassis_top + cy - r) / 2, 0.0], (0.16, max(0.05, cy - r - p.chassis_top), p.width - 0.5))
        piece("body", "trim", v, t)
    for side in (-1.0, 1.0):
        v, t = _box([(x0 + x1) / 2, cy, side * (r + 0.02)], (x0 - x1 - 0.6, 0.10, 0.05))
        piece("body", "chrome", v, t)
    if p.tank_walkway:
        xw0, xw1 = x1 + 0.55, x0 - 0.55
        v, t = _box([(xw0 + xw1) / 2, cy + r + 0.02, 0.0], (xw0 - xw1, 0.05, 0.62))
        piece("body", "tank", v, t)
        posts = max(2, int(round(3 * p.detail)))
        for k in range(posts):
            x = xw1 + (xw0 - xw1) * (k + 0.5) / posts
            for side in (-1.0, 1.0):
                v, t = _box([x, cy + r + 0.16, side * 0.30], (0.04, 0.30, 0.04))
                piece("body", "chrome", v, t)
        for side in (-1.0, 1.0):
            v, t = _box([(xw0 + xw1) / 2, cy + r + 0.30, side * 0.30], (xw0 - xw1, 0.04, 0.04))
            piece("body", "chrome", v, t)


def _body_dump(p: TruckParams, parts, piece):
    """An open bin on a rear hinge, a part of its own so the pose can tip it."""
    x0, x_rear = _body_span(p)
    x1 = x_rear + 0.04
    half = p.width / 2
    floor = p.body_floor
    top = floor + p.bin_height
    name = "bin"
    parts[name] = Part(name, "bin", pivot=np.array([x1, floor, 0.0]), axis=np.array([0.0, 0.0, 1.0]),
                       open_angle=p.dump_angle)
    v, t = _box([(x0 + x1) / 2, floor, 0.0], (x0 - x1, 0.12, p.width))
    piece(name, "body", v, t)
    for side in (-1.0, 1.0):
        v, t = _box([(x0 + x1) / 2, (floor + top) / 2, side * (half - 0.05)], (x0 - x1, p.bin_height, 0.10))
        piece(name, "body", v, t)
    v, t = _box([x0 - 0.05, (floor + top) / 2, 0.0], (0.10, p.bin_height, p.width))
    piece(name, "body", v, t)
    v, t = _box([x1 + 0.05, floor + p.bin_height * 0.45, 0.0], (0.10, p.bin_height * 0.9, p.width))
    piece(name, "body", v, t)
    # The open inside, dark so the bin reads as a container rather than a closed box.
    v, t = _box([(x0 + x1) / 2, floor + 0.065, 0.0], (x0 - x1 - 0.22, 0.012, p.width - 0.22))
    piece(name, "underbody", v, t)
    for side in (-1.0, 1.0):
        v, t = _box([(x0 + x1) / 2, (floor + top) / 2, side * (half - 0.11)],
                    (x0 - x1 - 0.22, p.bin_height - 0.10, 0.012))
        piece(name, "underbody", v, t)
    v, t = _box([x0 - 0.11, (floor + top) / 2, 0.0], (0.012, p.bin_height - 0.10, p.width - 0.22))
    piece(name, "underbody", v, t)
    v, t = _box([x1 + 0.11, (floor + top) / 2, 0.0], (0.012, p.bin_height - 0.10, p.width - 0.22))
    piece(name, "underbody", v, t)
    # A chrome lip along the top of each side and the back.
    for side in (-1.0, 1.0):
        v, t = _box([(x0 + x1) / 2, top + 0.04, side * (half - 0.05)], (x0 - x1, 0.06, 0.14))
        piece(name, "chrome", v, t)
    v, t = _box([x1 + 0.05, top + 0.04, 0.0], (0.14, 0.06, p.width))
    piece(name, "chrome", v, t)
    # The hinge line at the pivot, so the tipping edge is visible.
    v, t = _tube((x1 + 0.02, floor - 0.02, -half + 0.10), (x1 + 0.02, floor - 0.02, half - 0.10), 0.055,
                 max(8, int(round(12 * p.detail))))
    piece(name, "chrome", v, t)
    v, t = _tube((x1 + 0.55, floor - 0.05, 0.0), (x1 + 1.35, floor + 0.55, 0.0), 0.09,
                 max(8, int(round(12 * p.detail))))
    piece("body", "chrome", v, t)


def _body_fire(p: TruckParams, parts, piece):
    """A fire body: roller-door lockers on both flanks, a roof ladder, a hose reel at the rear and beacons."""
    x0, x_rear = _body_span(p)
    x1 = x_rear + 0.04
    half = p.width / 2
    verts, tris = _shell(x0, x1, half, p.body_floor, lambda x: p.body_height, 0.08, 0.08,
                         max(8, int(round(14 * p.detail))))
    piece("body", "body", verts, tris)
    # Roller doors, split into equal lockers on each flank.
    n = max(1, p.lockers)
    step = (x0 - x1 - 0.36) / n
    for side in (-1.0, 1.0):
        for k in range(n):
            xc = x1 + 0.18 + step * (k + 0.5)
            v, t = _box([xc, 1.80, side * (half + 0.025)], (step - 0.06, 0.95, 0.05))
            piece("body", "locker", v, t)
            slats = max(2, int(round(5 * p.detail)))
            for j in range(slats):
                y = 1.38 + 0.84 * (j + 0.5) / slats
                v, t = _box([xc, y, side * (half + 0.052)], (step - 0.12, 0.012, 0.012))
                piece("body", "trim", v, t)
            v, t = _box([xc + (step - 0.06) / 2 - 0.10, 1.80, side * (half + 0.056)], (0.10, 0.035, 0.028))
            piece("body", "chrome", v, t)
    if p.ladder:
        y = p.body_height + 0.09
        for side in (-1.0, 1.0):
            v, t = _box([(x0 + x1) / 2, y, side * 0.38], (x0 - x1 - 0.5, 0.06, 0.07))
            piece("body", "ladder", v, t)
        rungs = max(4, int(round(14 * p.detail)))
        for k in range(rungs):
            x = x1 + 0.30 + (x0 - x1 - 0.60) * k / (rungs - 1)
            v, t = _box([x, y, 0.0], (0.05, 0.05, 0.76))
            piece("body", "ladder", v, t)
    if p.hose_reel:
        x = x1 + 0.28
        v, t = _tube((x, 1.55, -0.55), (x, 1.55, 0.55), 0.30, max(10, int(round(18 * p.detail))))
        piece("body", "hose", v, t)
        for z in (-0.62, 0.62):
            v, t = _tube((x, 1.55, z - 0.05), (x, 1.55, z + 0.05), 0.37, max(10, int(round(18 * p.detail))))
            piece("body", "chrome", v, t)
        v, t = _box([x + 0.20, 1.55, 0.78], (0.30, 0.05, 0.05))
        piece("body", "chrome", v, t)
    _beacons(p, parts, piece)


def _beacons(p: TruckParams, parts, piece):
    """Two small lamps on the cab roof, one of each beacon colour."""
    if not p.beacons:
        return
    x = p.length / 2 - 0.45
    for z, mat in ((-0.55, "beacon_a"), (0.55, "beacon_b")):
        v, t = _box([x, p.cab_height + 0.035, z], (0.22, 0.05, 0.22))
        piece("body", "black", v, t)
        v, t = _box([x, p.cab_height + 0.10, z], (0.17, 0.10, 0.17))
        piece("body", mat, v, t)


# --- paint areas, lamps and wheels ---------------------------------------------------------------------------


def _livery(p: TruckParams, parts, piece):
    """Paint areas as flat panels on both flanks: (u from the front, height) outlines."""
    half = p.width / 2
    for k, (mat, outline) in enumerate(p.livery):
        name = f"livery{k}" if _is_colour(mat) else str(mat)
        flat = np.array([(p.length / 2 - u * p.length, y) for u, y in outline], float)
        tri = np.array(_triangulate(flat))
        for side in (-1.0, 1.0):
            v = np.column_stack([flat, np.full(len(flat), side * (half + 0.008))])
            t = tri if side > 0 else tri[:, ::-1]
            piece("body", name, v, t)


def _tail_lights(p: TruckParams, parts, piece):
    """Tail lamps and a plate on the rear bumper."""
    x_rear = -p.length / 2
    y = 0.62
    v, t = _box([x_rear + 0.06, y, 0.0], (0.12, 0.22, 0.94 * p.width))
    piece("body", "chrome", v, t)
    for side in (-1.0, 1.0):
        v, t = _box([x_rear + 0.10, y + 0.02, side * 0.78], (0.035, 0.22, 0.26))
        piece("body", "black", v, t)
        v, t = _box([x_rear + 0.12, y + 0.02, side * 0.78], (0.016, 0.16, 0.20))
        piece("body", "tail", v, t)
    v, t = _box([x_rear + 0.12, y, 0.0], (0.012, 0.11, 0.42))
    piece("body", "plate", v, t)


def _wheels(p: TruckParams, parts, piece):
    """Single front wheels and, per rear axle, twin wheels at the outer and the inner track."""
    x_front = p.length / 2
    front_axle = x_front - p.front_overhang
    axles = [(front_axle, True, 0)] + [(x, False, k + 1) for k, x in enumerate(_rear_axles(p))]
    for xa, front, index in axles:
        for side, sign in (("l", -1.0), ("r", 1.0)):
            if front or not p.twin_rear:
                tracks = [(p.track, "")]
            else:
                tracks = [(p.track, ""), (p.track - 1.15 * p.tyre_width, "i")]
            for track, suffix in tracks:
                name = (f"wheel_f{side}" if front else f"wheel_r{index}{suffix}{side}")
                meshes, hub = _wheel(replace(p, track=track), xa, sign)
                parts[name] = Part(name, "wheel", pivot=hub, axis=np.array([0.0, 0.0, 1.0]), steer=front)
                for mat, pieces in meshes.items():
                    for v, t in pieces:
                        piece(name, mat, v, t)


# --- the truck -----------------------------------------------------------------------------------------------


def build_truck(p: TruckParams | None = None) -> Truck:
    """A truck from its measurements: a cab, a chassis and one of five bodies. The default is a cab-over box
    truck, about 7.8 m long and 3.5 m tall."""
    p = p or TruckParams()
    p.validate()
    parts: dict[str, Part] = {"body": Part("body", "body")}

    def piece(part_name, mat, v, t, n=None):
        """Add triangles to a part's mesh of one material."""
        part = parts[part_name]
        v, t = np.asarray(v, float), np.asarray(t)
        n = _mesh(v, t).normals if n is None else np.asarray(n, float)
        if mat in part.pieces:
            old = part.pieces[mat]
            t = np.concatenate([old.faces, t + len(old.positions)])
            v = np.concatenate([old.positions, v])
            n = np.concatenate([old.normals, n])
        part.pieces[mat] = Mesh(positions=v.astype(np.float32), faces=t.astype(np.uint32),
                                normals=n.astype(np.float32))

    _build_cab(p, parts, piece)
    _chassis(p, parts, piece)
    if p.body == "box":
        _body_box(p, parts, piece)
    elif p.body == "flatbed":
        _body_flatbed(p, parts, piece)
    elif p.body == "tanker":
        _body_tanker(p, parts, piece)
    elif p.body == "dump":
        _body_dump(p, parts, piece)
    else:
        _body_fire(p, parts, piece)
    if p.body != "fire":
        _beacons(p, parts, piece)
    _livery(p, parts, piece)
    _tail_lights(p, parts, piece)
    _wheels(p, parts, piece)
    return Truck(p, [part for part in parts.values() if part.pieces])


# --- presets -------------------------------------------------------------------------------------------------


def cargo_truck_1980s() -> TruckParams:
    """A bonneted 1980s cargo truck: faded blue cab, a wooden flatbed with low sides, two axles."""
    return TruckParams(
        length=7.3, width=2.4, height=2.8, axles=2,
        wheelbase=3.9, front_overhang=1.2, track=1.98, wheel_radius=0.52, tyre_width=0.28,
        cab="forward", cab_length=1.55, cab_height=2.6, bonnet_length=1.35, bonnet_height=1.35,
        nose_height=1.10, exhaust=True, body="flatbed", deck_height=1.15, side_height=0.42,
        cab_paint="#6f9fb8", body_paint="#6f9fb8",
    )


def box_truck() -> TruckParams:
    """A cab-over box truck: a white cab and a grey cargo box with two rear doors, two axles."""
    return TruckParams(
        length=7.8, width=2.45, height=3.5, axles=2,
        wheelbase=4.35, front_overhang=1.25, track=2.02, wheel_radius=0.52, tyre_width=0.28,
        cab="over", cab_length=1.85, cab_height=2.65, body="box", body_height=3.5,
        cab_paint="#f1f2f3", body_paint="#b9bdc2",
    )


def tanker() -> TruckParams:
    """A bonneted tanker: a yellow cab, a silver tank with a walkway, three axles."""
    return TruckParams(
        length=9.6, width=2.45, height=3.2, axles=3,
        wheelbase=4.5, rear_axle_spacing=1.35, front_overhang=1.4, track=2.02,
        wheel_radius=0.52, tyre_width=0.28,
        cab="forward", cab_length=1.6, cab_height=2.65, bonnet_length=1.4, bonnet_height=1.35,
        nose_height=1.10, exhaust=True, body="tanker", tank_radius=1.0, tank_center=2.02,
        cab_paint="#e7c33b", body_paint="#c9ccd0",
    )


def dump_truck() -> TruckParams:
    """A cab-over tipper with an orange open bin and three axles."""
    return TruckParams(
        length=8.2, width=2.5, height=2.95, axles=3,
        wheelbase=4.2, rear_axle_spacing=1.35, front_overhang=1.3, track=2.05,
        wheel_radius=0.54, tyre_width=0.30,
        cab="over", cab_length=1.9, cab_height=2.75, exhaust=True,
        body="dump", bin_height=1.35, body_floor=1.20, cab_paint="#d96a1f", body_paint="#d96a1f",
    )


def fire_engine() -> TruckParams:
    """A cab-over fire engine: red paint with a white stripe, lockers, a roof ladder, a hose reel and beacons."""
    return TruckParams(
        length=8.6, width=2.5, height=3.25, axles=2,
        wheelbase=4.6, front_overhang=1.5, track=2.05, wheel_radius=0.54, tyre_width=0.30,
        cab="over", cab_length=2.0, cab_height=2.75, body="fire", body_height=3.15,
        lockers=3, ladder=True, hose_reel=True, beacons=True,
        cab_paint="#c62828", body_paint="#c62828",
        livery=(("#f4f4f2", [(0.25, 2.28), (1.0, 2.28), (1.0, 2.56), (0.25, 2.56)]),),
    )
