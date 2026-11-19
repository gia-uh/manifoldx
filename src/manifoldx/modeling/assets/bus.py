"""A parametric city bus: named measurements in, an articulated low-floor bus out.

A bus is not a stretched car: its body is a box with the vertical corners rounded in plan and the roof edges
softened, its front and rear faces are nearly upright, and its sides carry one long band of glass interrupted by
pillars and by double doors. The body is a loft of the same section along the length, its half-width following the
plan's rounded corners, so the ends close on a flat face without a boolean. Wheel arches are cut the way the car
cuts them: every point inside an arch's circle is raised onto it.

Each door's leaves are parts of their own, hinged on a vertical axis at their outer edges, so they swing out
of the flank. Wheels are the car's wheels (``car._wheel``), turned and steered by :meth:`Bus.pose`; lamps, plates and paint
areas are outlines projected onto the skin with the car's decal machinery.

Paint areas (`livery`, `front_livery`, `rear_livery`, `roof_livery`) are polygons in the side, front, rear or plan
view, named by a palette material or an sRGB hex colour, and projected onto the skin they name.

Axes are the shared ones: metres, Y up, the front at +X, the right side at +Z, the centre on the ground at the
origin.
"""

from __future__ import annotations

from dataclasses import dataclass
from string import ascii_lowercase

import numpy as np

from manifoldx.modeling.assets.car import (
    _axis_angle,
    _box,
    _corner_normals,
    _drape,
    _is_colour,
    _mesh,
    _quat_mul,
    _wheel,
)
from manifoldx.modeling.assets.vehicle import PALETTE, ParamsRanges, Part, Vehicle, _rotate, choice, rng
from manifoldx.modeling.mesh import Mesh


@dataclass
class BusParams(ParamsRanges):
    """Every measurement of the bus. Heights are metres from the ground; `_u` values are fractions of the
    length from the front (0) to the rear (1)."""

    # Package: the spec sheet.
    length: float = rng(12.0, 6.0, 18.0, step=0.01, unit='m')
    width: float = rng(2.55, 2.0, 3.0, step=0.01, unit='m')
    height: float = rng(3.1, 2.5, 4.5, step=0.01, unit='m')
    corner_radius: float = rng(0.25, 0.0, 0.8, step=0.01, unit='m')  # of the vertical corners, in plan
    floor: float = rng(0.35, 0.1, 1.2, step=0.01, unit='m')  # the low floor, from the ground
    front_overhang: float = rng(2.6, 0.5, 5.5, step=0.01, unit='m')
    wheelbase: float = rng(6.0, 2.0, 9.0, step=0.01, unit='m')
    wheel_radius: float = rng(0.5, 0.25, 0.8, step=0.005, unit='m')
    tyre_width: float = rng(0.3, 0.15, 0.45, step=0.005, unit='m')
    track: float = rng(2.1, 1.2, 2.8, step=0.01, unit='m')
    rim_ratio: float = rng(0.64, 0.35, 0.95, step=0.01, unit='')
    arch_gap: float = rng(0.05, 0.0, 0.2, step=0.005, unit='m')
    # Front face.
    windscreen_bottom: float = rng(0.9, 0.4, 2.7, step=0.01, unit='m')
    windscreen_top: float = rng(2.75, 1.0, 3.2, step=0.01, unit='m')
    windscreen_rake: float = rng(0.08, 0.0, 0.5, step=0.005, unit='m')  # how far the screen's top sits back
    sign_bottom: float = rng(2.75, 2.0, 2.95, step=0.01, unit='m')  # the route sign band, over the screen
    sign_top: float = rng(3.0, 2.8, 3.8, step=0.01, unit='m')
    headlamp: str = choice("twin", 'twin', 'rect', 'round')  # "twin", "rect" or "round"
    headlamp_center: tuple | None = None  # (lateral, height); None uses the default position
    headlamp_size: tuple = (0.48, 0.22)
    # Rear face.
    rear_window_bottom: float = rng(1.25, 0.5, 2.5, step=0.01, unit='m')
    rear_window_top: float = rng(2.55, 1.3, 3.2, step=0.01, unit='m')
    taillamp: str = choice("bar", 'bar', 'round')  # "bar" or "round"
    taillamp_center: tuple | None = None  # (lateral, height); None uses the default position
    taillamp_size: tuple = (0.42, 0.18)
    # Sides.
    window_bottom: float = rng(1.25, 0.5, 2.55, step=0.01, unit='m')
    window_top: float = rng(2.6, 1.3, 3.2, step=0.01, unit='m')
    pillar_u: tuple | None = None  # where the pillars stand; None spaces them `pillar_spacing` apart
    pillar_spacing: float = rng(1.3, 0.5, 3.0, step=0.01, unit='m')
    pillar_width: float = rng(0.10, 0.02, 0.3, step=0.005, unit='m')
    doors: tuple = ((0.04, 1.2, 2), (0.47, 1.2, 2), (0.86, 1.2, 2))  # (u of the front edge, width, leaves)
    door_bottom: float = rng(0.30, 0.0, 1.0, step=0.01, unit='m')
    # A coach: a high floor over luggage bays along the lower flank, small tinted windows and one front door.
    body_style: str = choice("city", 'city', 'coach')  # "city" or "coach"
    luggage_bays: int = rng(0, 0, 8, step=1, unit='')  # bay doors on each flank
    bay_bottom: float = rng(0.35, 0.1, 1.2, step=0.01, unit='m')
    bay_top: float = rng(0.85, 0.3, 2.0, step=0.01, unit='m')
    # Paint areas: (material or hex colour, polygon) in the side, front, rear and plan views.
    livery: tuple = ()
    front_livery: tuple = ()
    rear_livery: tuple = ()
    roof_livery: tuple = ()
    # Resolution and small parts.
    sections: int = rng(24, 4, 80, step=1, unit='')
    detail: float = rng(1.0, 0.25, 2.0, step=0.05, unit='x')  # scales the stations, the wheel's segments and the decals
    crease: float = rng(38.0, 0.0, 180.0, step=1.0, unit='deg')  # degrees: an edge that turns more than this is shaded sharp
    wheel: str | dict = choice("spokes", 'spokes', 'hubcap')
    spokes: int = rng(8, 0, 32, step=1, unit='')
    whitewall: float = rng(0.0, 0.0, 0.2, step=0.005, unit='m')
    mirrors: bool = True

    def _validate_extra(self) -> None:
        if self.wheelbase + self.front_overhang >= self.length:
            raise ValueError(
                f"BusParams.wheelbase + front_overhang ({self.wheelbase + self.front_overhang:.2f}) "
                f"must be below length ({self.length:.2f})"
            )
        if self.wheelbase <= 2 * self.wheel_radius:
            raise ValueError(
                f"BusParams.wheelbase ({self.wheelbase}) overlaps two wheel_radius ({self.wheel_radius}) wheels"
            )
        spans = []
        for door in self.doors:
            try:
                u, width, _leaves = door
            except (TypeError, ValueError) as exc:
                raise ValueError(f"BusParams.doors: {door!r} is not (u, width, leaves)") from exc
            start, end = u * self.length, u * self.length + width
            if width <= 0 or u < 0 or end > self.length:
                raise ValueError(f"BusParams.doors: door at u={u} with width {width} does not fit in length {self.length}")
            spans.append((start, end))
        spans.sort()
        for (a0, a1), (b0, b1) in zip(spans, spans[1:]):
            if b0 < a1:
                raise ValueError(f"BusParams.doors: doors at {a0}..{a1} and {b0}..{b1} overlap")
        for lo, hi, name in (
            (self.windscreen_bottom, self.windscreen_top, "windscreen"),
            (self.rear_window_bottom, self.rear_window_top, "rear_window"),
            (self.window_bottom, self.window_top, "window"),
            (self.sign_bottom, self.sign_top, "sign"),
            (self.bay_bottom, self.bay_top, "bay"),
        ):
            if not lo < hi:
                raise ValueError(
                    f"BusParams.{name}_bottom ({lo}) must sit below BusParams.{name}_top ({hi})"
                )
        if self.luggage_bays > 0 and self.bay_top >= self.floor:
            raise ValueError(
                f"BusParams.bay_top ({self.bay_top}) must sit below floor ({self.floor}) for a luggage bay"
            )


@dataclass
class Bus(Vehicle):
    """A built bus: its measurements and its rigid parts. Vehicle is its shared shape, so a client treating every
    vehicle alike (posing it, merging it for a crowd) can take a Bus without knowing the bus's own options."""

    params: BusParams

    @property
    def palette(self) -> dict:
        """The shared palette, plus a material for each hex paint area and the route sign's amber."""
        extra = {"sign": ("#f0b23c", 0.45, 0.0), "tinted": ("#101619", 0.04, 0.0), "bay": ("#9aa1a6", 0.45, 0.3)}
        for k, (mat, _) in enumerate(_livery_entries(self.params)):
            if _is_colour(mat):
                extra[f"livery{k}"] = (mat, 0.35, 0.0)
        return {**PALETTE, **extra}

    @property
    def landmarks(self) -> dict:
        """Named three-dimensional points on the bus, for registering a drawing by keypoints."""
        p = self.params
        axles = (p.length / 2 - p.front_overhang, p.length / 2 - p.front_overhang - p.wheelbase)
        out = {}
        for x, end in zip(axles, ("front", "rear")):
            for side, sign in (("left", -1.0), ("right", 1.0)):
                out[f"wheel_{end}_{side}"] = np.array([x, p.wheel_radius, sign * p.track / 2])
        out["nose"] = np.array([p.length / 2, p.floor, 0.0])
        out["tail"] = np.array([-p.length / 2, p.floor, 0.0])
        out["roof"] = np.array([0.0, p.height, 0.0])
        out["windscreen_foot"] = np.array([p.length / 2, p.windscreen_bottom, 0.0])
        out["windscreen_top"] = np.array([p.length / 2 - p.windscreen_rake, p.windscreen_top, 0.0])
        return out

    def pose(self, doors=0.0, wheel_turn=0.0, steer=0.0, travel=None) -> dict:
        """Each part's (position, quaternion xyzw) so that bus-space vertices land in their posed place.

        doors: 0 shut to 1 fully open (or a dict by leaf name). wheel_turn: radians the wheels have rolled.
        steer: radians the front wheels turn, positive to the left.
        """
        out = {}
        lift = iter(travel if travel is not None else ())  # metres each wheel rises, in parts order
        for p in self.parts:
            if p.kind == "door":
                angle = doors.get(p.name, 0.0) if isinstance(doors, dict) else doors
                q = _axis_angle(p.axis, np.radians(p.open_angle) * angle)
            elif p.kind == "wheel":
                q = _axis_angle(np.array([0.0, 0.0, 1.0]), -wheel_turn)
                if p.steer:
                    q = _quat_mul(_axis_angle(np.array([0.0, 1.0, 0.0]), steer), q)
            else:
                q = np.array([0.0, 0.0, 0.0, 1.0])
            offset = p.pivot - _rotate(q, p.pivot)
            if p.kind == "wheel" and travel is not None:
                offset = offset + np.array([0.0, float(next(lift, 0.0)), 0.0])
            out[p.name] = (offset, q)
        return out


# --- measurements the body is drawn from ---------------------------------------------------------------------


def _livery_entries(p: BusParams):
    """Every paint area, in the order the palette gives each hex colour its name."""
    return [entry for entries in (p.livery, p.front_livery, p.rear_livery, p.roof_livery) for entry in entries]


def _underside(p: BusParams) -> float:
    """The height of the body's floor pan, just under the low floor."""
    return max(0.18, p.floor - 0.09)


def _axles(p: BusParams) -> tuple:
    """The two axles' positions along the bus, front first."""
    front = p.length / 2 - p.front_overhang
    return front, front - p.wheelbase


def _pillars(p: BusParams) -> tuple:
    """Where the pillars stand, as length fractions: the given ones, or an even spacing."""
    if p.pillar_u is not None:
        return tuple(p.pillar_u)
    span = max(p.length - 2 * p.corner_radius, 1e-6)
    count = max(1, int(round(span / max(p.pillar_spacing, 0.3))))
    return tuple((k + 0.5) / count for k in range(count))


def _halfwidth(p: BusParams, x: float) -> float:
    """Half the body's width at `x`: full along the flanks, an arc round each vertical corner in plan."""
    half, r = p.width / 2, p.corner_radius
    rear, front = -p.length / 2 + r, p.length / 2 - r
    if x < rear:
        d = x - rear
    elif x > front:
        d = x - front
    else:
        return half
    return half - r + np.sqrt(max(0.0, r * r - d * d))


def _front_x(p: BusParams, y: float) -> float:
    """The front face's x at height `y`: upright below the screen, `windscreen_rake` back at the roof."""
    if not p.windscreen_rake:
        return p.length / 2
    t = np.clip((y - p.windscreen_bottom) / max(p.height - p.windscreen_bottom, 1e-6), 0.0, 1.0)
    return p.length / 2 - p.windscreen_rake * t * t


def _section(p: BusParams, halfw: float) -> list:
    """Half a cross-section, from the floor's centre round the side to the roof's centre. The points at the
    window's edges and at the windscreen's foot and top are kept, so a face region ends on a section line."""
    y0, top = _underside(p), p.height
    edge = min(0.16, max(0.06, 0.03 * top))
    roof_edge = top - edge
    floors = sorted({round(v, 6) for v in (y0 + 0.30, p.windscreen_bottom, p.window_bottom - 0.09,
                                           p.window_bottom, p.window_top, p.window_top + 0.10,
                                           p.windscreen_top, p.sign_top)
                     if y0 + 0.12 < v < roof_edge})
    pts = [(0.0, y0), (0.62 * halfw, y0), (0.90 * halfw, y0 + 0.10)]
    pts += [(halfw, y) for y in floors]
    pts += [(0.985 * halfw, roof_edge), (0.90 * halfw, top - 0.02), (0.55 * halfw, top), (0.0, top)]
    return pts


def _stations(p: BusParams) -> np.ndarray:
    """Section positions along the bus: even stations, the corners' arcs in plan, and one at every region edge."""
    L, r = p.length, p.corner_radius
    x0, x1 = -L / 2 + r, L / 2 - r
    n = max(8, int(round(p.sections * p.detail)))
    xs = list(np.linspace(x0, x1, n))
    for u0, width, _ in p.doors:
        for u in (u0, u0 + width / L):
            x = L / 2 - u * L
            if x0 < x < x1:
                xs.append(x)
    for u in _pillars(p):
        for uu in (u - p.pillar_width / 2 / L, u + p.pillar_width / 2 / L):
            x = L / 2 - uu * L
            if x0 < x < x1:
                xs.append(x)
    R = p.wheel_radius + p.arch_gap
    for xa in _axles(p):
        for t in np.linspace(-np.pi / 2, np.pi / 2, max(5, int(round(9 * p.detail)))):
            x = xa + R * np.sin(t)
            if x0 < x < x1:
                xs.append(x)
    corner = max(3, int(round(4 * p.detail)))
    arc = np.linspace(0.0, np.pi / 2, corner)[1:]
    xs += [x1 + r * np.sin(t) for t in arc] + [x0 - r * np.sin(t) for t in arc]
    return np.unique(np.round(xs, 6))


def _grid(p: BusParams):
    """The loft's vertices: one ring per station, with the front's rake and the wheel arches raised into it."""
    xs = _stations(p)
    S = len(xs)
    rings = []
    for x in xs:
        h = _section(p, _halfwidth(p, x))
        rings.append(h + [(-z, y) for z, y in reversed(h[1:-1])])
    RING = len(rings[0])
    P = np.zeros((S, RING, 3))
    for i, ring in enumerate(rings):
        P[i, :, 2] = [z for z, _ in ring]
        P[i, :, 1] = [y for _, y in ring]
        P[i, :, 0] = xs[i]
    if p.windscreen_rake:  # near the front, every point sits back by the rake as it climbs
        climb = np.clip((P[..., 1] - p.windscreen_bottom) / max(p.height - p.windscreen_bottom, 1e-6),
                        0.0, 1.0) ** 2
        along = np.clip((P[..., 0] - (p.length / 2 - 0.5)) / 0.5, 0.0, 1.0)
        P[..., 0] -= p.windscreen_rake * climb * along
    R = p.wheel_radius + p.arch_gap
    raised = np.zeros(P.shape[:2], bool)
    for xa in _axles(p):
        dx = P[..., 0] - xa
        inside = np.abs(dx) < R
        arch = p.wheel_radius + np.sqrt(np.clip(R * R - dx * dx, 0.0, None))
        m = inside & (P[..., 1] < arch)
        P[..., 1] = np.where(m, arch, P[..., 1])
        raised |= m
    return xs, P, raised


# --- materials of the skin ------------------------------------------------------------------------------------


def _door_at(p: BusParams, u: float, side: int) -> bool:
    """Whether a door leaf stands in front of the flank at `u` (the doors are on the right side)."""
    if side <= 0:
        return False
    return any(u0 - 0.012 <= u <= u0 + width / p.length + 0.012 for u0, width, _ in p.doors)


def _pillar_at(p: BusParams, u: float) -> bool:
    """Whether a pillar or an end's black corner interrupts the window band at `u`."""
    if u < 0.025 or u > 0.985:
        return True
    return any(abs(u - c) <= p.pillar_width / 2 / p.length for c in _pillars(p))


def _side_material(p: BusParams, u: float, y: float, side: int, raised: bool) -> str:
    """The material of a loft face from where it sits: the band of glass, its black frames, or paint."""
    if raised or y <= _underside(p) + 0.12:
        return "underbody"
    if p.window_bottom - 0.09 <= y <= p.window_bottom:
        return "black"
    if p.window_bottom < y < p.window_top:
        if _door_at(p, u, side):
            return "paint"  # behind a shut leaf: the cabin's light wall shows through its glass
        return "black" if _pillar_at(p, u) else ("tinted" if p.body_style == "coach" else "glass")
    if p.window_top <= y <= p.window_top + 0.10:
        return "black"
    return "paint"


def _end_material(p: BusParams, sign: float, y: float) -> str:
    """The material of a horizontal band of an end: screen, rear window, the route sign band, or paint."""
    if sign > 0:
        if p.body_style != "coach" and p.sign_bottom <= y <= p.sign_top:
            return "black"
        if p.windscreen_bottom <= y <= p.windscreen_top:
            return "glass"
        if p.windscreen_bottom - 0.08 <= y < p.windscreen_bottom:
            return "black"
    else:
        if p.rear_window_bottom <= y <= p.rear_window_top:
            return "glass"
        if p.rear_window_bottom - 0.06 <= y < p.rear_window_bottom:
            return "black"
        if p.rear_window_top < y <= p.rear_window_top + 0.30:
            return "black"  # the dark panel over the rear window
    return "paint"


def _loft_faces(p: BusParams, xs, P, raised):
    """Triangles of the loft, with a material each; the section is tagged by the height of each face."""
    S, RING = P.shape[:2]
    HALF = (RING + 2) // 2
    verts = P.reshape(-1, 3)
    tris, mats = [], []

    def vid(i, j):
        return i * RING + j % RING

    for i in range(S - 1):
        for j in range(RING):
            a, c, d, e = vid(i, j), vid(i, j + 1), vid(i + 1, j + 1), vid(i + 1, j)
            far = j >= HALF - 1
            quad = [(a, c, e), (c, d, e)] if far else [(a, c, d), (a, d, e)]
            inside = raised[i, j] and raised[i, (j + 1) % RING] and raised[i + 1, j] and raised[i + 1, (j + 1) % RING]
            for t in quad:
                pts = verts[list(t)]
                u = (p.length / 2 - pts[:, 0].mean()) / p.length
                side = 1 if pts[:, 2].mean() > 0 else -1
                mats.append(_side_material(p, u, pts[:, 1].mean(), side, inside))
                tris.append(t)
    return verts, np.array(tris), np.array(mats)


def _cap_faces(p: BusParams, P):
    """The flat face at each end, as one horizontal band per section level, so each region ends on a level."""
    verts, tris, mats = [], [], []
    base = 0
    half = (P.shape[1] + 2) // 2
    for i, sign in ((0, -1.0), (len(P) - 1, 1.0)):
        ring = P[i, :half]
        levels = {}
        for z, y in ring[:, [2, 1]]:
            levels[round(float(y), 6)] = max(levels.get(round(float(y), 6), 0.0), abs(float(z)))
        ys = sorted(levels)
        for y0, y1 in zip(ys[:-1], ys[1:]):
            z0, z1 = levels[y0], levels[y1]
            x0 = _front_x(p, y0) if sign > 0 else -p.length / 2
            x1 = _front_x(p, y1) if sign > 0 else -p.length / 2
            quad = np.array([[x0, y0, -z0], [x0, y0, z0], [x1, y1, z1], [x1, y1, -z1]])
            # In (z, y) this winds counter-clockwise, which faces -X: right at the rear, reversed at the front.
            t = np.array([[0, 1, 2], [0, 2, 3]])
            if sign > 0:
                t = t[:, ::-1]
            verts.append(quad)
            tris.append(t + base)
            mats += [_end_material(p, sign, (y0 + y1) / 2)] * 2
            base += len(quad)
    return np.concatenate(verts), np.concatenate(tris), np.array(mats)


# --- decals and paint areas ----------------------------------------------------------------------------------


def _circle(cz: float, cy: float, r: float, n: int = 20) -> list:
    """A circle in a face's (lateral, height) coordinates."""
    return [(cz + r * np.cos(t), cy + r * np.sin(t)) for t in np.linspace(0, 2 * np.pi, max(6, n), endpoint=False)]


def _rect(z0, y0, z1, y1) -> list:
    """A rectangle in a face's (lateral, height) coordinates."""
    return [(z0, y0), (z1, y0), (z1, y1), (z0, y1)]


def _decals(p: BusParams, verts, tris):
    """Paint areas and small parts projected onto the body: (material, vertices, triangles) pieces."""
    W, L, H = p.width, p.length, p.height
    out = []

    def lay(mat, outline, origin, across, up, toward, lift, step=0.07):
        laid = _drape(outline, 1.0, np.asarray(origin, float), np.asarray(across, float), np.asarray(up, float),
                      np.asarray(toward, float), verts, tris, lift, step=step)
        if laid:
            out.append((mat, laid[1], laid[2]))

    def side(mat, outline, lift=0.004, step=0.30):
        for sign in (-1.0, 1.0):
            lay(mat, outline, (0.0, 0.0, sign * (W / 2 + 1)), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0),
                (0.0, 0.0, -sign), lift, step)

    def front(mat, outline, lift=0.003, step=0.06):
        lay(mat, outline, (L / 2 + 1, 0.0, 0.0), (0.0, 0.0, 1.0), (0.0, 1.0, 0.0), (-1.0, 0.0, 0.0), lift, step)

    def rear(mat, outline, lift=0.003, step=0.06):
        lay(mat, outline, (-L / 2 - 1, 0.0, 0.0), (0.0, 0.0, 1.0), (0.0, 1.0, 0.0), (1.0, 0.0, 0.0), lift, step)

    def plan(mat, outline, lift=0.003, step=0.12):
        lay(mat, outline, (0.0, H + 1, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, 1.0), (0.0, -1.0, 0.0), lift, step)

    names = {mat: f"livery{k}" for k, (mat, _) in enumerate(_livery_entries(p)) if _is_colour(mat)}

    def along(outline):  # from (u, v) to the x the view's outline is drawn in
        return [(p.length / 2 - u * p.length, v) for u, v in outline]

    for entries, paint, convert in ((p.livery, side, along), (p.front_livery, front, None),
                                    (p.rear_livery, rear, None), (p.roof_livery, plan, along)):
        for mat, outline in entries:
            paint(names.get(mat, mat), convert(outline) if convert else outline)

    # The windscreen's frame: a bar at each side and one down the middle, standing over the glass.
    for z0, z1 in ((-1.03, -0.88), (0.88, 1.03), (-0.035, 0.035)):
        front("black", _rect(z0, p.windscreen_bottom, z1, p.windscreen_top), 0.004, step=0.10)

    # The route sign's amber marks, over the black band on the front (a coach has none).
    y0, y1 = p.sign_bottom + 0.05, p.sign_top - 0.06
    if p.body_style != "coach":
        for z0, z1 in ((-0.42, -0.28), (-0.22, -0.06), (0.04, 0.16), (0.24, 0.40)):
            front("sign", _rect(z0, y0, z1, y1), 0.010)

    # Lamps: a dark surround, two lit lenses and an amber marker on each side.
    cy, cz = p.headlamp_center or (0.72, 0.68)
    a, bh = p.headlamp_size[0] / 2, p.headlamp_size[1] / 2
    n = max(8, int(round(18 * p.detail)))
    for sign in (-1.0, 1.0):
        z = sign * cy
        if p.headlamp == "round":
            front("black", _circle(z, cz, bh * 1.25, n), 0.006)
            front("lamp", _circle(z, cz, bh, n), 0.009)
        elif p.headlamp == "rect":
            front("black", _rect(z - a - 0.02, cz - bh - 0.02, z + a + 0.02, cz + bh + 0.02), 0.006)
            front("lamp", _rect(z - a, cz - bh, z + a, cz + bh), 0.009)
        else:  # twin: two round lenses in one dark housing
            front("black", _rect(z - a - 0.02, cz - bh - 0.02, z + a + 0.02, cz + bh + 0.02), 0.006)
            for k in (-1.0, 1.0):
                front("lamp", _circle(z + k * 0.36 * a, cz, 0.78 * bh, n), 0.009)
        front("sign", _rect(z + 0.55 * a, cz + 0.45 * bh, z + 0.85 * a, cz + 0.75 * bh), 0.010)

    # The rear window's frame: a black pillar at each side, so the window does not run into the corners.
    for z0, z1 in ((-1.03, -0.80), (0.80, 1.03)):
        rear("black", _rect(z0, p.rear_window_bottom, z1, p.rear_window_top), 0.004, step=0.10)

    # Tail lamps: a red bar or disc on each side of the rear, in a black housing.
    cy, cz = p.taillamp_center or (0.76, 0.70)
    a, bh = p.taillamp_size[0] / 2, p.taillamp_size[1] / 2
    for sign in (-1.0, 1.0):
        z = sign * cy
        if p.taillamp == "round":
            rear("black", _circle(z, cz, bh * 1.25, n), 0.006)
            rear("tail", _circle(z, cz, bh, n), 0.009)
        else:
            rear("black", _rect(z - a - 0.02, cz - bh - 0.02, z + a + 0.02, cz + bh + 0.02), 0.006)
            rear("tail", _rect(z - a, cz - bh, z + a, cz + bh), 0.009)

    # Number plates.
    front("plate", _rect(-0.26, 0.42, 0.26, 0.53), 0.008)
    rear("plate", _rect(-0.26, 0.52, 0.26, 0.63), 0.008)
    return out


# --- the coach's luggage bays ---------------------------------------------------------------------------------


def _coach_skirt(p: BusParams, parts, piece) -> None:
    """A coach's lower flank: a painted skirt between the wheel arches, with its luggage-bay doors."""
    if p.body_style != "coach" or p.luggage_bays <= 0:
        return
    half = p.width / 2
    bottom = p.bay_bottom
    top = max(_underside(p) + 0.02, p.bay_top + 0.08)
    reach = p.wheel_radius + 0.32
    spans, cursor = [], -p.length / 2 + 0.30
    for xa in sorted(_axles(p)):
        spans.append((cursor, xa - reach))
        cursor = xa + reach
    spans.append((cursor, p.length / 2 - 0.45))
    spans = [(a, b) for a, b in spans if b - a > 0.35]
    total = sum(b - a for a, b in spans)
    counts = [max(1, round(p.luggage_bays * (b - a) / total)) for a, b in spans]
    while sum(counts) > p.luggage_bays:
        counts[counts.index(max(counts))] -= 1
    while sum(counts) < p.luggage_bays:
        counts[counts.index(min(counts))] += 1
    for (a, b), n in zip(spans, counts):
        v, t = _box([(a + b) / 2, (bottom + top) / 2, half - 0.02], (b - a, top - bottom, 0.04))
        piece("body", "paint", v, t)
        v, t = _box([(a + b) / 2, (bottom + top) / 2, -half + 0.02], (b - a, top - bottom, 0.04))
        piece("body", "paint", v, t)
        step = (b - a) / n
        for k in range(n):
            cx = a + step * (k + 0.5)
            leaf = step - 0.14
            h = p.bay_top - bottom
            mid = bottom + h / 2
            for sign in (-1.0, 1.0):
                v, t = _box([cx, mid, sign * (half - 0.004)], (leaf + 0.04, h + 0.04, 0.018))
                piece("body", "black", v, t)
                v, t = _box([cx, mid, sign * (half + 0.006)], (leaf, h, 0.020))
                piece("body", "bay", v, t)
                v, t = _box([cx + leaf / 2 - 0.14, mid + h * 0.18, sign * (half + 0.018)],
                            (0.16, 0.035, 0.022))
                piece("body", "chrome", v, t)


# --- parts ---------------------------------------------------------------------------------------------------


def _doors(p: BusParams, parts, piece):
    """Each door leaf as a part of its own: a dark lower panel in a frame round its glass, hinged at its edge."""
    half, top = p.width / 2, p.window_top
    z0, thick = half + 0.006, 0.04
    for d, (u0, width, leaves) in enumerate(p.doors):
        x_front = p.length / 2 - u0 * p.length
        leaf = width / leaves
        for k in range(leaves):
            col_front = x_front - k * leaf
            col_rear = col_front - leaf
            front_hinged = k % 2 == 0
            hinge = col_front if front_hinged else col_rear
            other = hinge - (leaf - 0.012) if front_hinged else hinge + (leaf - 0.012)
            x0, x1 = min(hinge, other), max(hinge, other)
            name = f"door_{d + 1}{ascii_lowercase[k % len(ascii_lowercase)]}"
            parts[name] = Part(name, "door", pivot=np.array([hinge, p.floor, half]),
                               axis=np.array([0.0, 1.0, 0.0]), open_angle=72.0 if front_hinged else -72.0)
            centre = (x0 + x1) / 2
            sill = max(p.door_bottom + 0.45, p.window_bottom - 0.35)  # a bus door's glass reaches lower than a window
            # The lower panel is solid dark; above it a frame holds a single glass pane, so the glass shows the
            # cabin's light wall behind it rather than a black panel.
            v, t = _box([centre, (p.door_bottom + sill) / 2, z0 + thick / 2],
                        (x1 - x0, sill - p.door_bottom, thick))
            piece(name, "black", v, t)
            bar = min(0.07, 0.18 * (x1 - x0))
            mid = (sill + top) / 2
            for cx, cy, sx, sy in ((centre, top - bar / 2, x1 - x0, bar),
                                   (x0 + bar / 2, mid, bar, top - sill),
                                   (x1 - bar / 2, mid, bar, top - sill)):
                v, t = _box([cx, cy, z0 + thick / 2], (sx, sy, thick))
                piece(name, "black", v, t)
            gx0, gx1 = x0 + bar, x1 - bar
            gy0, gy1 = sill + 0.02, top - bar - 0.01
            gv = np.array([[gx0, gy0, z0 + thick / 2], [gx1, gy0, z0 + thick / 2],
                           [gx1, gy1, z0 + thick / 2], [gx0, gy1, z0 + thick / 2]])
            piece(name, "glass", gv, np.array([[0, 1, 2], [0, 2, 3]]))


def _mirrors(p: BusParams, parts, piece):
    """A tall black mirror on a stalk at each front corner."""
    if not p.mirrors:
        return
    half = p.width / 2
    x = p.length / 2 - 0.75
    for side, sign in (("l", -1.0), ("r", 1.0)):
        name = f"mirror_{side}"
        parts[name] = Part(name, "body")
        v, t = _box([x, 2.18, sign * (half + 0.17)], (0.14, 0.42, 0.12))
        piece(name, "black", v, t)
        v, t = _box([x, 2.42, sign * (half + 0.07)], (0.05, 0.05, 0.24))
        piece(name, "black", v, t)


def _wheels(p: BusParams, parts, piece):
    """The car's wheels, two axles, the front ones steered by the pose."""
    front = _axles(p)[0]
    for xa, is_front in ((front, True), (front - p.wheelbase, False)):
        for side, s in (("l", -1.0), ("r", 1.0)):
            name = f"wheel_{'f' if is_front else 'r'}{side}"
            meshes, hub = _wheel(p, xa, s)
            parts[name] = Part(name, "wheel", pivot=hub, axis=np.array([0.0, 0.0, 1.0]), steer=is_front)
            for mat, pieces in meshes.items():
                for v, t in pieces:
                    piece(name, mat, v, t)


# --- the bus -------------------------------------------------------------------------------------------------


def build_bus(p: BusParams | None = None) -> Bus:
    """A bus from its measurements: a lofted body with rounded plan corners and windows, its doors, wheels and
    lamps. The default is a 12 m low-floor city bus."""
    p = p or BusParams()
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

    xs, P, raised = _grid(p)
    verts, tris, mats = _loft_faces(p, xs, P, raised)
    tris = tris[:, ::-1]  # the ring winds inward; the end faces are built outward already
    cverts, ctris, cmats = _cap_faces(p, P)
    base = len(verts)
    verts = np.concatenate([verts, cverts])
    tris = np.concatenate([tris, ctris + base])
    mats = np.concatenate([mats, cmats])
    smooth = _corner_normals(verts, tris, p.crease)
    for mat in np.unique(mats):
        sel = mats == mat
        t = tris[sel]
        piece("body", str(mat), verts[t].reshape(-1, 3), np.arange(3 * len(t)).reshape(-1, 3),
              smooth[sel].reshape(-1, 3))

    for mat, dv, dt in _decals(p, verts, tris):
        piece("body", mat, dv, dt)
    _coach_skirt(p, parts, piece)
    _doors(p, parts, piece)
    _mirrors(p, parts, piece)
    _wheels(p, parts, piece)
    return Bus(p, [part for part in parts.values() if part.pieces])


def coach() -> BusParams:
    """An intercity coach: a high floor, luggage bays along the lower flank, small tinted windows, one front door.

    13 m long, 3.6 m over the roof, with a raked screen and no route-sign band.
    """
    return BusParams(
        length=13.0, width=2.55, height=3.6, corner_radius=0.32, floor=1.00,
        front_overhang=2.70, wheelbase=6.60, wheel_radius=0.52, tyre_width=0.30,
        track=2.10, rim_ratio=0.64, arch_gap=0.05,
        windscreen_bottom=1.55, windscreen_top=3.05, windscreen_rake=0.45,
        sign_bottom=2.95, sign_top=3.35,
        rear_window_bottom=1.90, rear_window_top=2.70,
        window_bottom=1.95, window_top=2.75, pillar_spacing=1.55, pillar_width=0.12,
        doors=((0.03, 1.05, 1),), door_bottom=0.90,
        body_style="coach", luggage_bays=4, bay_bottom=0.36, bay_top=0.82,
        wheel="hubcap", mirrors=True,
    )
