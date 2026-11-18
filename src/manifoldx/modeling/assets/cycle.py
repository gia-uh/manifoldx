"""A parametric cycle: bicycles, cargo tricycles, cycle rickshaws and motorcycles.

A cycle is a frame of tubes between named joints, two or three wheels, a saddle and a handlebar, and one of three
kinds of work: pedals, a two-stroke engine or a hub motor. The handlebar and the fork are one part that turns about
the head tube, so a parked bicycle's bars and front wheel swing together; every wheel is a part that rolls about its
axle. A payload sits over the rear axle: nothing, a cargo box, or a bench for two under an optional canopy.

The measurements are the ones a drawing or a tape measure gives: the wheelbase, the wheel radius and tyre width,
the rear track of a tricycle, the joints of the frame in metres, and the height of the saddle, the bars and the
canopy. :func:`build_cycle` turns them into a :class:`Cycle`, a :class:`~manifoldx.modeling.assets.vehicle.Vehicle`
like the car: parts in vehicle coordinates, a pose that poses them, and one merged instancing mesh.

Axes are the ones every vehicle shares: metres, Y up, the front at +X, the right side at +Z, the centre on the
ground at the origin.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from manifoldx.modeling.assets.vehicle import PALETTE, ParamsRanges, Part, Vehicle, _rotate, choice, rng
from manifoldx.modeling.mesh import Mesh

_FRONT = np.array([1.0, 0.0, 0.0])
_UP = np.array([0.0, 1.0, 0.0])
_AXLE = np.array([0.0, 0.0, 1.0])


# --- measurements --------------------------------------------------------------------------------------------


@dataclass
class CycleParams(ParamsRanges):
    """Every measurement of a cycle, in metres. Joints are vehicle coordinates: X forward, Y up, Z to the right."""

    wheels: int = choice(2, 2, 3)  # 2 for a bicycle or a motorcycle, 3 for a tricycle
    layout: str = choice("inline", 'inline', 'delta', 'tadpole')  # "inline", "delta" (one front, two rear) or "tadpole" (two front, one rear)
    wheel_radius: float = rng(0.34, 0.15, 0.6, step=0.005, unit='m')
    tyre_width: float = rng(0.045, 0.02, 0.15, step=0.001, unit='m')
    rim_ratio: float = rng(0.86, 0.3, 0.95, step=0.01, unit='')  # rim radius over wheel radius
    spokes: int = rng(16, 0, 48, step=1, unit='')  # 0 draws a disc wheel
    wheelbase: float = rng(1.12, 0.6, 2.5, step=0.01, unit='m')
    rear_track: float = rng(0.86, 0.3, 1.5, step=0.01, unit='m')  # between the rear wheels of a delta, or the front wheels of a tadpole
    # The frame, as named joints. Empty means the defaults below; a preset names only the joints it moves.
    joints: dict = field(default_factory=dict)
    tube_radius: float = rng(0.021, 0.008, 0.08, step=0.001, unit='m')
    # The rider's seat and the bars.
    seat_style: str = choice("saddle", 'saddle', 'long', 'bench')  # "saddle" for a bicycle, "long" for a motorcycle, "bench" for a wide saddle
    seat_width: float = rng(0.19, 0.1, 0.6, step=0.01, unit='m')
    handlebar_width: float = rng(0.58, 0.3, 1.0, step=0.01, unit='m')
    # The payload over the rear axle.
    payload: str = choice("none", 'none', 'cargo_box', 'bench')  # "none", "cargo_box" or "bench"
    cargo_size: tuple = (1.05, 0.78, 0.55)  # length, width, height of the box
    bench_length: float = rng(0.82, 0.3, 1.5, step=0.01, unit='m')
    bench_width: float = rng(0.86, 0.3, 1.5, step=0.01, unit='m')
    canopy: bool = False
    canopy_height: float = rng(2.0, 1.0, 3.0, step=0.01, unit='m')  # the top of the hood, from the ground
    # How it is driven.
    power: str = choice("pedal", 'pedal', 'engine', 'hub')  # "pedal", "engine" or "hub"
    step_through: bool = False  # a scooter-like frame of panels instead of a diamond
    footboard: bool = False  # a flat board between the leg shield and the seat
    headlamp: str | None = choice(None, None, 'round', 'rect')  # None, "round" or "rect"
    tail_lamp: bool = False
    mudguards: bool = False
    # Paint: the frame, the box or bench, and the canopy.
    frame_paint: str = field(default="#b9c0c4", metadata={"kind": "colour"})
    box_paint: str = field(default="#6fc2b8", metadata={"kind": "colour"})
    canopy_paint: str = field(default="#efe4c6", metadata={"kind": "colour"})
    detail: float = rng(1.0, 0.25, 2.0, step=0.05, unit='x')  # scales the segments round wheels, tubes and the hood; 1 is today

    def _validate_extra(self) -> None:
        expected = 2 if self.layout == "inline" else 3
        if self.wheels != expected:
            raise ValueError(f"CycleParams.layout: {self.layout!r} needs {expected} wheels, not {self.wheels}")
        if self.wheelbase <= 0:
            raise ValueError(f"CycleParams.wheelbase: {self.wheelbase} must be positive")
        if self.wheelbase <= 2 * self.wheel_radius:
            raise ValueError(
                f"CycleParams.wheelbase: {self.wheelbase} does not clear two wheels of radius {self.wheel_radius}"
            )
        if self.wheels == 3 and self.rear_track <= self.tyre_width:
            raise ValueError(
                f"CycleParams.rear_track: {self.rear_track} does not clear a tyre width of {self.tyre_width}"
            )
        if self.canopy and self.payload == "none":
            raise ValueError("CycleParams.canopy: a canopy needs a bench or a box to stand on")


@dataclass
class Cycle(Vehicle):
    """A built cycle: its measurements, its rigid parts and the joints its frame was drawn through."""

    params: CycleParams
    joints: dict = field(default_factory=dict)

    def part(self, name: str) -> Part:
        return next(p for p in self.parts if p.name == name)

    @property
    def palette(self) -> dict:
        """The shared palette, plus the three paint colours this cycle was ordered in and its own materials."""
        p = self.params
        return {
            **PALETTE,
            "paint": (p.frame_paint, 0.35, 0.05),
            "box": (p.box_paint, 0.45, 0.0),
            "canopy": (p.canopy_paint, 0.65, 0.0),
            "saddle": ("#24242a", 0.75, 0.0),
            "engine": ("#b9bcc2", 0.35, 0.7),
        }

    def pose(self, wheel_turn=0.0, steer=0.0) -> dict:
        """Each part's (position, quaternion xyzw) so that cycle-space vertices land in their posed place.

        wheel_turn: radians the wheels have rolled. steer: radians the bars and the fork turn about the head tube,
        positive to the left, carrying the front wheel or wheels.
        """
        out = {}
        for p in self.parts:
            if p.kind == "wheel":
                q = _axis_angle(_AXLE, -wheel_turn)
                if p.steer:
                    q = _quat_mul(_axis_angle(_UP, steer), q)
            elif p.kind == "steer":
                q = _axis_angle(p.axis, steer)
            else:
                q = np.array([0.0, 0.0, 0.0, 1.0])
            out[p.name] = (p.pivot - _rotate(q, p.pivot), q)
        return out


# --- presets --------------------------------------------------------------------------------------------------


def bicycle() -> CycleParams:
    """A city bicycle: silver frame, a saddle, upright bars and a chain. 1.8 m long, bars at 1.1 m."""
    return CycleParams(wheels=2, layout="inline", wheel_radius=0.34, tyre_width=0.042, spokes=16,
                       wheelbase=1.12, tube_radius=0.019, seat_style="saddle", seat_width=0.20,
                       handlebar_width=0.56, power="pedal", frame_paint="#b9c0c4")


def cycle_rickshaw() -> CycleParams:
    """A bicitaxi: silver delta frame, a canopy over a two-person bench, pedalled. About 2.6 m long and 2.1 m tall."""
    return CycleParams(
        wheels=3, layout="delta", wheel_radius=0.33, tyre_width=0.042, spokes=18, wheelbase=1.75,
        rear_track=0.92, tube_radius=0.021, seat_style="saddle", handlebar_width=0.58,
        joints={
            "bottom_bracket": (-0.18, 0.27, 0.0),
            "seat": (-0.34, 0.88, 0.0),
            "head": (0.70, 0.68, 0.0),
            "head_top": (0.68, 0.84, 0.0),
            "handlebar": (0.64, 1.06, 0.0),
        },
        payload="bench", bench_length=0.88, bench_width=0.90, canopy=True, canopy_height=2.06,
        power="pedal", frame_paint="#b9c0c4", box_paint="#6fc2b8", canopy_paint="#efe4c6",
    )


def cargo_tricycle() -> CycleParams:
    """An electric cargo tricycle: a delta of two rear wheels under a yellow box, a canopy and pedals for show."""
    return CycleParams(
        wheels=3, layout="delta", wheel_radius=0.31, tyre_width=0.075, rim_ratio=0.58, spokes=8, wheelbase=1.90,
        rear_track=0.94, tube_radius=0.022, seat_style="saddle", seat_width=0.30, handlebar_width=0.62,
        joints={
            "bottom_bracket": (-0.52, 0.25, 0.0),
            "seat": (-0.02, 0.72, 0.0),
            "head": (0.72, 0.82, 0.0),
            "head_top": (0.70, 0.96, 0.0),
            "handlebar": (0.64, 1.02, 0.0),
        },
        payload="cargo_box", cargo_size=(1.18, 0.88, 0.56), canopy=True, canopy_height=1.98,
        power="hub", step_through=True, footboard=True, headlamp="rect", tail_lamp=True,
        frame_paint="#ffe94f", box_paint="#ffe94f", canopy_paint="#ffe94f",
    )


def motorcycle() -> CycleParams:
    """A 1970s two-stroke: mint tank and mudguards, a long flat seat, a round headlamp and a chrome exhaust."""
    return CycleParams(
        wheels=2, layout="inline", wheel_radius=0.30, tyre_width=0.085, rim_ratio=0.62, spokes=18, wheelbase=1.34,
        tube_radius=0.022, seat_style="long", seat_width=0.24, handlebar_width=0.64,
        joints={
            "bottom_bracket": (-0.26, 0.30, 0.0),
            "seat": (-0.20, 0.70, 0.0),
            "head": (0.50, 0.72, 0.0),
            "head_top": (0.47, 0.90, 0.0),
            "handlebar": (0.45, 1.01, 0.0),
        },
        power="engine", headlamp="round", tail_lamp=True, mudguards=True,
        frame_paint="#a8e6d8", box_paint="#1c1c1e", canopy_paint="#1c1c1e",
    )


def electric_moto() -> CycleParams:
    """An electric scooter: a coral step-through body, a mint seat, a footboard, a basket-sized battery and a hub."""
    return CycleParams(
        wheels=2, layout="inline", wheel_radius=0.25, tyre_width=0.075, rim_ratio=0.60, spokes=5, wheelbase=1.28,
        tube_radius=0.023, seat_style="long", seat_width=0.26, handlebar_width=0.62,
        joints={
            "bottom_bracket": (-0.22, 0.28, 0.0),
            "seat": (-0.24, 0.66, 0.0),
            "head": (0.52, 0.80, 0.0),
            "head_top": (0.49, 0.96, 0.0),
            "handlebar": (0.46, 1.02, 0.0),
        },
        power="hub", step_through=True, footboard=True, headlamp="round", tail_lamp=True,
        frame_paint="#e8553f", box_paint="#bfe8d8", canopy_paint="#bfe8d8",
    )


# --- small geometry helpers -----------------------------------------------------------------------------------


def _axis_angle(axis, angle) -> np.ndarray:
    axis = np.asarray(axis, np.float64)
    axis = axis / np.linalg.norm(axis)
    half = angle / 2.0
    return np.array([*(axis * np.sin(half)), np.cos(half)])


def _quat_mul(a, b) -> np.ndarray:
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return np.array([
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    ])


def _turn(mesh: Mesh, angle: float, axis) -> Mesh:
    """A mesh turned by `angle` radians about `axis` through the origin."""
    axis = np.asarray(axis, float)
    axis = axis / np.linalg.norm(axis)
    c, s = np.cos(angle), np.sin(angle)
    x, y, z = axis
    r = np.array([[c + x * x * (1 - c), x * y * (1 - c) - z * s, x * z * (1 - c) + y * s],
                  [y * x * (1 - c) + z * s, c + y * y * (1 - c), y * z * (1 - c) - x * s],
                  [z * x * (1 - c) - y * s, z * y * (1 - c) + x * s, c + z * z * (1 - c)]])
    return mesh.with_positions(np.asarray(mesh.positions, float) @ r.T)


def _scale(mesh: Mesh, sx: float, sy: float, sz: float) -> Mesh:
    return mesh.with_positions(np.asarray(mesh.positions, float) * np.array([sx, sy, sz]))


def _add(pieces: dict, material: str, mesh: Mesh) -> None:
    """Add a piece to a part's material bucket, welding the vertex lists together."""
    old = pieces.get(material)
    if old is None:
        pieces[material] = mesh
        return
    pieces[material] = Mesh(
        positions=np.vstack([old.positions, mesh.positions]),
        faces=np.vstack([old.faces, np.asarray(mesh.faces) + len(old.positions)]),
    )


def _rod(a, b, radius: float, segments: int = 8) -> Mesh:
    """A tube between two points: the frame's builder, a cylinder stood on its end and turned to the bone."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = b - a
    length = float(np.linalg.norm(d))
    if length < 1e-9:
        return Mesh.cylinder(radius, 1e-6, max(3, segments)).translate(a)
    y = d / length
    helper = _FRONT if abs(y @ _FRONT) < 0.9 else _AXLE
    x = np.cross(y, helper)
    x /= np.linalg.norm(x)
    z = np.cross(x, y)
    body = Mesh.cylinder(radius, length, max(3, segments))
    return Mesh(positions=np.asarray(body.positions, float) @ np.column_stack([x, y, z]).T + (a + b) / 2,
                faces=body.faces)


def _box_mesh(width: float, height: float, depth: float) -> Mesh:
    """A box with every face its own vertices, so a thin slab gets flat normals instead of cancelling ones."""
    hx, hy, hz = width / 2, height / 2, depth / 2
    p = [(-hx, -hy, -hz), (hx, -hy, -hz), (hx, hy, -hz), (-hx, hy, -hz),
         (-hx, -hy, hz), (hx, -hy, hz), (hx, hy, hz), (-hx, hy, hz)]
    quads = [(0, 1, 2, 3), (5, 4, 7, 6), (4, 0, 3, 7), (1, 5, 6, 2), (3, 2, 6, 7), (4, 5, 1, 0)]
    positions, faces = [], []
    for a, b, c, d in quads:
        base = len(positions)
        positions += [p[a], p[b], p[c], p[d]]
        faces += [[base, base + 2, base + 1], [base, base + 3, base + 2]]
    return Mesh(positions=np.array(positions, np.float32), faces=np.array(faces, np.uint32))


def _box_at(center, size, pitch: float = 0.0, yaw: float = 0.0) -> Mesh:
    """A box of `size` (length, height, width) centred at `center`, pitched about Z and yawed about Y."""
    mesh = _box_mesh(float(size[0]), float(size[1]), float(size[2]))
    if pitch:
        mesh = _turn(mesh, pitch, _AXLE)
    if yaw:
        mesh = _turn(mesh, yaw, _UP)
    return mesh.translate(np.asarray(center, float))


def _arc(center, radius: float, thickness: float, width: float, a0: float, a1: float, segments: int) -> Mesh:
    """A curved shell round a wheel: a mudguard or a canopy bow, from angle a0 to a1 in the XY plane."""
    c = np.asarray(center, float)
    angles = np.linspace(a0, a1, max(2, segments) + 1)
    inner, outer = radius - thickness / 2, radius + thickness / 2
    rings = []
    for a in angles:
        x, y = np.cos(a), np.sin(a)
        rings.append([
            (c[0] + inner * x, c[1] + inner * y, c[2] - width / 2),
            (c[0] + outer * x, c[1] + outer * y, c[2] - width / 2),
            (c[0] + outer * x, c[1] + outer * y, c[2] + width / 2),
            (c[0] + inner * x, c[1] + inner * y, c[2] + width / 2),
        ])
    positions = np.array(rings, float).reshape(-1, 3)
    faces = []
    for i in range(len(rings) - 1):
        a, b = 4 * i, 4 * (i + 1)
        faces += [[a + 1, b + 1, b + 2], [a + 1, b + 2, a + 2],  # outside
                  [a + 3, b + 3, b + 0], [a + 3, b + 0, a + 0],  # inside
                  [a + 0, b + 0, b + 1], [a + 0, b + 1, a + 1],  # the -z rim
                  [a + 2, b + 2, b + 3], [a + 2, b + 3, a + 3]]  # the +z rim
    faces += [[0, 2, 1], [0, 3, 2], [len(positions) - 4, len(positions) - 3, len(positions) - 2],
              [len(positions) - 4, len(positions) - 2, len(positions) - 1]]
    return Mesh(positions=positions.astype(np.float32), faces=np.array(faces, np.uint32))


def _round(radius: float, sx: float, sy: float, sz: float, at) -> Mesh:
    """A rounded lump: an icosphere stretched to size and moved into place."""
    return _scale(Mesh.icosphere(2, radius), sx, sy, sz).translate(np.asarray(at, float))


# --- the frame ------------------------------------------------------------------------------------------------


def _default_joints(p: CycleParams) -> dict:
    """The joints a diamond frame needs, from the wheelbase, the wheel radius and nothing else."""
    r, xr, xf, wb = p.wheel_radius, -p.wheelbase / 2, p.wheelbase / 2, p.wheelbase
    return {
        "rear_axle": (xr, r, 0.0),
        "front_axle": (xf, r, 0.0),
        "bottom_bracket": (xr + 0.33 * wb, 0.80 * r, 0.0),
        "seat": (xr + 0.16 * wb, r + 0.55, 0.0),
        "head": (xf - 0.10, r + 0.34, 0.0),
        "head_top": (xf - 0.12, r + 0.50, 0.0),
        "handlebar": (xf - 0.12, r + 0.72, 0.0),
    }


def _joints(p: CycleParams) -> dict:
    out = _default_joints(p)
    out.update(p.joints)
    return {k: np.asarray(v, float) for k, v in out.items()}


def _segments(p: CycleParams) -> int:
    return max(4, int(round(8 * p.detail)))


def _frame(p: CycleParams, j: dict, body: dict) -> None:
    """The diamond frame: three triangles of tubes between the head, the bracket, the seat and the rear axle."""
    tube = p.tube_radius
    seg = _segments(p)
    if not p.step_through:
        for a, b in (("head", "bottom_bracket"), ("head", "seat"), ("seat", "bottom_bracket")):
            _add(body, "paint", _rod(j[a], j[b], tube, seg))
    _add(body, "paint", _rod(j["head"], j["head_top"], tube * 1.3, seg))
    rear_z = p.rear_track / 2 if p.wheels == 3 and p.layout == "delta" else 0.0
    for side in (-1.0, 1.0):
        start = j["bottom_bracket"] + np.array([0.0, 0.0, side * 0.03])
        _add(body, "paint", _rod(start, (j["rear_axle"][0], p.wheel_radius, side * rear_z), tube * 0.85, seg))
        start = j["seat"] + np.array([0.0, 0.0, side * 0.03])
        _add(body, "paint", _rod(start, (j["rear_axle"][0], p.wheel_radius, side * rear_z), tube * 0.8, seg))


def _panels(p: CycleParams, j: dict, body: dict) -> None:
    """The scooter's body: a leg shield, a footboard, a rear box and a seat post, in place of a diamond."""
    seg = _segments(p)
    head, seat, bb = j["head"], j["seat"], j["bottom_bracket"]
    board = max(0.16, p.wheel_radius - 0.09)
    if p.footboard:
        _add(body, "paint", _box_at((head[0] - 0.36, board, 0.0), (0.62, 0.045, 0.38)))
        _add(body, "trim", _box_at((head[0] - 0.36, board + 0.028, 0.0), (0.48, 0.015, 0.30)))
    height = max(0.35, head[1] - board - 0.02)
    _add(body, "paint", _box_at((head[0] - 0.06, board + height / 2, 0.0), (0.12, height, 0.44), pitch=0.14))
    front = seat[0] + 0.14
    back = j["rear_axle"][0] - 0.12
    bottom = max(0.14, p.wheel_radius * 0.55)
    top = seat[1] + 0.04
    _add(body, "paint", _box_at(((front + back) / 2, (bottom + top) / 2, 0.0),
                                (front - back, top - bottom, 0.35)))
    _add(body, "paint", _round(1.0, 0.30, 0.12, 0.185,
                               (j["rear_axle"][0] + 0.02, top - 0.10, 0.0)))
    _add(body, "paint", _rod(seat, seat + np.array([0.0, 0.05, 0.0]), p.tube_radius, seg))
    if p.tail_lamp:
        _add(body, "tail", _box_at((back - 0.02, top - 0.12, 0.0), (0.02, 0.08, 0.14)))
    _add(body, "chrome", _rod((seat[0] - 0.30, seat[1] + 0.02, -0.13),
                              (seat[0] - 0.30, seat[1] + 0.02, 0.13), 0.012, seg))


def _steer(p: CycleParams, j: dict, steer: dict) -> None:
    """The fork, the stem and the bars, all of one part so that they turn about the head tube together."""
    seg = _segments(p)
    tube = p.tube_radius
    head, top, bar = j["head"], j["head_top"], j["handlebar"]
    front_z = p.rear_track / 2 if p.layout == "tadpole" else 0.0
    if front_z:
        for side in (-1.0, 1.0):
            _add(steer, "paint", _rod((head[0], head[1] + 0.02, side * 0.03),
                                      (j["front_axle"][0], p.wheel_radius, side * front_z), tube, seg))
    else:
        for side in (-1.0, 1.0):
            _add(steer, "paint", _rod((head[0], head[1] + 0.02, side * 0.045),
                                      (j["front_axle"][0], p.wheel_radius, side * 0.045), tube, seg))
    _add(steer, "paint", _rod(top, bar, tube * 0.9, seg))
    half = p.handlebar_width / 2
    _add(steer, "paint", _rod((bar[0], bar[1], -half * 0.55), (bar[0], bar[1], half * 0.55), tube * 0.95, seg))
    for side in (-1.0, 1.0):
        end = np.array([bar[0] - 0.07, bar[1] + 0.005, side * half])
        _add(steer, "paint", _rod((bar[0], bar[1], side * half * 0.55), end, tube * 0.85, seg))
        grip = np.array([bar[0] - 0.045, bar[1] + 0.003, side * (half - 0.10)])
        _add(steer, "black", _rod(grip, end + np.array([-0.01, 0.0, 0.0]), 0.024, seg))
    if p.headlamp:
        lamp_at = np.array([head[0] + 0.09, head[1] + 0.14, 0.0])
        if p.headlamp == "round":
            bucket = _turn(Mesh.cylinder(0.105, 0.075, max(8, int(round(20 * p.detail)))), np.pi / 2, _AXLE)
            _add(steer, "chrome", bucket.translate(lamp_at))
            face = _turn(Mesh.cylinder(0.088, 0.012, max(8, int(round(20 * p.detail)))), np.pi / 2, _AXLE)
            _add(steer, "lamp", face.translate(lamp_at + np.array([0.043, 0.0, 0.0])))
        else:
            _add(steer, "chrome", _box_at(lamp_at, (0.07, 0.13, 0.20)))
            _add(steer, "lamp", _box_at(lamp_at + np.array([0.04, 0.0, 0.0]), (0.012, 0.10, 0.17)))
        _add(steer, "chrome", _rod((head[0] + 0.02, head[1] + 0.10, -0.08),
                                   (head[0] + 0.02, head[1] + 0.10, 0.08), 0.012, seg))
    if p.mudguards or p.step_through:
        _add(steer, "paint", _arc(j["front_axle"], p.wheel_radius + 0.055, 0.012,
                                  p.tyre_width + 0.06, 0.30, 2.60, max(4, int(round(10 * p.detail)))))


def _seat(p: CycleParams, j: dict, body: dict) -> None:
    """The rider's saddle, long seat or bench, on the seat joint."""
    seg = _segments(p)
    at = np.asarray(j["seat"], float)
    _add(body, "paint", _rod(at, at + np.array([0.0, 0.05, 0.0]), p.tube_radius * 0.9, seg))
    if p.seat_style == "long":
        _add(body, "saddle", _box_at(at + np.array([0.04, 0.085, 0.0]), (0.58, 0.075, p.seat_width)))
        _add(body, "saddle", _box_at(at + np.array([-0.24, 0.105, 0.0]), (0.12, 0.10, p.seat_width)))
    elif p.seat_style == "bench":
        _add(body, "saddle", _box_at(at + np.array([0.0, 0.06, 0.0]), (0.44, 0.07, p.seat_width)))
        _add(body, "saddle", _box_at(at + np.array([-0.20, 0.15, 0.0]), (0.08, 0.22, p.seat_width)))
    else:
        _add(body, "saddle", _round(1.0, 0.085, 0.035, p.seat_width * 0.55, at + np.array([-0.02, 0.07, 0.0])))
        _add(body, "saddle", _round(1.0, 0.12, 0.028, p.seat_width * 0.32, at + np.array([0.08, 0.065, 0.0])))


# --- the payload ---------------------------------------------------------------------------------------------


def _bench(p: CycleParams, j: dict, body: dict) -> None:
    """A bench for two over the rear axle: a slatted cushion, a backrest and an arm rail."""
    seg = _segments(p)
    x, y, z = j["rear_axle"][0] - 0.04, 0.52, 0.0
    half_l, half_w = p.bench_length / 2, p.bench_width / 2
    _add(body, "paint", _box_at((x, y - 0.09, z), (p.bench_length, 0.07, p.bench_width)))
    slats = 5
    for k in range(slats):
        cx = x - half_l + (k + 0.5) * p.bench_length / slats
        material = "box" if k % 2 == 0 else "canopy"
        _add(body, material, _box_at((cx, y, z), (p.bench_length / slats - 0.012, 0.09, p.bench_width)))
    _add(body, "box", _box_at((x - half_l - 0.03, y + 0.20, z), (0.09, 0.46, p.bench_width)))
    _add(body, "paint", _rod((x - half_l + 0.02, y + 0.06, -half_w - 0.04),
                             (x + half_l - 0.06, y + 0.06, -half_w - 0.04), p.tube_radius * 0.8, seg))
    _add(body, "paint", _rod((x - half_l + 0.02, y + 0.06, half_w + 0.04),
                             (x + half_l - 0.06, y + 0.06, half_w + 0.04), p.tube_radius * 0.8, seg))


def _cargo_box(p: CycleParams, j: dict, body: dict) -> None:
    """A box over the rear axle, with a rim and a tail lamp."""
    seg = _segments(p)
    length, width, height = p.cargo_size
    x, z = j["rear_axle"][0] + 0.20, 0.0
    bottom = max(p.wheel_radius * 1.15, 0.36)
    _add(body, "box", _box_at((x, bottom + height / 2, z), (length, height, width)))
    _add(body, "paint", _box_at((x, bottom + height, z), (length + 0.03, 0.035, width + 0.03)))
    _add(body, "trim", _box_at((x, bottom - 0.015, z), (length - 0.04, 0.03, width - 0.06)))
    for side in (-1.0, 1.0):
        _add(body, "paint", _rod((x - length / 2, bottom, side * width / 2),
                                 (x - length / 2, bottom + height, side * width / 2), p.tube_radius * 0.7, seg))
    if p.tail_lamp:
        _add(body, "tail", _box_at((x - length / 2 - 0.01, bottom + height * 0.7, z), (0.02, 0.10, 0.14)))


def _canopy(p: CycleParams, j: dict, body: dict) -> None:
    """A hood on two bows, on four posts over the bench or the box."""
    seg = _segments(p)
    top = p.canopy_height
    if p.payload == "bench":
        length, width = p.bench_length + 0.30, p.bench_width + 0.14
        x = j["rear_axle"][0] - 0.04 + 0.06
        base = 0.62
    else:
        length, width = p.cargo_size[0] + 0.36, p.cargo_size[1] + 0.12
        x = j["rear_axle"][0] + 0.28
        base = max(p.wheel_radius * 1.15, 0.36) + p.cargo_size[2]
    half_l, half_w = length / 2, width / 2
    _add(body, "canopy", _box_at((x, top - 0.012, 0.0), (length, 0.025, width)))
    for bx in (x - half_l + 0.09, x + half_l - 0.09):
        bow = [(bx, top - 0.20, -half_w), (bx, top - 0.07, -half_w * 0.5), (bx, top - 0.025, 0.0),
               (bx, top - 0.07, half_w * 0.5), (bx, top - 0.20, half_w)]
        for a, b in zip(bow, bow[1:]):
            _add(body, "paint", _rod(a, b, 0.016, seg))
        for side in (-1.0, 1.0):
            _add(body, "paint", _rod((bx, base, side * (half_w - 0.06)),
                                     (bx, top - 0.20, side * half_w), 0.019, seg))
    for side in (-1.0, 1.0):
        _add(body, "paint", _rod((x - half_l + 0.09, base, side * (half_w - 0.06)),
                                 (x + half_l - 0.09, base, side * (half_w - 0.06)), 0.016, seg))


# --- power ---------------------------------------------------------------------------------------------------


def _pedals(p: CycleParams, j: dict, body: dict) -> None:
    """Cranks, pedals, a chainring and a chain to the rear hub."""
    seg = _segments(p)
    bb = np.asarray(j["bottom_bracket"], float)
    _add(body, "chrome", _rod(bb + np.array([0.0, 0.0, -0.09]), bb + np.array([0.0, 0.0, 0.09]), 0.018, seg))
    ring = _turn(Mesh.cylinder(0.095, 0.008, max(8, int(round(24 * p.detail)))), np.pi / 2, _FRONT)
    _add(body, "chrome", ring.translate(bb + np.array([0.0, 0.0, 0.075])))
    for side, sign in ((-1.0, 1.0), (1.0, -1.0)):
        arm = np.array([bb[0] + sign * 0.07, bb[1], side * 0.085])
        _add(body, "engine", _box_at(arm, (0.16, 0.028, 0.016)))
        _add(body, "black", _box_at((bb[0] + sign * 0.16, bb[1], side * 0.115), (0.09, 0.022, 0.07)))
    for up in (-1.0, 1.0):
        _add(body, "trim", _rod((bb[0], bb[1] + up * 0.09, 0.07),
                                (j["rear_axle"][0] + 0.02, p.wheel_radius + up * 0.035, 0.055),
                                0.006, max(3, seg // 2)))


def _engine(p: CycleParams, j: dict, body: dict) -> None:
    """A two-stroke: a tank on the top tube, a finned block under it, an exhaust to the rear and mudguards."""
    seg = _segments(p)
    head, seat, bb = j["head"], j["seat"], j["bottom_bracket"]
    tank = np.array([(head[0] + seat[0]) / 2 - 0.02, (head[1] + seat[1]) / 2 + 0.13, 0.0])
    _add(body, "paint", _round(1.0, 0.23, 0.115, 0.15, tank))
    block = np.array([bb[0] + 0.22, bb[1] + 0.02, 0.0])
    _add(body, "engine", _box_at(block, (0.30, 0.16, 0.20)))
    _add(body, "engine", _box_at((block[0] - 0.01, block[1] + 0.17, 0.0), (0.20, 0.16, 0.17)))
    for k in range(5):
        _add(body, "engine", _box_at((block[0] - 0.01, block[1] + 0.11 + k * 0.026, 0.0),
                                     (0.24, 0.010, 0.20)))
    _add(body, "black", _box_at((block[0] + 0.02, block[1] - 0.10, 0.0), (0.24, 0.08, 0.18)))
    pipe = [np.array([block[0] + 0.14, block[1] - 0.06, -0.10]), np.array([bb[0] - 0.02, 0.24, -0.13]),
            np.array([bb[0] - 0.34, 0.22, -0.15]), np.array([j["rear_axle"][0] + 0.12, 0.25, -0.16])]
    for a, b in zip(pipe, pipe[1:]):
        _add(body, "chrome", _rod(a, b, 0.032, seg))
    _add(body, "chrome", _rod(pipe[-2], pipe[-1] + np.array([0.10, 0.0, 0.0]), 0.055, seg))
    if p.tail_lamp:
        _add(body, "chrome", _rod((seat[0] - 0.24, 0.70, 0.0), (j["rear_axle"][0] - 0.26, 0.58, 0.0),
                                   0.010, seg))
        _add(body, "tail", _box_at((j["rear_axle"][0] - 0.26, 0.55, 0.0), (0.07, 0.09, 0.16)))
    if p.mudguards:
        _add(body, "paint", _arc(j["rear_axle"], p.wheel_radius + 0.055, 0.012, p.tyre_width + 0.06,
                                 0.35, 2.75, max(4, int(round(10 * p.detail)))))


def _hub(p: CycleParams, j: dict, body: dict) -> None:
    """A battery box under the seat, and a hub motor inside the rear wheel (added by the wheel builder)."""
    head, seat, bb = j["head"], j["seat"], j["bottom_bracket"]
    if p.step_through:
        _add(body, "black", _box_at((j["rear_axle"][0] + 0.16, 0.48, 0.0), (0.34, 0.15, 0.24)))
    else:
        _add(body, "black", _box_at(((seat[0] + bb[0]) / 2, 0.44, 0.0), (0.30, 0.20, 0.20)))


# --- wheels --------------------------------------------------------------------------------------------------


def _wheel(p: CycleParams, center, rear: bool) -> dict:
    """Tyre, rim, spokes or a disc, and a hub, as {material: mesh} round one axle."""
    r, w = p.wheel_radius, p.tyre_width
    c = np.asarray(center, float)
    seg = max(8, int(round(32 * p.detail)))
    minor = max(3, int(round(8 * p.detail)))
    out = {}
    tyre = _turn(Mesh.torus(max(r - w / 2, 0.02), w / 2, seg, minor), np.pi / 2, _FRONT).translate(c)
    out["tyre"] = tyre
    rr = r * p.rim_ratio
    rim = _turn(Mesh.torus(rr, max(w * 0.35, 0.008), seg, minor), np.pi / 2, _FRONT).translate(c)
    out["rim"] = rim
    hub_r = max(0.028, r * 0.085)
    hub = _turn(Mesh.cylinder(hub_r, w * 1.25, max(6, seg // 2)), np.pi / 2, _FRONT).translate(c)
    out["chrome"] = hub
    count = max(0, int(round(p.spokes * p.detail))) if p.spokes else 0
    if count:
        spokes = []
        for k in range(count):
            a = 2 * np.pi * k / count
            start = c + np.array([np.cos(a) * hub_r, np.sin(a) * hub_r, w * (0.3 if k % 2 else -0.3)])
            end = c + np.array([np.cos(a) * (rr - 0.004), np.sin(a) * (rr - 0.004), 0.0])
            spokes.append(_rod(start, end, max(0.0035, r * 0.018), max(2, seg // 3)))
        out["spoke"] = _weld(spokes)
    else:
        disc = _turn(Mesh.cylinder(rr - 0.006, 0.008, seg), np.pi / 2, _FRONT).translate(c)
        out["rim"] = _weld([out["rim"], disc])
    if rear and p.power == "hub":
        motor = _turn(Mesh.cylinder(r * 0.30, w * 1.35, max(8, seg // 2)), np.pi / 2, _FRONT).translate(c)
        out["engine"] = motor
    return out


def _weld(meshes) -> Mesh:
    """Several meshes of one material, welded into one."""
    positions = np.vstack([np.asarray(m.positions, float) for m in meshes])
    faces, base = [], 0
    for m in meshes:
        faces.append(np.asarray(m.faces) + base)
        base += len(m.positions)
    return Mesh(positions=positions.astype(np.float32), faces=np.vstack(faces).astype(np.uint32))


# --- the builder ---------------------------------------------------------------------------------------------


def build_cycle(p: CycleParams | None = None) -> Cycle:
    """Build a cycle from its measurements: a frame, a steering part, 2 or 3 wheels, a payload and its power."""
    p = CycleParams() if p is None else p
    p.validate()
    j = _joints(p)
    body, steer = {}, {}
    _frame(p, j, body)
    if p.step_through:
        _panels(p, j, body)
    _seat(p, j, body)
    if p.payload == "bench":
        _bench(p, j, body)
    elif p.payload == "cargo_box":
        _cargo_box(p, j, body)
    if p.canopy:
        _canopy(p, j, body)
    if p.power == "pedal":
        _pedals(p, j, body)
    elif p.power == "engine":
        _engine(p, j, body)
    else:
        _hub(p, j, body)
    _steer(p, j, steer)

    parts = [Part("frame", "body", body, np.zeros(3)),
             Part("steer", "steer", steer, j["head"], j["head_top"] - j["head"])]
    r = p.wheel_radius
    xf, xr = j["front_axle"][0], j["rear_axle"][0]
    track = p.rear_track / 2
    if p.layout == "inline":
        front = [("wheel_front", (xf, r, 0.0))]
        rear = [("wheel_rear", (xr, r, 0.0))]
    elif p.layout == "delta":
        front = [("wheel_front", (xf, r, 0.0))]
        rear = [("wheel_rear_left", (xr, r, -track)), ("wheel_rear_right", (xr, r, track))]
    else:
        front = [("wheel_front_left", (xf, r, -track)), ("wheel_front_right", (xf, r, track))]
        rear = [("wheel_rear", (xr, r, 0.0))]
    for name, at in front:
        parts.append(Part(name, "wheel", _wheel(p, at, rear=False), np.asarray(at, float), _AXLE, steer=True))
    for name, at in rear:
        parts.append(Part(name, "wheel", _wheel(p, at, rear=True), np.asarray(at, float), _AXLE))
    return Cycle(params=p, parts=parts, joints=j)
