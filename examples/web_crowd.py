"""A crowd of boxes the ECS runs in the page: the app `manifoldx export --web` replays end to end.

Space throws a ring of 24 boxes up from above the ground; each wave's ring is wider and takes
the next of three colours. The boxes fall, spin, and are destroyed once they drop below
y = -10. A left click picks the box under the pointer: it turns gold and the last one gets its
colour back; a click on the sky clears the pick. The panel counts the boxes and the waves and
shows how fast the fastest box falls.

Every system is a query over the entity store: the spawn, the drift and the pick are numpy
expressions on whole columns, which the web export transpiles; the ground and the camera are
built at setup, which stays in Python.

    uv run python examples/web_crowd.py
    uv run manifoldx export --web examples/web_crowd.py out/
"""

import math

import numpy as np

import manifoldx as mx
from manifoldx.components import Component, Material, Mesh, Transform
from manifoldx.gui import Panel, ValueDisplay
from manifoldx.types import Float, Vector3

RING = 24  # boxes per wave
RADIUS = 4.0  # the first ring's radius, m; each wave adds 1
HEIGHT = 6.0  # where a ring starts, m
OUT = 1.5  # outward speed, m/s
UP = 3.0  # upward speed, m/s
FLOOR = -10.0  # below this a box is destroyed
HIT = 0.7  # how close to a box's centre a click's ray must pass, m

GRAVITY = np.array([0.0, -6.0, 0.0], np.float32)
SPIN = np.array([0.0, 0.0499792, 0.0, 0.9987503], np.float32)  # 0.1 rad about y per frame


class Velocity(Component):
    v: Vector3


class Wave(Component):
    k: Float  # which of COLORS the box wears, so a pick can give it back


engine = mx.Engine("Web crowd", width=800, height=500)
engine.state.setdefault("waves", 0)
engine.state.setdefault("boxes", 0)
engine.state.setdefault("fall", 0.0)  # the fastest box's drop this frame, m
engine.state.setdefault("picked", None)  # the gold box's entity index
engine.state.setdefault("picked_k", 0)  # its colour, to give back

CUBE = mx.geometry.cube(1, 1, 1)
COLORS = [mx.material.phong("#c0392b"), mx.material.phong("#2e86c1"), mx.material.phong("#27ae60")]
GOLD = mx.material.phong("#f2c14e")


def spawn_wave(q: mx.Query[Transform], dt: float):
    if not engine.input.just_pressed("Space"):
        return
    wave: int = engine.state["waves"]
    a = np.arange(RING) * (2.0 * np.pi / RING)
    r = RADIUS + wave
    pos = np.column_stack([np.cos(a) * r, np.cos(a) * 0.0 + HEIGHT, np.sin(a) * r])
    vel = np.column_stack([np.cos(a) * OUT, np.cos(a) * 0.0 + UP, np.sin(a) * OUT])
    k = wave % 3
    engine.spawn(
        Mesh(CUBE),
        Material(COLORS[k]),
        Transform(pos=pos, scale=(0.8, 0.8, 0.8)),
        Velocity(v=vel),
        Wave(k=float(k)),
        n=RING,
    )
    engine.state["waves"] = wave + 1


def drift(q: mx.Query[Transform, Velocity], dt: float):
    engine.state["boxes"] = len(q)
    if len(q) > 0:
        step = q[Velocity].v.data[:, 1] * dt  # how far each box rises this frame, float32
        engine.state["fall"] = -float(step[int(np.argmin(step))])
    q[Velocity].v += GRAVITY * dt
    q[Transform].pos += q[Velocity].v * dt
    q[Transform].rot += SPIN
    engine.destroy(q[Transform].pos[:, 1] < FLOOR)


def ray(cam, x: float, y: float) -> list[float]:
    """The unit direction from the camera through pixel (x, y), in float64."""
    f = cam.get_forward()
    fx, fy, fz = float(f[0]), float(f[1]), float(f[2])
    n = math.hypot(fz, fx)  # right = forward x (0, 1, 0), normalised
    rx, rz = -fz / n, fx / n
    ux, uy, uz = -rz * fy, rz * fx - rx * fz, rx * fy  # up = right x forward
    t = math.tan(math.radians(cam.fov) / 2)
    sx = (2.0 * x / engine.w - 1.0) * t * engine.w / engine.h
    sy = (1.0 - 2.0 * y / engine.h) * t
    d = [fx + rx * sx + ux * sy, fy + uy * sy, fz + rz * sx + uz * sy]
    m = math.sqrt(d[0] * d[0] + d[1] * d[1] + d[2] * d[2])
    return [d[0] / m, d[1] / m, d[2] / m]


def pick(q: mx.Query[Transform, Velocity, Wave], dt: float):
    """After drift: the rows are the ones before the flush, as on the desktop."""
    ids = [i for i in q]
    prev: int | None = engine.state["picked"]
    if prev is not None and prev not in ids:  # the gold box fell
        prev = None
        engine.state["picked"] = None
    if not engine.input.just_mouse_pressed(1):
        return
    cam = engine.camera
    d: list[float] = ray(cam, float(engine.input.mouse_pos[0]), float(engine.input.mouse_pos[1]))
    origin = np.array([float(cam.position[0]), float(cam.position[1]), float(cam.position[2])])
    rel = q[Transform].pos.data - origin
    along = rel[:, 0] * d[0] + rel[:, 1] * d[1] + rel[:, 2] * d[2]
    foot = np.column_stack([along * d[0], along * d[1], along * d[2]])
    off = np.linalg.norm(rel - foot, axis=1)
    dist = np.where((off < HIT) & (along > 0.0), along, np.inf)
    if prev is not None:
        k: int = engine.state["picked_k"]
        engine.set_material(prev, COLORS[k])
    engine.state["picked"] = None
    if len(ids) == 0:
        return
    j = int(np.argmin(dist))
    if float(dist[j]) < math.inf:
        engine.set_material(ids[j], GOLD)
        engine.state["picked"] = ids[j]
        engine.state["picked_k"] = int(float(q[Wave].k.data[:, 0][j]))


def boxes_line() -> str:
    n: int = engine.state["boxes"]
    return f"boxes: {n}"


def fall_line() -> str:
    f: float = engine.state["fall"]
    return f"fastest fall: {f * 100.0:.1f} cm per frame"


def waves_line() -> str:
    n: int = engine.state["waves"]
    return f"waves: {n} (Space throws one)"


def picked_line() -> str:
    i: int | None = engine.state["picked"]
    return "picked: nothing" if i is None else f"picked: entity {i}"


engine.spawn(
    Mesh(CUBE),
    Material(mx.material.phong("#d5d8c8")),
    Transform(pos=(0.0, -0.05, 0.0), scale=(30.0, 0.1, 30.0)),
)
Velocity.register(engine.store)  # drift's and pick's queries read them before the first wave
Wave.register(engine.store)
for i, x in enumerate((0.0, 12.0, 22.0)):
    engine.camera.position[i] = x
    engine.camera.target[i] = 0.0

engine.system(spawn_wave)
engine.system(drift)
engine.system(pick)
engine.gui.append(
    Panel(
        children=[
            ValueDisplay(boxes_line),
            ValueDisplay(fall_line),
            ValueDisplay(waves_line),
            ValueDisplay(picked_line),
        ],
        anchor="top-left",
        offset=(16, 16),
        style_overrides={"width": 260, "height": 98, "padding": 8, "bg": "#1a1a1aC0"},
    )
)

if __name__ == "__main__":
    engine.cli()
