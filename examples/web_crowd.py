"""A crowd of boxes the ECS runs in the page: the app `manifoldx export --web` replays end to end.

Space throws a ring of 24 boxes up from above the ground; each wave's ring is wider, jittered
by a seeded generator, and takes the next of three colours. The boxes fall, spin, and are
destroyed once they drop below y = -10. A left click picks the box under the pointer: it turns
gold and the last one gets its colour back; a click on the sky clears the pick. The wheel moves
the camera in and out. The first panel counts the boxes and the waves, shows how fast the
fastest box falls and reads the clock.

The second panel holds the controls: "Throw a wave" throws one as Space does (and clears the
pick); the gravity Slider sets how hard the boxes fall (click it, then the arrow keys step it);
the spin Toggle turns the spin on and off; the wind Joystick blows the boxes sideways while it
is held; the compass is an SVG Image. A handler per event keeps a count or the last payload in
engine.state.

Every system is a query over the entity store: the spawn, the drift and the pick are numpy
expressions on whole columns, which the web export transpiles; the ground and the camera are
built at setup, which stays in Python.

    uv run python examples/web_crowd.py
    uv run manifoldx export --web examples/web_crowd.py out/
"""

import math
import random
import time

import numpy as np

import manifoldx as mx
from manifoldx.components import Component, Material, Mesh, Transform
from manifoldx.gui import Button, Image, Joystick, Panel, Slider, Toggle, ValueDisplay
from manifoldx.types import Float, Vector3

RING = 24  # boxes per wave
RADIUS = 4.0  # the first ring's radius, m; each wave adds 1
HEIGHT = 6.0  # where a ring starts, m
OUT = 1.5  # outward speed, m/s
UP = 3.0  # upward speed, m/s
FLOOR = -10.0  # below this a box is destroyed
HIT = 0.7  # how close to a box's centre a click's ray must pass, m
JITTER = 0.3  # each box's radius moves by up to this much, m
WIND_SPEED = 2.0  # what the Joystick at its rim adds to a box's speed each second, m/s
ZOOM = 0.002  # the camera's distance change per wheel pixel, as a fraction
CAMERA = (0.0, 12.0, 22.0)

DOWN = np.array([0.0, -1.0, 0.0], np.float32)
EAST = np.array([1.0, 0.0, 0.0], np.float32)
NORTH = np.array([0.0, 0.0, -1.0], np.float32)  # the Joystick's up, away from the camera
TURN = np.array([0.0, 0.0499792, 0.0, 0.9987503], np.float32)  # 0.1 rad about y per frame
RNG = random.Random(1107)
COMPASS = (
    '<svg xmlns="http://www.w3.org/2000/svg" width="48" height="48" viewBox="-24 -24 48 48">'
    '<circle r="22" fill="none" stroke="#d5d8c8" stroke-width="2"/>'
    '<path d="M0-20L5 0L0 20L-5 0Z" fill="#d5d8c8"/><path d="M0-20L5 0L-5 0Z" fill="#c0392b"/>'
    "</svg>"
)


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
engine.state.setdefault("throws", 0)  # waves the Button asked for, not yet thrown
engine.state.setdefault("g", 6.0)  # gravity, m/s^2, as the Slider last set it
engine.state.setdefault("commits", [])  # the Slider's value at each release
engine.state.setdefault("spins", 0)  # how often the Toggle changed
engine.state.setdefault("wind", [0.0, 0.0])  # the Joystick's last value
engine.state.setdefault("fallen", 0)  # boxes destroyed below the floor
engine.state.setdefault("zoom", 1.0)  # the camera's distance over its first one
engine.state.setdefault("elapsed", 0.0)
engine.state.setdefault("sec", 0.0)  # the clock's seconds in its minute
for name in ("key_down", "key_up"):
    engine.state.setdefault(name, 0)
for name in ("pointer_down", "pointer_up", "pointer_move"):
    engine.state.setdefault(name, [0.0, 0.0, 0])
engine.state.setdefault("resize", [0, 0])

CUBE = mx.geometry.cube(1, 1, 1)
COLORS = [mx.material.phong("#c0392b"), mx.material.phong("#2e86c1"), mx.material.phong("#27ae60")]
GOLD = mx.material.phong("#f2c14e")

WAVE = Button(name="wave", label="Throw a wave")
GRAVITY = Slider(name="gravity", min=0.0, max=12.0, value=6.0, label="gravity")
SPIN = Toggle(name="spin", value=True, label="spin")
WIND = Joystick(name="wind")


def spawn_wave(q: mx.Query[Transform], dt: float):
    throws: int = engine.state["throws"]
    if not engine.input.just_pressed("Space"):
        if throws <= 0:
            return
        engine.state["throws"] = throws - 1
    wave: int = engine.state["waves"]
    a = np.arange(RING) * (2.0 * np.pi / RING)
    r = np.array([RNG.uniform(-JITTER, JITTER) for _ in range(RING)]) + (RADIUS + wave)
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
    g: float = engine.state["g"]
    wind = WIND.value
    q[Velocity].v += (
        DOWN * (g * dt) + EAST * (wind[0] * WIND_SPEED * dt) + NORTH * (wind[1] * WIND_SPEED * dt)
    )
    q[Transform].pos += q[Velocity].v * dt
    if SPIN.value:
        q[Transform].rot += TURN
    below = q[Transform].pos[:, 1] < FLOOR
    k = len(below)  # the entity indices of the boxes below the floor
    if k > 0:
        engine.emit("fell", {"n": k})
    engine.destroy(below)


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
    if WAVE.was_clicked() and prev is not None:  # a new wave clears the pick
        k: int = engine.state["picked_k"]
        engine.set_material(prev, COLORS[k])
        engine.state["picked"] = None
        return
    # A click on the controls is a click too; the GUI says it was its own.
    if not engine.input.just_mouse_pressed(1) or engine.gui.pointer_over_gui:
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
        pk: int = engine.state["picked_k"]
        engine.set_material(prev, COLORS[pk])
    engine.state["picked"] = None
    if len(ids) == 0:
        return
    j = int(np.argmin(dist))
    if float(dist[j]) < math.inf:
        engine.set_material(ids[j], GOLD)
        engine.state["picked"] = ids[j]
        engine.state["picked_k"] = int(float(q[Wave].k.data[:, 0][j]))


def clock(q, dt: float):
    engine.state["sec"] = time.time() % 60.0


def on_wave(p: dict):
    engine.emit("throw", {})


def on_throw(p: dict):
    throws: int = engine.state["throws"]
    engine.state["throws"] = throws + 1


def on_gravity(p: dict):
    engine.state["g"] = p["value"]


def on_gravity_commit(p: dict):
    commits: list[float] = engine.state["commits"]
    commits.append(p["value"])


def on_spin(p: dict):
    spins: int = engine.state["spins"]
    engine.state["spins"] = spins + 1


def on_wind(p: dict):
    v = p["value"]
    engine.state["wind"] = [v[0], v[1]]


def on_fell(p: dict):
    fallen: int = engine.state["fallen"]
    n: int = p["n"]
    engine.state["fallen"] = fallen + n


def on_frame(p: dict):
    engine.state["elapsed"] = p["elapsed"]


def on_key_down(ev):
    n: int = engine.state["key_down"]
    engine.state["key_down"] = n + 1


def on_key_up(ev):
    n: int = engine.state["key_up"]
    engine.state["key_up"] = n + 1


def on_pointer_down(ev):
    engine.state["pointer_down"] = [ev.x, ev.y, ev.pointer_id]


def on_pointer_up(ev):
    engine.state["pointer_up"] = [ev.x, ev.y, ev.pointer_id]


def on_pointer_move(ev):
    engine.state["pointer_move"] = [ev.x, ev.y, ev.pointer_id]


def on_wheel(ev):
    """The camera along its view: the wheel's dy in pixels, the distance kept in 0.5 to 2 times
    the first one."""
    zoom: float = engine.state["zoom"]
    zoom = min(2.0, max(0.5, zoom * (1.0 + ev.dy * ZOOM)))
    engine.state["zoom"] = zoom
    for i in range(3):
        engine.camera.position[i] = CAMERA[i] * zoom


def on_resize(ev):
    engine.state["resize"] = [ev.width, ev.height]


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


def clock_line() -> str:
    s: float = engine.state["sec"]
    return f"clock: {s:.1f} s past the minute"


engine.spawn(
    Mesh(CUBE),
    Material(mx.material.phong("#d5d8c8")),
    Transform(pos=(0.0, -0.05, 0.0), scale=(30.0, 0.1, 30.0)),
)
Velocity.register(engine.store)  # drift's and pick's queries read them before the first wave
Wave.register(engine.store)
for i, x in enumerate(CAMERA):
    engine.camera.position[i] = x
    engine.camera.target[i] = 0.0

engine.system(spawn_wave)
engine.system(drift)
engine.system(pick)
engine.system(clock)
engine.on("gui:button:wave:click")(on_wave)
engine.on("throw")(on_throw)
engine.on("gui:slider:gravity:change")(on_gravity)
engine.on("gui:slider:gravity:commit")(on_gravity_commit)
engine.on("gui:toggle:spin:change")(on_spin)
engine.on("gui:joystick:wind:change")(on_wind)
engine.on("fell")(on_fell)
engine.on("frame")(on_frame)
engine.on("key_down")(on_key_down)
engine.on("key_up")(on_key_up)
engine.on("pointer_down")(on_pointer_down)
engine.on("pointer_up")(on_pointer_up)
engine.on("pointer_move")(on_pointer_move)
engine.on("wheel")(on_wheel)
engine.on("resize")(on_resize)
# Both panels top-left: the desktop's hit test places top-left panels only.
engine.gui.append(
    Panel(
        children=[
            ValueDisplay(boxes_line),
            ValueDisplay(fall_line),
            ValueDisplay(waves_line),
            ValueDisplay(picked_line),
            ValueDisplay(clock_line),
        ],
        anchor="top-left",
        offset=(16, 16),
        style_overrides={"width": 300, "height": 116, "padding": 8, "bg": "#1a1a1aC0"},
    )
)
engine.gui.append(
    Panel(
        children=[
            WAVE,
            GRAVITY,
            SPIN,
            # A row keeps the Joystick at its 96 x 96: a column stretches it across the panel.
            Panel(
                children=[WIND, Image(COMPASS)],
                style_overrides={
                    "direction": "h",
                    "gap": 12,
                    "height": 96,
                    "padding": 0,
                    "bg": "#00000000",
                },
            ),
        ],
        anchor="top-left",
        offset=(16, 144),
        style_overrides={"width": 180, "height": 192, "padding": 8, "gap": 6, "bg": "#1a1a1aC0"},
    )
)

if __name__ == "__main__":
    engine.cli()
