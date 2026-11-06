"""Walk a yard in first person and pick boxes: the app `manifoldx export --web` replays.

W/A/S/D walk and Tab toggles flying (Space rises, Shift sinks). A left click captures the
pointer for mouse look and Escape releases it; a right drag looks without capturing. A left
click while captured picks the box at the centre of the view: it turns gold and the panel
names it.

Everything that runs per frame (the rig, the ray test, the panel's lines) is plain Python
the web export transpiles; the boxes are built at startup, which stays in Python.

    uv run python examples/web_walk.py
    uv run manifoldx export --web examples/web_walk.py out/
"""

import math

import manifoldx as mx
from manifoldx.components import Material, Mesh, Transform
from manifoldx.gui import Panel, ValueDisplay

engine = mx.Engine("Web walk", width=800, height=500)
engine.state.setdefault("selected", None)  # the picked box's name

EYE = 1.6  # eye height while walking, m
WALK_SPEED = 3.0  # m/s
FLY_SPEED = 6.0
LOOK = 0.004  # radians per pixel of mouse motion
REACH = 40.0  # the farthest a click picks, m

GOLD = mx.material.phong("#f2c14e")


class Box:
    """A named box: its centre, half its side, its own colour and the entity drawing it."""

    def __init__(self, name, center, side, color):
        self.name = name
        self.center = list(center)
        self.half = side / 2
        self.material = mx.material.phong(color)
        self.entity = None

    def hit(self, origin: list, direction: list):
        """How far along the ray the box starts; inf when the ray misses it."""
        near = 0.0
        far = REACH
        for i in range(3):
            lo = self.center[i] - self.half
            hi = self.center[i] + self.half
            if abs(direction[i]) < 1e-9:
                if origin[i] < lo or origin[i] > hi:
                    return math.inf
            else:
                t1 = (lo - origin[i]) / direction[i]
                t2 = (hi - origin[i]) / direction[i]
                near = max(near, min(t1, t2))
                far = min(far, max(t1, t2))
                if near > far:
                    return math.inf
        return near


class Rig:
    """A first-person rig: where the eyes are, where they look, walking or flying."""

    def __init__(self, pos):
        self.pos = list(pos)
        self.yaw = 0.0  # radians; 0 looks toward -z
        self.pitch = 0.0
        self.mode = "walk"

    def forward(self) -> list:
        c = math.cos(self.pitch)
        return [math.sin(self.yaw) * c, math.sin(self.pitch), -math.cos(self.yaw) * c]

    def look(self, dx: float, dy: float):
        self.yaw = self.yaw + dx * LOOK
        self.pitch = max(-1.4, min(1.4, self.pitch - dy * LOOK))

    def move(self, ahead: float, side: float, rise: float, dt: float):
        fly = self.mode == "fly"
        step = (FLY_SPEED if fly else WALK_SPEED) * dt
        f = self.forward() if fly else [math.sin(self.yaw), 0.0, -math.cos(self.yaw)]
        right = [math.cos(self.yaw), 0.0, math.sin(self.yaw)]
        for i in range(3):
            self.pos[i] = self.pos[i] + (f[i] * ahead + right[i] * side) * step
        if fly:
            self.pos[1] = max(0.2, self.pos[1] + rise * step)
        else:
            self.pos[1] = EYE


boxes = [
    Box("red crate", (-3.0, 0.5, -4.0), 1.0, "#c0392b"),
    Box("blue crate", (2.0, 0.5, -5.0), 1.0, "#2e86c1"),
    Box("green tower", (5.0, 1.5, -1.0), 3.0, "#27ae60"),
    Box("purple cube", (-6.0, 1.0, 1.0), 2.0, "#8e44ad"),
    Box("orange block", (0.0, 0.75, -9.0), 1.5, "#e67e22"),
    Box("grey stone", (4.0, 0.25, 4.0), 0.5, "#7f8c8d"),
]
rig = Rig((0.0, EYE, 4.0))


def place_camera(cam, rig: Rig):
    f = rig.forward()
    for i in range(3):
        cam.position[i] = rig.pos[i]
        cam.target[i] = rig.pos[i] + f[i]


def pick(cam):
    """The nearest box at the centre of the view, or None."""
    origin = [float(x) for x in cam.position]  # float32, on the desktop as in the browser
    direction = [float(x) for x in cam.get_forward()]
    best = None
    best_t = REACH
    for box in boxes:
        t = box.hit(origin, direction)
        if t < best_t:
            best = box
            best_t = t
    return best


def select(box):
    """Mark `box` gold and give the last one its colour back; None only clears the mark."""
    current: str | None = engine.state["selected"]
    for b in boxes:
        if b.name == current:
            engine.set_material(b.entity, b.material)
    if box is None:
        engine.state["selected"] = None
    else:
        engine.set_material(box.entity, GOLD)
        engine.state["selected"] = box.name


def axis(plus: str, minus: str) -> float:
    return float(engine.input.is_pressed(plus)) - float(engine.input.is_pressed(minus))


def walk(query, dt: float):
    inp = engine.input
    if inp.just_pressed("Tab"):
        rig.mode = "fly" if rig.mode == "walk" else "walk"
    if inp.just_pressed("Escape"):
        inp.capture_pointer(False)
    if inp.just_mouse_pressed(1):
        if inp.pointer_captured:
            select(pick(engine.camera))
        else:
            inp.capture_pointer(True)
    if inp.pointer_captured or inp.is_mouse_pressed(2):
        rig.look(inp.mouse_delta[0], inp.mouse_delta[1])
    rig.move(axis("w", "s"), axis("d", "a"), axis("Space", "Shift"), dt)
    place_camera(engine.camera, rig)


def picked_line() -> str:
    name: str | None = engine.state["selected"]
    return "picked: nothing" if name is None else f"picked: {name}"


def mode_line() -> str:
    return f"mode: {rig.mode} (Tab toggles)"


def where_line() -> str:
    return f"at {rig.pos[0]:.1f}, {rig.pos[1]:.1f}, {rig.pos[2]:.1f}"


@engine.on("startup")
def build(_):
    cube = mx.geometry.cube(1, 1, 1)
    ground = mx.material.phong("#d5d8c8")
    engine.spawn(Mesh(cube), Material(ground), Transform(pos=(0, -0.05, 0), scale=(30, 0.1, 30)))
    for box in boxes:
        s = box.half * 2
        box.entity = engine.spawn(
            Mesh(cube), Material(box.material), Transform(pos=box.center, scale=(s, s, s))
        )


engine.system(walk)
place_camera(engine.camera, rig)
engine.gui.append(
    Panel(
        children=[ValueDisplay(picked_line), ValueDisplay(mode_line), ValueDisplay(where_line)],
        anchor="top-left",
        offset=(16, 16),
        style_overrides={"width": 240, "height": 76, "padding": 8, "bg": "#1a1a1aC0"},
    )
)

if __name__ == "__main__":
    engine.cli()
