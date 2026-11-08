"""The plaza with a control panel in its corner."""

import numpy as np
from PIL import Image

import manifoldx as mx
from manifoldx.components import Component, Material, Mesh, Transform
from manifoldx.gui import Button, Joystick, Panel, Slider, Text, Toggle, ValueDisplay
from manifoldx.gui import Image as GuiImage
from manifoldx.gui import style
from manifoldx.gui.layout import LayoutBox, anchored_slot, compute_layout
from manifoldx.input import PointerEvent
from manifoldx.modeling.primitives import cylinder
from manifoldx.resources import DirectionalLight, StandardMaterial, cube, plane

style.set_theme({"font_size": 13, "fg": "#e8e2d6", "gap": 6})
style.define(
    "panel",
    {
        "bg": "#12161ce8",
        "radius": 10,
        "padding": 10,
        "gap": 6,
        "width": 190,
        "border": 1,
        "border_color": "#3a4552",
    },
)
style.define("row", {"bg": "#00000000", "gap": 8, "height": 22, "padding": 0, "direction": "h"})
style.define("title", {"font_size": 16, "fg": "#f2c14e"})
style.define("readout", {"font_size": 12, "fg": "#9fb3c8"})

engine = mx.Engine("plaza", width=480, height=270)

FLOOR = (-0.70710678, 0.0, 0.0, 0.70710678)  # a quarter turn about x
STONE = StandardMaterial(color="#c8b89a", roughness=0.9)
BASIN = StandardMaterial(color="#8a9bb0", roughness=0.6)
WATER = StandardMaterial(color="#7ad9f5", roughness=0.2)
COAT = StandardMaterial(color="#b04a42", roughness=0.7)

engine.spawn(Mesh(plane(40, 40)), Material(STONE), Transform(pos=(0, 0, 0), rot=FLOOR))
engine.spawn(Mesh(cube(4, 1, 4)), Material(BASIN), Transform(pos=(0, 0.5, 0)))
jet = engine.spawn(
    Mesh(cube(0.8, 3.4, 0.8)),
    Material(WATER),
    Transform(pos=(0, 2.7, 0), scale=(0, 0, 0)),
)

HOME = ((16, 11, 16), (0, 0.8, 0))
engine.camera.set_pose(position=HOME[0], target=HOME[1])
engine.set_sun(DirectionalLight(color="#ffffff", intensity=3.0, direction=(-0.4, -1, -0.3)))

CROWD = 240
rng = np.random.default_rng(7)


@engine.component
class Visitor(Component):
    """Marks a person crossing the plaza."""


phase = rng.uniform(0, 2 * np.pi, CROWD)
ring = rng.uniform(3.5, 15.0, CROWD)
speed = rng.uniform(0.05, 0.25, CROWD)


def crowd_positions():
    x = ring * np.cos(phase)
    z = ring * np.sin(phase)
    return np.stack([x, np.full(CROWD, 0.85), z], axis=1).astype(np.float32)


engine.spawn(
    Mesh(cylinder(0.24, 1.7, 10).to_geometry()),
    Material(COAT),
    Transform(pos=crowd_positions()),
    Visitor(),
    n=CROWD,
)

people = [120]
fountain = [False]
walk = [np.zeros(2, dtype=np.float32)]


@engine.system
def walker(query: mx.Query[Transform, Visitor], dt: float):
    global phase
    phase = phase + speed * dt
    query[Transform].pos = crowd_positions()
    scale = np.zeros((CROWD, 3), dtype=np.float32)
    scale[: people[0]] = 1.0
    query[Transform].scale = scale
    jet.transform.scale = (1, 1, 1) if fountain[0] else (0, 0, 0)
    vx, vy = walk[0]
    if vx or vy:
        engine.camera.move_by(engine.camera.get_forward() * float(vy) * 4.0 * dt)
        engine.camera.move_by(engine.camera.get_right() * float(vx) * 4.0 * dt)


DROP = """<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24">
<path d="M12 2C12 2 5 10 5 15a7 7 0 0 0 14 0C19 10 12 2 12 2Z" fill="#7fbfe0"/>
</svg>"""


def frame_ms() -> str:
    split = engine.frame_split
    return f"frame {1000 * (split['update'] + split['acquire'] + split['render']):.1f} ms"


title = Text("Plaza", style="title")
header = Panel(
    children=[GuiImage(DROP, style_overrides={"width": 20, "height": 20}), title],
    style="row",
)
slider = Slider(name="people", min=0, max=CROWD, value=people[0])
toggle = Toggle(name="fountain", value=False, label="Fountain")
button = Button(name="reset", label="Reset the plaza")
readout = ValueDisplay(getter=frame_ms, style="readout", min_width=165)
joystick = Joystick(name="walk", style_overrides={"width": 60})
joystick_row = Panel(children=[joystick], style="row", style_overrides={"height": 60})

panel = Panel(
    children=[header, slider, toggle, button, readout, joystick_row],
    style="panel",
    anchor="top-right",
    offset=(12, 12),
)
engine.gui.append(panel)


@engine.on("gui:slider:people:change")
def on_people(payload):
    people[0] = int(round(payload["value"]))


@engine.on("gui:toggle:fountain:change")
def on_fountain(payload):
    fountain[0] = payload["value"]


@engine.on("gui:button:reset:click")
def on_reset(payload):
    people[0] = 120
    slider.value = 120.0
    fountain[0] = False
    toggle.value = False
    engine.camera.set_pose(position=HOME[0], target=HOME[1])


@engine.on("gui:joystick:walk:change")
def on_walk(payload):
    walk[0] = np.asarray(payload["value"], dtype=np.float32)


def box_of(widget):
    """Where the panel's layout puts one of its children, in UI pixels."""
    spec = panel.build_layout_spec()
    scale = engine.gui.effective_scale(engine.h)
    viewport = LayoutBox(0, 0, engine.w / scale, engine.h / scale)
    slot = anchored_slot(panel.anchor, panel.offset, spec, viewport)
    boxes = compute_layout(spec, viewport=slot)
    for child, child_spec in zip(panel.children, spec["children"]):
        if child is widget:
            return boxes[id(child_spec)]
    raise ValueError("not a child of the panel")


def pointer(phase_name, x, y):
    engine.emit(
        f"pointer_{phase_name}",
        PointerEvent(
            x=x, y=y, dx=0.0, dy=0.0, button=1, buttons=(1,), modifiers=(), phase=phase_name
        ),
    )


def center(widget):
    box = box_of(widget)
    return box.x + box.w / 2, box.y + box.h / 2


# The first frames of a process build pipelines and cost far more than a steady
# one; draw them before the readout has a frame worth reporting.
for _ in range(2):
    engine.render_frame(supersample=1)

# The two clicks the chapter made: the slider to 40% and the fountain on.
slider_box = box_of(slider)
pointer("down", slider_box.x + slider_box.w * 0.4, slider_box.y + slider_box.h / 2)
pointer("up", slider_box.x + slider_box.w * 0.4, slider_box.y + slider_box.h / 2)
pointer("down", *center(toggle))

engine.render_frame(supersample=1)
Image.fromarray(engine.render_frame(supersample=1))
