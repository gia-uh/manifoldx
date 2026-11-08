# A control panel

The plaza is running. A crowd crosses it on rings around the fountain, the
camera looks in from the south-east, and the fountain is off. Every number
that decides what you see is a constant in a file: to find out how the plaza
looks with forty people you edit one and restart. The person watching
should have the controls where they can see them: a slider for how many people,
a toggle for the fountain, a readout of how long a frame takes, a button to put
the plaza back the way it was, and, because the same page runs on a phone, a
joystick to walk the camera.

## The plaza as it runs

The scene is the one from the chapters before this one: forty metres of paving,
the basin in the middle, a jet of water hidden until someone asks for it, and a
crowd walking rings around it. Each person is an entity with a `Transform`; a
`Visitor` component with no fields marks which entities are people, so one
system can find them and leave the paving and the basin alone. The system walks
the crowd, hides the people past the current count by giving them a zero scale,
and raises the jet when the fountain is on.

```python {run}
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

people = [120]  # what the panel will change
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


Image.fromarray(engine.render_frame(supersample=1))
```

The three lists are the plaza's state, not the panel's: the slider will report a
number, the toggle a boolean, the joystick two floats, and these are where the
handlers put them. The system reads them every frame, so the scene and the
controls never touch each other directly.

## A panel in the corner

`engine.gui` is the engine's list of root panels. Whatever is in it is drawn
after the scene and covers whatever is under it. The GUI has one base class,
`Widget`: something that knows its style and how big it wants to be, measured
in UI pixels. A `Panel` is a `Widget` that holds other widgets and lays them out
in a stack; a `Text` is a run of glyphs. Those two are enough for the first
piece: a card with the plaza's name on it.

```python {run continue}
panel = Panel(
    children=[Text("Plaza")],
    anchor="top-right",
    offset=(12, 12),
    style_overrides={
        "width": 190,
        "padding": 10,
        "gap": 6,
        "bg": "#12161ce8",
        "radius": 10,
        "border": 1,
        "border_color": "#3a4552",
    },
)
engine.gui.append(panel)

Image.fromarray(engine.render_frame(supersample=1))
```

`anchor` names the corner the panel is pinned to and `offset` is how far its
edges sit from the window's, both in UI pixels. The panel sizes itself from its
content: 190 px wide because we said so, as tall as a line of text plus its
padding. A root panel has one asymmetry worth knowing: with `anchor="top-left"`
and no width or height it fills the whole viewport, measured from the offset,
so an offset pushes its far edges off the screen; every other anchor fits it to
its content. We keep the panel on the right, where either behaviour would do.

A panel stacks its children vertically by default. A title with an icon beside
it is a second, nested panel with `direction: "h"`; its background is fully
transparent so only the icon and the text show over the card. `Image` draws SVG
at the size it is given, which keeps the icon sharp when the window grows. It
shares its name with Pillow's `Image`, which the guide uses for frames, so one
of them has to be imported as `GuiImage`.

```python {run continue}
DROP = """<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24">
<path d="M12 2C12 2 5 10 5 15a7 7 0 0 0 14 0C19 10 12 2 12 2Z" fill="#7fbfe0"/>
</svg>"""

header = Panel(
    children=[GuiImage(DROP, style_overrides={"width": 20, "height": 20}), panel.children[0]],
    style_overrides={
        "direction": "h",
        "gap": 8,
        "height": 22,
        "padding": 0,
        "bg": "#00000000",
    },
)
panel.children[:] = [header]

Image.fromarray(engine.render_frame(supersample=1))
```

The header is 22 px tall and the icon 20, so the two sit on one line. In a
vertical stack a child gets the full width of the panel's content box; a nested
panel is how you get a row instead, and its own children are sized along it.

## One look, three layers

The card carries its whole look in one dict of overrides. A second card would
copy it, and every control inside the card still starts from the defaults, so
the two would have to agree by hand. The GUI resolves a widget's style in three
layers — the theme, a named class, then the widget's own overrides — and the
last one wins. Writing the look once in the theme and in a handful of classes is
what makes a panel read as one piece of furniture rather than a pile of
rectangles.

```python {run continue}
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

title = Text("Plaza", style="title")
header = Panel(
    children=[GuiImage(DROP, style_overrides={"width": 20, "height": 20}), title],
    style="row",
)
panel = Panel(children=[header], style="panel", anchor="top-right", offset=(12, 12))
engine.gui.clear()
engine.gui.append(panel)

Image.fromarray(engine.render_frame(supersample=1))
```

`style.set_theme` merges into the defaults every widget starts from, and
`style.define` registers a class that any widget can name with `style=`. A
widget that names a class still takes `style_overrides` on top, as the icon does
for its size: the class says how all rows look, the override says how big this
icon is. A name that is not defined raises a `KeyError` at paint time — a typo
shows up at once rather than as a silently unstyled widget.

## The controls

Four widgets do the work, and each announces what the visitor did as an event
named after itself: `gui:slider:<name>:change`, `gui:toggle:<name>:change`,
`gui:button:<name>:click`. A `Slider` is a bar you click or drag to a value
between `min` and `max`; dragging also emits a `commit` when the pointer is
released. A `Toggle` is a box and a label that flips on a click. A `Button` is a
label that reports a press. A `ValueDisplay` has no input at all: it calls a
getter once per frame, during the paint, and draws whatever string comes back —
which is how a number that changes every frame gets into the panel without
anything rebuilding.

```python {run continue}
def frame_ms() -> str:
    split = engine.frame_split
    return f"frame {1000 * (split['update'] + split['acquire'] + split['render']):.1f} ms"


slider = Slider(name="people", min=0, max=CROWD, value=people[0])
toggle = Toggle(name="fountain", value=False, label="Fountain")
button = Button(name="reset", label="Reset the plaza")
readout = ValueDisplay(getter=frame_ms, style="readout", min_width=165)

panel.children += [slider, toggle, button, readout]

Image.fromarray(engine.render_frame(supersample=1))
```

`panel.children` is a plain list: append to it and the next layout picks the new
widgets up, no rebuild. The panel is now a column of five rows: the header, the
slider, the toggle, the button and the readout, each 170 px wide because that is
what the card's width and padding leave, and 6 px apart because that is the
theme's `gap`. A widget's own `width` is ignored in a vertical stack; the cross
axis is the panel's to give. The readout asks for 165 px minimum so that a
changing number does not make the card breathe every frame, and shows the three
parts of `engine.frame_split` — the time the engine spent updating, acquiring
and drawing the last frame — added up.

Nothing has changed in the plaza yet. A widget moves itself and says what
happened; what to do about it is a handler.

```python {run continue}
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


print(people[0], fountain[0])
```

Each handler writes the list the `walker` system reads, so the next frame shows
the change. The reset handler is the exception to "the widget owns its value":
it writes `slider.value` and `toggle.value` itself so the controls move back to
match the plaza. That write does not emit a `change` event — a widget reports
what a pointer did, not what its own value became — which is why the handler
sets both the lists and the widgets.

## Reaching the panel

A widget in a window is driven by pointer events, and the GUI hears them on the
same bus as everything else: the input bridge turns a press on the canvas into a
`PointerEvent` and emits it as `pointer_down`, `pointer_move` or `pointer_up`. A
`_GuiBridge` — the engine's, made with it — listens. On every event it divides
the position by the GUI scale, asks the layout where each root panel's slot is,
and walks the panels from the last one added (the topmost) down into their
children. The first widget under the point that accepts input wins; the bridge
sets `engine.gui.pointer_over_gui` so a camera controller that reads it can
ignore a press meant for the panel, and hands the event to that widget. A
`Slider` and a `Joystick` ask for the pointer and keep it until it goes up, so a
drag that leaves the widget still moves it.

The guide has no window, so we post the same events the bridge would have
posted. To put the pointer over a widget we need its box, and the layout that
the paint uses can tell us: `build_layout_spec`, `anchored_slot` and
`compute_layout` return a `LayoutBox` for every widget, in UI pixels.

```python {run continue}
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


slider_box = box_of(slider)
print(slider_box)
pointer("down", slider_box.x + slider_box.w * 0.4, slider_box.y + slider_box.h / 2)
pointer("up", slider_box.x + slider_box.w * 0.4, slider_box.y + slider_box.h / 2)
engine.render_frame(supersample=1)
print("the press:", slider.value, people[0])
frame = engine.render_frame(supersample=1)
print("the frame after:", slider.value, people[0])
Image.fromarray(frame)
```

The box is at x 288, y 50, 170 by 16: the card's right edge 12 px in from 480,
its padding 10 px in, and the slider below the 22 px header and the 6 px gap.
The press lands at 40% of the bar, which sets the slider to 96 of 240. The print
after that frame still says 120 people: the press reached the widget, and the
`change` it emitted is queued for the next frame like any other event, so the
handler that writes `people` runs there. The frame after is the one that shows
the crowd thinned to 96. A click is two frames wide, and the widget moves before
the world does.

## The switch and the button

A toggle is a click with nothing to drag, and the bridge does not capture the
pointer for it: one press flips the value and emits. The button works the same
way, and reports on the press rather than the release, so its `click` arrives
one event earlier than a mouse user might expect.

```python {run continue}
toggle_box = box_of(toggle)
pointer("down", toggle_box.x + toggle_box.w / 2, toggle_box.y + toggle_box.h / 2)
engine.render_frame(supersample=1)
print("the press:", toggle.value, fountain[0])
frame = engine.render_frame(supersample=1)
print("the frame after:", toggle.value, fountain[0])
Image.fromarray(frame)
```

The same two frames: the box ticks on the first, and the `change` it emitted is
what the handler reads on the second, so the jet stands over the basin only in
the frame after that. The widget is honest about what a press did; the world
catches up on the next tick.

The reset button is where the panel's state and the plaza's meet. It puts the
count back to 120, switches the fountain off, sends the camera home, and moves
the slider and the toggle to match so the panel does not lie about the scene.

```python {run continue}
button_box = box_of(button)
pointer("down", button_box.x + button_box.w / 2, button_box.y + button_box.h / 2)
engine.render_frame(supersample=1)
print("the press:", people[0], slider.value, fountain[0], toggle.value)
frame = engine.render_frame(supersample=1)
print("the frame after:", people[0], slider.value, fountain[0], toggle.value)
Image.fromarray(frame)
```

## A joystick for the phone

A `Joystick` is a round pad with a knob. A drag gives a pair of floats in the
unit disc, x to the right and y up, clipped to the circle; letting go puts it
back at `(0, 0)`. On a touch screen that is a thumbstick; with a mouse it is
the same drag. It is the one control here that is not about the plaza's numbers
but about the visitor's place in it: our handler turns the pair into camera
motion, forward for y and sideways for x.

```python {run continue}
@engine.on("gui:joystick:walk:change")
def on_walk(payload):
    walk[0] = np.asarray(payload["value"], dtype=np.float32)


joystick = Joystick(name="walk", style_overrides={"width": 60})
row = Panel(children=[joystick], style="row", style_overrides={"height": 60})
panel.children.append(row)

row_box = box_of(row)
pointer("down", row_box.x + 30, row_box.y + 30)
pointer("move", row_box.x + 55, row_box.y + 15)
engine.render_frame(supersample=1)
print(joystick.value)
Image.fromarray(engine.render_frame(supersample=1))
```

The pointer went 25 px right and 15 px up from the centre of a 60 px pad, so the
value is about `(0.83, 0.5)`: x grows to the right, and y grows upward, against
the screen's own convention, because the camera wants a direction and not a
pixel. While the pointer is held the camera walks forward and right. The row is
a nested panel only to keep the pad square: a joystick in the column would be
stretched to the card's width and painted as a capsule.

On a phone the window is tall and narrow, and a panel drawn for 270 px is a
stamp in the middle of it. `engine.gui.design_height` makes the UI grow with
the window: set it to the height the layout was drawn for, and every size,
padding, radius and font size is multiplied by the window's height over it.
`engine.gui.scale` is a fixed factor instead, for a UI that should not grow.
Neither changes the layout's proportions, so a panel drawn for a wide window
still needs its own width and corner for a narrow one — and the joystick wants
a panel of its own, where a thumb reaches.

```python {run continue}
print("scale at 270:", engine.gui.effective_scale(270))
engine.gui.design_height = 640
print("scale at 844:", round(engine.gui.effective_scale(844), 2))
engine.gui.design_height = None  # back to 1:1 for the frames below
```

At 844 px against that design the card's 190 px become 251, which still leaves
room across a 390 px wide phone, and the 60 px pad becomes 79 px, which is a
thumbstick.

## How the panel is drawn

The widgets say what to draw; a `Painter` collects it. Before the GPU pass, the
render pass lays the panels out and walks them, and each widget adds its
rectangles, its text and its images to the painter as it goes:

- a `RectOp` for every filled rectangle, with its `LayoutBox`, fill, border and
  radius — the card, the rows, the slider's track and its filled part, the
  toggle's box and tick, the button, the joystick's pad and knob;
- a `TextOp` for every line of text, with its box, string, font size and colour;
- an `ImageOp` for every image, with the box it lands in and the raster to
  upload for it.

The painter keeps the three lists and an `order` list that says which op came
when, because the pass draws them back to front: a later panel covers an earlier
one's text as well as its rectangles.

```python {run continue}
from manifoldx.render.passes.gui import layout_gui

painter = layout_gui(engine)
print(len(painter.rect_ops), "rects")
print(len(painter.text_ops), "texts,", len(painter.image_ops), "image")
print(painter.rect_ops[0])
print([op.text for op in painter.text_ops])
```

The first op is the card itself: its box 12 px from the top and right, radius
10, a 1 px border. The four texts are the title, the toggle's label, the
button's label and the readout. The colours in the ops are already linear light:
the pass decodes each hex through sRGB before the shader writes it, so `#12161c`
lands on screen as `#12161c` and not a washed-out grey.

The rectangles are all drawn by one instanced draw call. `RectMaterial` is the
material for it: a shader that takes a unit quad and, per instance, a rectangle
in pixels, its radius and its border, and solves a rounded-box distance field to
antialias the corners. Everything a frame of rectangles needs is packed into a
row of 16 floats per rectangle, in the order the shader's struct expects.

```python {run continue}
from manifoldx.gui.material import RectMaterial

rows = RectMaterial.pack_instances(painter.rect_ops)
print(rows.shape)
print(rows[0])
```

Nine rectangles, sixteen floats each. The first four floats are the card's left,
top, width and height; then its radius and border; then the fill and border
colours with their alpha. One buffer, one bind group, one draw — the panel costs
the same whether it is four rectangles or four hundred.

## The finished program

The whole thing, from the empty engine to the frame above, is one program. It is
the plaza from the earlier chapters with the panel built on top of it, the
handlers that connect the two, and — because this copy has to run where there is
no window — a last pair of events put through the same path a person's finger
would take, so the frame it draws shows the panel in use.

```python {run export=examples/control_panel.py}
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


```

```python {run continue export=examples/control_panel.py}
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


```

```python {run continue export=examples/control_panel.py}
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
```

To look at the plaza in a window instead, end the program with `engine.run()`,
which opens the window and draws a frame every time the screen refreshes, with
the panel live under the pointer.

```python
engine.run()
```

The panel is ten widgets, and none of them knows what a person, a fountain or a
camera is: they report, and the handlers decide. That is what makes the
next chapter's crowd cheap to control too — when the people move onto the GPU,
the slider still only has to say how many.
