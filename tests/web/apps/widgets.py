"""The five widgets through the desktop's widget code: one top-left panel with a Button, a
Slider, a Toggle, a Joystick and an Image. tick reads them into engine.state each frame and,
on frame 20, writes the Slider's value, the Button's label and the Toggle's value; a handler
per widget event appends [frame, value] (the click [frame]); on_speed and on_pad leave their
payload unannotated, so it types from its event."""

import manifoldx as mx
from manifoldx.gui import Button, Image, Joystick, Panel, Slider, Toggle

SVG = '<svg xmlns="http://www.w3.org/2000/svg" width="16" height="8"><rect width="16" height="8"/></svg>'

engine = mx.Engine("Widgets", width=320, height=200)
GO = Button(name="go", label="Go")
SPEED = Slider(name="speed", min=0, max=4, value=1, label="speed")
LIT = Toggle(name="lit", value=False, label="lit")
PAD = Joystick(name="pad")
ICON = Image(SVG)
engine.gui.append(Panel(children=[GO, SPEED, LIT, PAD, ICON]))

engine.state["frame"] = -1
engine.state["clicked"] = False
engine.state["speed"] = 0.0
engine.state["lit"] = False
engine.state["pad"] = [0.0, 0.0]
engine.state["over"] = False
engine.state["label"] = ""
for name in ("clicks", "speed_changes", "speed_commits", "lit_changes", "pad_changes"):
    engine.state[name] = []


def on_frame(p: dict):
    engine.state["frame"] = p["frame"]


def tick(q, dt: float):
    f: int = engine.state["frame"]
    engine.state["clicked"] = GO.was_clicked()
    engine.state["speed"] = SPEED.value
    engine.state["lit"] = LIT.value
    engine.state["pad"] = PAD.value
    engine.state["over"] = engine.gui.pointer_over_gui
    engine.state["label"] = GO.label
    if f == 20:
        SPEED.value = 3.0
        GO.label = "Again"
        LIT.value = True


def on_click(p: dict):
    f: int = engine.state["frame"]
    clicks: list[list[int]] = engine.state["clicks"]
    clicks.append([f])


def on_speed(p):
    f: int = engine.state["frame"]
    changes: list[list] = engine.state["speed_changes"]
    changes.append([f, p["value"]])


def on_speed_commit(p: dict):
    f: int = engine.state["frame"]
    commits: list[list] = engine.state["speed_commits"]
    commits.append([f, p["value"]])


def on_lit(p: dict):
    f: int = engine.state["frame"]
    changes: list[list] = engine.state["lit_changes"]
    changes.append([f, p["value"]])


def on_pad(p):
    f: int = engine.state["frame"]
    changes: list[list] = engine.state["pad_changes"]
    changes.append([f, p["value"]])


engine.on("frame")(on_frame)
engine.system(tick)
engine.on("gui:button:go:click")(on_click)
engine.on("gui:slider:speed:change")(on_speed)
engine.on("gui:slider:speed:commit")(on_speed_commit)
engine.on("gui:toggle:lit:change")(on_lit)
engine.on("gui:joystick:pad:change")(on_pad)
engine.run()
