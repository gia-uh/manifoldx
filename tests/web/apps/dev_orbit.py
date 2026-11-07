"""An orbit camera for the dev loop of manifoldx run --web: its yaw, pitch and dist live in
engine.state.cam, so a save keeps the view. A and D turn it, the bump event turns it too, the
box's colour is a constant to edit, and the panel shows SPEED. ("cache", 1) holds what cannot
cross to the browser, as a texture cache would."""

import math

import numpy as np

import manifoldx as mx
from manifoldx.components import Material, Mesh, Transform
from manifoldx.gui import Panel, ValueDisplay

BOX = "#4080c0"
SPEED = 1.0  # radians per second

engine = mx.Engine("Dev orbit", width=320, height=200)
engine.state.setdefault("cam", {"yaw": 0.0, "pitch": 0.3, "dist": 6.0})
engine.state.setdefault(("cache", 1), np.zeros((4, 4)))

engine.spawn(
    Mesh(mx.geometry.cube(1, 1, 1)), Material(mx.material.phong(BOX)), Transform(pos=(0, 0, 0))
)


def place_camera(cam):
    yaw: float = engine.state.cam["yaw"]
    pitch: float = engine.state.cam["pitch"]
    dist: float = engine.state.cam["dist"]
    cam.position[0] = dist * math.cos(pitch) * math.sin(yaw)
    cam.position[1] = dist * math.sin(pitch)
    cam.position[2] = dist * math.cos(pitch) * math.cos(yaw)
    for i in range(3):
        cam.target[i] = 0.0


@engine.system
def orbit(query, dt: float):
    turn = float(engine.input.is_pressed("d")) - float(engine.input.is_pressed("a"))
    yaw: float = engine.state.cam["yaw"]
    engine.state.cam["yaw"] = yaw + turn * SPEED * dt
    place_camera(engine.camera)


@engine.on("bump")
def bump(payload):
    """Turn the camera: {"by": degrees}."""
    by: float = payload["by"]
    yaw: float = engine.state.cam["yaw"]
    engine.state.cam["yaw"] = yaw + math.radians(by)


def speed_line() -> str:
    return f"SPEED = {SPEED:.1f}"


place_camera(engine.camera)
engine.gui.append(
    Panel(
        children=[ValueDisplay(speed_line)],
        anchor="top-left",
        offset=(16, 16),
        style_overrides={"width": 160, "height": 32, "padding": 8, "bg": "#202830"},
    )
)
engine.run()
