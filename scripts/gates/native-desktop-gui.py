"""Verify native widget routing, composition order, and the two public examples."""

import argparse
import importlib.machinery
import json
import runpy
import sys
from pathlib import Path

import manifoldx as mx
import manifoldx._core_native as native
import numpy as np
from manifoldx.camera import Camera
from manifoldx.components import Layer, Material, Mesh, Transform
from manifoldx.gui import Button, Image, Panel, Slider, Text, ValueDisplay
from manifoldx.post import Grade
from manifoldx.resources import FlatMaterial, plane
from PIL import Image as PillowImage


def _pointer(engine, name, point, button=1):
    engine._window_events([(name, "", (*point, 0, 0), button, [])])
    engine._pump_events()
    return engine.gui.pointer_over_gui


def _assert_pixels(output, image, name):
    PillowImage.fromarray(image).save(output / f"{name}.png")


def check_gui(output, fault):
    engine = mx.Engine("Native GUI gate", width=360, height=240, max_entities=8)
    engine.camera.layers = {0}
    view = Camera()
    view.layers = {1}
    view.set_pose((0, 0, 4), (0, 0, 0))
    engine.spawn(Mesh(plane(2, 2)), Material(FlatMaterial("#ec3522")), Transform(), Layer(layer=1))
    live = engine.render_to_texture(view, (48, 48), background="#3066d3")
    events = []
    engine.on("gui:slider:drag:change")(lambda data: events.append(("change", data["value"])))
    engine.on("gui:slider:drag:commit")(lambda data: events.append(("commit", data["value"])))
    engine.on("gui:button:action:click")(lambda data: events.append(("button", None)))
    engine.on("wheel")(lambda data: events.append(("wheel", data.dy)))
    slider = Slider(name="drag", min=0, max=1, value=0.25)
    text = Text("Native Å", style_overrides={"fg": "#ffffff"})
    getter = ValueDisplay(lambda: f"Value {slider.value:.2f}")
    button = Button(name="action", label="Press")
    hud = Panel([text, slider, getter, button, Image(live, style_overrides={"width": 48, "radius": 24})],
                anchor="top-right", offset=(12, 12),
                style_overrides={"width": 140, "height": 132, "padding": 6, "bg": "#d54422"})
    engine.gui.append(hud)
    engine.set_post(grade=Grade(saturation=0), fxaa=True)
    if fault == "hud-order":
        engine._graphics.fixture_gui_before_post(True)
    start = engine.render_frame()
    _assert_pixels(output, start, "gui-before")
    report = engine.last_frame
    assert report["frame_order"][-3:] == ["composite", "fxaa", "gui"], report
    assert report["gui_quads"][0] >= 4 and report["gui_quads"][1] == 3
    assert report["gui_quads"][2] == report["gui_getter_calls"] == 1
    box = engine._graphics.gui_boxes()[2][0]
    x, y = box[0] + box[2] * 0.25, box[1] + box[3] * 0.5
    if fault == "capture":
        engine._graphics.fixture_gui_omit_capture(True)
    assert _pointer(engine, "pointer_down", (x, y))
    assert _pointer(engine, "pointer_move", (520, 220), 0), "drag outside the panel lost capture"
    assert _pointer(engine, "pointer_up", (520, 220)), "release outside the panel lost capture"
    assert slider.value == 1
    assert events[:3] == [("change", 0.25), ("change", 1.0), ("commit", 1.0)], events
    engine._window_events([("key_down", "ArrowLeft", (0, 0, 0, 0), 0, [])])
    engine._pump_events()
    assert slider.value < 1 and events[-1][0] == "commit"

    view.set_pose((5, 0, 4), (5, 0, 0))
    text.text = "Updated ß"
    updated = engine.render_frame()
    _assert_pixels(output, updated, "gui-after")
    image_box = engine._graphics.gui_boxes()[5][0]
    image_x = round(image_box[0] + image_box[2] * 0.5)
    image_y = round(image_box[1] + image_box[3] * 0.5)
    assert start[image_y, image_x, 0] > start[image_y, image_x, 2] + 50
    assert updated[image_y, image_x, 2] > updated[image_y, image_x, 0] + 50
    assert np.count_nonzero(np.any(start != updated, axis=2)) > 340

    engine.gui.scale = 1.25
    engine._desktop_begin_frame((400, 260), 1.5, engine.elapsed, [], True)
    engine._world._running = False
    resized = engine.render_frame()
    _assert_pixels(output, resized, "gui-resized")
    expanded = engine._graphics.gui_boxes()[2][0]
    assert engine._gui_runtime.scale == 1.875 and box != expanded
    sx, sy = (expanded[0] + expanded[2] * 0.5) * 1.875, (expanded[1] + expanded[3] * 0.5) * 1.875
    engine._window_events([("wheel", "", (sx, sy, 0, 100), 0, [])])
    engine._pump_events()
    assert engine.gui.pointer_over_gui and engine.input.wheel_delta == (0, 100)
    assert events[-1] == ("wheel", 100)
    assert _pointer(engine, "pointer_down", (sx, sy))
    engine.gui.remove(hud)
    engine.render_frame()
    assert engine._graphics.gui_focus() is None, "focus retained after removing the panel"
    assert not _pointer(engine, "pointer_up", (sx, sy)), "capture retained after removing the panel"
    engine.close()
    return {"passes": report["frame_order"], "boxes": [box, expanded], "events": events}


def check_screens(output, examples):
    namespace = runpy.run_path(str(examples / "screens.py"))
    engine = namespace["engine"]
    images = []
    states = []
    for label in ("a", "b", "a-return"):
        if images:
            engine._window_events([("key_down", "Space", (0, 0, 0, 0), 0, [])])
        image = engine.render_frame()
        _assert_pixels(output, image, f"screen-{label}")
        images.append(image)
        assert engine.world.query("Transform").read().numpy().shape == (1, 10)
        systems = [callback.__name__ for callback in engine._application_systems]
        assert systems == ([] if label == "b" else ["spin"]), systems
        states.append({"screen": label, "systems": systems, "text": engine.gui[0].children[0].text})
        engine._window_events([("key_up", "Space", (0, 0, 0, 0), 0, [])])
        engine.render_frame()
    assert states[0]["text"] == states[2]["text"] and states[0]["text"] != states[1]["text"]
    assert np.count_nonzero(np.any(images[0] != images[1], axis=2)) > 900
    engine.close()
    assert not engine._tasks
    return states


def check_portrait(output, examples):
    namespace = runpy.run_path(str(examples / "render_to_texture.py"))
    engine = namespace["engine"]
    images = [engine.render_frame(dt=0.1) for _ in range(3)]
    for i, image in enumerate(images):
        _assert_pixels(output, image, f"portrait-{i}")
    report = engine.last_frame
    assert report["frame_order"][0] == "view" and report["frame_order"][-1] == "gui", report
    assert report["gui_quads"][1:] == [1, 2], report["gui_quads"]
    center = images[0][470:520, 95:145].astype(np.int16)
    assert int((center[:, :, 0] > center[:, :, 2] + 30).sum()) > 100
    assert np.count_nonzero(np.any(images[0][390:570, 30:210] != images[-1][390:570, 30:210], axis=2)) > 200
    engine.close()
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("examples", type=Path)
    parser.add_argument("--fault", choices=("hud-order", "capture"))
    args = parser.parse_args()
    assert sys.flags.isolated and Path(mx.__file__).resolve().is_relative_to(Path(sys.prefix))
    assert isinstance(native.__loader__, importlib.machinery.ExtensionFileLoader)
    args.output.mkdir(parents=True, exist_ok=True)
    result = {"gui": check_gui(args.output, args.fault),
              "screens": check_screens(args.output, args.examples),
              "portrait": check_portrait(args.output, args.examples)}
    (args.output / "gui.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
