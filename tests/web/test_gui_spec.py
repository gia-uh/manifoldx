"""gui_spec: engine.gui as the page's HTML GUI reads it, from state.json's "gui"."""

import json
from pathlib import Path

import pytest

import manifoldx as mx
from manifoldx.gui import Button, Image, Joystick, Panel, Slider, Text, Toggle, ValueDisplay, style
from manifoldx.gui.widgets import Widget
from manifoldx.web.export import capture_engine, export_web
from manifoldx.web.gui import GuiUnsupported, gui_spec, widget_list

WEB_WALK = Path(__file__).parents[2] / "examples" / "web_walk.py"

THEME = {
    "bg": "#222",
    "fg": "#ddd",
    "font_size": 14,
    "padding": 4,
    "gap": 4,
    "border": 0,
    "border_color": "#444",
    "radius": 0,
    "width": None,
    "height": None,
    "flex": None,
    "direction": "v",
}

GAUGE = """
import manifoldx as mx
from manifoldx.gui import Panel, Text
from manifoldx.gui.widgets import Widget


class Gauge(Widget):
    def intrinsic_size(self):
        return (8.0, 8.0)


engine = mx.Engine("gauge", width=64, height=64)
engine.gui.append(Panel(children=[Text("hi"), Gauge()]))
engine.run()
"""

SVG = '<svg xmlns="http://www.w3.org/2000/svg" width="16" height="8"><rect width="16" height="8"/></svg>'


class Gauge(Widget):
    """A widget the page does not have."""

    def intrinsic_size(self):
        return (8.0, 8.0)


@pytest.fixture(autouse=True)
def _theme():
    style.reset()
    yield
    style.reset()


def test_web_walk_panel():
    engine, _ = capture_engine(None, str(WEB_WALK), [])
    vd = {"kind": "ValueDisplay", "min_width": None, "style": THEME, "base": THEME}
    overrides = {"width": 240, "height": 76, "padding": 8, "bg": "#1a1a1aC0"}
    assert gui_spec(engine) == {
        "scale": 1.0,
        "design_height": None,
        "panels": [
            {
                "kind": "Panel",
                "anchor": "top-left",
                "offset": [16, 16],
                "widget": 0,
                "style": {**THEME, **overrides},
                "base": THEME,
                "overrides": overrides,
                "children": [
                    {**vd, "value": v, "widget": v + 1, "overrides": {}} for v in (0, 1, 2)
                ],
            }
        ],
    }


def test_a_style_class_is_resolved_and_values_count_across_panels():
    style.define("hud", {"bg": "#000000", "direction": "h", "gap": 10})
    engine = mx.Engine("gui", width=64, height=64)
    engine.gui.design_height = 720
    engine.gui.append(Panel(children=[ValueDisplay(lambda: "a", min_width=80)]))
    engine.gui.append(
        Panel(
            children=[Panel(children=[Text("x"), ValueDisplay(lambda: "b")], style="hud")],
            anchor="bottom-right",
            offset=(4, 6),
        )
    )
    spec = gui_spec(engine)
    assert spec["design_height"] == 720
    first, second = spec["panels"]
    assert first["children"][0] == {
        "kind": "ValueDisplay",
        "value": 0,
        "min_width": 80,
        "widget": 1,
        "style": THEME,
        "base": THEME,
        "overrides": {},
    }
    assert second["anchor"] == "bottom-right" and second["offset"] == [4, 6]
    nested = second["children"][0]
    hud = {**THEME, "bg": "#000000", "direction": "h", "gap": 10}
    assert nested["style"] == nested["base"] == hud
    assert nested["children"][0] == {
        "kind": "Text",
        "text": "x",
        "widget": 4,
        "style": THEME,
        "base": THEME,
        "overrides": {},
    }
    assert nested["children"][1]["value"] == 1
    json.dumps(spec)  # it goes into state.json as it is


def _five():
    engine = mx.Engine("gui", width=64, height=64)
    widgets = [
        Button(name="go", label="Go"),
        Slider(name="speed", min=0, max=4, value=1, label="speed"),
        Toggle(name="lit", value=True, label="lit"),
        Joystick(name="pad"),
        Image(SVG),
    ]
    engine.gui.append(Panel(children=[Text("hi"), *widgets]))
    return engine, widgets


def test_the_five_widgets_are_nodes_with_their_fields():
    engine, _ = _five()
    children = gui_spec(engine)["panels"][0]["children"]
    picked = [
        {k: c[k] for k in ("kind", "widget", "name", "label", "min", "max", "value", "svg", "size") if k in c}
        for c in children[1:]
    ]  # fmt: skip
    assert picked == [
        {"kind": "Button", "widget": 2, "name": "go", "label": "Go"},
        {"kind": "Slider", "widget": 3, "name": "speed", "label": "speed", "min": 0.0, "max": 4.0, "value": 1.0},
        {"kind": "Toggle", "widget": 4, "name": "lit", "label": "lit", "value": True},
        {"kind": "Joystick", "widget": 5, "name": "pad", "value": [0.0, 0.0]},
        {"kind": "Image", "widget": 6, "svg": SVG, "size": [16.0, 8.0]},
    ]  # fmt: skip
    assert all("style" in c and "overrides" in c for c in children)


def test_widget_list_is_indexed_by_the_nodes_widget():
    engine, widgets = _five()
    nodes = [gui_spec(engine)["panels"][0], *gui_spec(engine)["panels"][0]["children"]]
    listed = widget_list(engine)
    assert len(listed) == len(nodes)
    for node in nodes:
        assert type(listed[node["widget"]]).__name__ == node["kind"]
    assert listed[2:] == widgets


def test_an_unsupported_widget_names_its_place():
    engine = mx.Engine("gui", width=64, height=64)
    engine.gui.append(Panel(children=[Text("hi"), Gauge()]))
    with pytest.raises(GuiUnsupported) as e:
        gui_spec(engine)
    assert str(e.value) == (
        "the GUI widget Gauge is not supported by export --web yet (panel 0, child 1)"
    )


def test_export_refuses_an_unsupported_widget_and_writes_nothing(tmp_path, capsys):
    app = tmp_path / "gauge_app.py"
    app.write_text(GAUGE)
    out = tmp_path / "out"
    assert export_web(None, str(app), out, []) == 1
    assert (
        "the GUI widget Gauge is not supported by export --web yet (panel 0, child 1)"
        in capsys.readouterr().err
    )
    assert not out.exists()


def test_the_export_writes_the_spec_into_the_state(tmp_path):
    out = tmp_path / "out"
    assert export_web(None, str(WEB_WALK), out, []) == 0
    state = json.loads((out / "state.json").read_text())
    assert state["gui"]["panels"][0]["offset"] == [16, 16]
    assert [c["value"] for c in state["gui"]["panels"][0]["children"]] == [0, 1, 2]
