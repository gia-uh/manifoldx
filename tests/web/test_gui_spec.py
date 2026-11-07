"""gui_spec: engine.gui as the page's HTML GUI reads it, from state.json's "gui"."""

import json
from pathlib import Path

import pytest

import manifoldx as mx
from manifoldx.gui import Button, Panel, Text, ValueDisplay, style
from manifoldx.web.export import capture_engine, export_web
from manifoldx.web.gui import GuiUnsupported, gui_spec

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

BUTTON = """
import manifoldx as mx
from manifoldx.gui import Button, Panel, Text

engine = mx.Engine("button", width=64, height=64)
engine.gui.append(Panel(children=[Text("hi"), Button(name="go", label="go")]))
engine.run()
"""


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


def test_an_unsupported_widget_names_its_place():
    engine = mx.Engine("gui", width=64, height=64)
    engine.gui.append(Panel(children=[Text("hi"), Button(name="go", label="go")]))
    with pytest.raises(GuiUnsupported) as e:
        gui_spec(engine)
    assert str(e.value) == (
        "the GUI widget Button is not supported by export --web yet (panel 0, child 1)"
    )


def test_export_refuses_an_unsupported_widget_and_writes_nothing(tmp_path, capsys):
    app = tmp_path / "button_app.py"
    app.write_text(BUTTON)
    out = tmp_path / "out"
    assert export_web(None, str(app), out, []) == 1
    assert (
        "the GUI widget Button is not supported by export --web yet (panel 0, child 1)"
        in capsys.readouterr().err
    )
    assert not out.exists()


def test_the_export_writes_the_spec_into_the_state(tmp_path):
    out = tmp_path / "out"
    assert export_web(None, str(WEB_WALK), out, []) == 0
    state = json.loads((out / "state.json").read_text())
    assert state["gui"]["panels"][0]["offset"] == [16, 16]
    assert [c["value"] for c in state["gui"]["panels"][0]["children"]] == [0, 1, 2]
