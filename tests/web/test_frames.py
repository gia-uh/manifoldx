"""The frame form's optional fields (up, released, pos, wheel, mods) and the engine surface's
engine.gui.pointer_over_gui and engine.camera.fov: a system that reads all of them replays in
Node as in Python, and a script without the optional fields still replays."""

import json
import sys
from pathlib import Path

import pytest

from manifoldx.web.export import export_web
from manifoldx.web.replay import compare, replay_node, replay_python

APP = """
import manifoldx as mx

engine = mx.Engine("frames", width=800, height=500)
engine.camera.fov = 50


def tick(query, dt):
    inp = engine.input
    engine.state["r"] = [
        inp.just_released("w"),
        inp.just_mouse_released(1),
        list(inp.mouse_pos),
        list(inp.wheel_delta),
        list(inp.modifiers),
        engine.gui.pointer_over_gui,
        engine.camera.fov,
    ]


engine.system(tick)

if __name__ == "__main__":
    engine.run()
"""

FRAMES = [
    {"keys": ["w"], "just": ["w"], "d": [0, 0], "buttons": [1], "clicks": [1], "vp": [800, 500]},
    {"keys": [], "just": [], "d": [0, 0], "buttons": [], "clicks": [], "vp": [800, 500],
     "up": ["w"], "released": [1], "pos": [12, 30], "wheel": [0, -3], "mods": ["Shift"]},
]  # fmt: skip

WEB_WALK = Path(__file__).parents[2] / "examples" / "web_walk.py"
WEB_WALK_SCRIPT = Path(__file__).parent / "fixtures" / "web_walk_script.json"


@pytest.fixture
def app(tmp_path):
    folder = tmp_path / "src"
    folder.mkdir()
    yield folder
    for name, m in list(sys.modules.items()):
        if str(getattr(m, "__file__", None) or "").startswith(str(folder)):
            del sys.modules[name]


def test_the_optional_fields_replay_as_in_python(app, tmp_path):
    script, out = app / "app.py", tmp_path / "out"
    script.write_text(APP)
    assert export_web(None, str(script), out, []) == 0
    python = replay_python(None, str(script), FRAMES, [])
    mismatch = compare(python, replay_node(out, FRAMES, []))
    assert mismatch is None, str(mismatch)
    first, second = python[0]["state"]["r"], python[1]["state"]["r"]
    assert first == [False, False, [0.0, 0.0], [0.0, 0.0], [], False, 50.0]
    assert second == [True, True, [12.0, 30.0], [0.0, -3.0], ["Shift"], False, 50.0]


def test_a_script_without_the_optional_fields_still_replays(tmp_path):
    frames = json.loads(WEB_WALK_SCRIPT.read_text())[:120]
    assert not any(k in f for f in frames for k in ("up", "released", "pos", "wheel", "mods"))
    out = tmp_path / "out"
    assert export_web(None, str(WEB_WALK), out, []) == 0
    paths = ["rig.pos", "rig.yaw", "selected"]
    python = replay_python(None, str(WEB_WALK), frames, paths)
    mismatch = compare(python, replay_node(out, frames, paths))
    assert mismatch is None, str(mismatch)
