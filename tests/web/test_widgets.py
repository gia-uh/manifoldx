"""The five widgets in the runtime, through the desktop's widget code: a pointer event the page
routed to a widget carries "gui": [widget, fx, fy]; the Python replay hands it to the desktop's
GUI bridge in place of its hit test, and the page's port of the bridge and the widget hooks
does the same, so values change in the frame of the pointer and their events reach the
handlers the frame after."""

import json
import shutil
from pathlib import Path

import pytest

from manifoldx.web.export import export_web
from manifoldx.web.replay import compare, replay_node, replay_python

WIDGETS = Path(__file__).parent / "apps" / "widgets.py"
N = 32


def _frame(events=(), buttons=(), clicks=(), released=(), keys=(), just=()):
    return {"keys": list(keys), "just": list(just), "d": [0.0, 0.0], "buttons": list(buttons),
            "clicks": list(clicks), "vp": [320, 200], "up": [], "released": list(released),
            "pos": [0.0, 0.0], "wheel": [0.0, 0.0], "mods": [], "events": list(events)}  # fmt: skip


def _pointer(phase, widget, fx, fy, buttons, pointer_id=0):
    return [f"pointer_{phase}", {
        "x": 10.0, "y": 10.0, "dx": 0.0, "dy": 0.0, "button": 0 if phase == "move" else 1,
        "buttons": list(buttons), "modifiers": [], "phase": phase, "pointer_id": pointer_id,
        "gui": [widget, fx, fy],
    }]  # fmt: skip


def _indices(out: Path) -> dict[str, int]:
    """Each named widget's index, read from state.json's gui as the page reads it."""
    state = json.loads((out / "state.json").read_text())
    found = {}

    def walk(node):
        if "name" in node:
            found[node["name"]] = node["widget"]
        for c in node.get("children", ()):
            walk(c)

    for p in state["gui"]["panels"]:
        walk(p)
    return found


def script(w: dict[str, int]) -> list[dict]:
    """A down and up on go at 2; a down on speed at fx 0.25 at 5, moves to 0.5 at 6 and 1.4
    (outside, still captured) at 8, up at 9; a down and up on lit at 12; a down on pad at
    (0.75, 0.25) at 14 and a move, a tap of pointer 3 on go at 15 while pad is held, pad up
    at 17; a hover over speed with no button at 22; a press on speed at 0.5 at 25 and
    ArrowRight at 27, speed being the last widget pressed."""
    go, speed, lit, pad = w["go"], w["speed"], w["lit"], w["pad"]
    frames = [_frame() for _ in range(N)]
    frames[2] = _frame([_pointer("down", go, 0.5, 0.5, [1]), _pointer("up", go, 0.5, 0.5, [])],
                       clicks=[1], released=[1])  # fmt: skip
    frames[5] = _frame([_pointer("down", speed, 0.25, 0.5, [1])], buttons=[1], clicks=[1])
    frames[6] = _frame([_pointer("move", speed, 0.5, 0.5, [1])], buttons=[1])
    frames[7] = _frame(buttons=[1])
    frames[8] = _frame([_pointer("move", speed, 1.4, 0.5, [1])], buttons=[1])
    frames[9] = _frame([_pointer("up", speed, 1.4, 0.5, [])], released=[1])
    frames[12] = _frame([_pointer("down", lit, 0.5, 0.5, [1]), _pointer("up", lit, 0.5, 0.5, [])],
                        clicks=[1], released=[1])  # fmt: skip
    frames[14] = _frame([_pointer("down", pad, 0.75, 0.25, [1]),
                         _pointer("move", pad, 0.75, 0.25, [1])], buttons=[1], clicks=[1])  # fmt: skip
    frames[15] = _frame([_pointer("down", go, 0.5, 0.5, [1], pointer_id=3),
                         _pointer("up", go, 0.5, 0.5, [1], pointer_id=3)], buttons=[1])  # fmt: skip
    frames[16] = _frame(buttons=[1])
    frames[17] = _frame([_pointer("up", pad, 0.75, 0.25, [])], released=[1])
    frames[22] = _frame([_pointer("move", speed, 0.3, 0.5, [])])
    frames[25] = _frame([_pointer("down", speed, 0.5, 0.5, [1]), _pointer("up", speed, 0.5, 0.5, [])],
                        clicks=[1], released=[1])  # fmt: skip
    frames[27] = _frame([["key_down", {"key": "ArrowRight", "modifiers": [], "is_down": True}]],
                        keys=["ArrowRight"], just=["ArrowRight"])  # fmt: skip
    return frames


@pytest.fixture(scope="module")
def export(tmp_path_factory):
    out = tmp_path_factory.mktemp("widgets") / "out"
    assert export_web(None, str(WIDGETS), out, []) == 0
    return out, script(_indices(out))


@pytest.fixture(scope="module")
def python_trace(export):
    return replay_python(None, str(WIDGETS), export[1], [])


def test_widgets_replay(export, python_trace):
    out, frames = export
    assert compare(python_trace, replay_node(out, frames, [])) is None


def test_the_value_changes_in_the_frame_of_the_pointer_and_the_event_reaches_the_next(python_trace):
    assert python_trace[5]["state"]["speed"] == 1.0  # 0 + 0.25 * 4
    assert python_trace[5]["state"]["speed_changes"] == []
    assert python_trace[6]["state"]["speed_changes"] == [[6, 1.0]]
    assert python_trace[2]["state"]["clicked"] is True  # the latch, read in the same frame
    assert python_trace[3]["state"]["clicked"] is False
    assert python_trace[3]["state"]["clicks"] == [[3]]
    assert python_trace[3]["events"] == ["frame", "gui:button:go:click"]
    assert python_trace[13]["state"]["lit_changes"] == [[13, True]]


def test_a_drag_past_the_end_clamps_and_commits(python_trace):
    assert python_trace[8]["state"]["speed"] == 4.0
    assert python_trace[8]["state"]["over"] is True  # the capture keeps the pointer the GUI's
    assert python_trace[10]["state"]["speed_commits"] == [[10, 4.0]]


def test_the_joystick_keeps_its_pointer_while_the_button_is_tapped(python_trace):
    for f in (14, 15, 16):
        assert python_trace[f]["state"]["pad"] == [0.5, 0.5]
    assert python_trace[16]["state"]["clicks"] == [[3], [16]]
    assert python_trace[17]["state"]["pad"] == [0.0, 0.0]
    assert python_trace[18]["state"]["pad_changes"][-1] == [18, [0.0, 0.0]]


def test_hover_over_a_widget_sets_pointer_over_gui(python_trace):
    assert python_trace[22]["state"]["over"] is True
    assert python_trace[23]["state"]["over"] is False


def test_frame_code_writes_reach_the_widgets(python_trace):
    assert python_trace[21]["state"]["speed"] == 3.0
    assert python_trace[21]["state"]["label"] == "Again"
    changes = python_trace[24]["state"]["speed_changes"]
    assert all(frame <= 10 for frame, _ in changes)  # the write queued no event


def test_the_arrow_key_steps_the_last_slider_pressed(python_trace):
    assert python_trace[25]["state"]["speed"] == 2.0
    assert python_trace[27]["state"]["speed"] == 2.2  # 2 + (4 - 0) / 20
    assert python_trace[28]["state"]["speed_commits"][-1] == [28, 2.2]


def test_a_value_changed_after_its_event_is_a_mismatch(export, python_trace, tmp_path):
    out, frames = export
    broken = tmp_path / "out"
    shutil.copytree(out, broken)
    js = broken / "manifoldx.js"
    text = js.read_text()
    good = (
        "      w.value = sliderValue(w, ev.x);\n"
        '      engine.emit(`gui:slider:${w.name}:change`, new Map([["value", w.value]]));\n'
    )
    assert text.count(good) == 2  # down and move
    bad = (
        '      engine.emit(`gui:slider:${w.name}:change`, new Map([["value", w.value]]));\n'
        "      w.value = sliderValue(w, ev.x);\n"
    )
    js.write_text(text.replace(good, bad))
    assert compare(python_trace, replay_node(broken, frames, [])) is not None
