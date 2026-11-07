"""One event queue, in the desktop's order (Engine._draw_frame): the frame event first, then
what was emitted during the previous frame; an emit during delivery waits for the next frame.
Both replays deliver through it, and the trace's events are the names delivered each frame."""

from pathlib import Path

import pytest

from manifoldx.web.export import export_web
from manifoldx.web.replay import DT, compare, replay_node, replay_python

EVENTS = Path(__file__).parent / "apps" / "events.py"
IDLE = {"keys": [], "just": [], "d": [0.0, 0.0], "buttons": [], "clicks": [], "vp": [320, 200]}
FRAMES = [IDLE] * 12


@pytest.fixture(scope="module")
def python_trace():
    return replay_python(None, str(EVENTS), FRAMES, [])


def test_events_replay(tmp_path, python_trace):
    out = tmp_path / "out"
    assert export_web(None, str(EVENTS), out, []) == 0
    node = replay_node(out, FRAMES, [])
    assert compare(python_trace, node) is None
    assert [e["events"] for e in node] == [e["events"] for e in python_trace]


def test_an_emit_is_delivered_the_next_frame(python_trace):
    assert python_trace[3]["state"]["pings"] == []
    assert python_trace[4]["state"]["pings"] == [[4, 3]]  # emitted in frame 3
    assert python_trace[4]["events"] == ["frame", "ping"]
    assert python_trace[5]["events"] == ["frame", "ping"]
    assert python_trace[6]["events"] == ["frame", "pong"]  # emitted while frame 5 delivered
    assert python_trace[6]["state"]["pongs"] == 1
    assert python_trace[7]["events"] == ["frame"]


def test_the_frame_event_comes_first_with_elapsed(python_trace):
    assert all(e["events"][0] == "frame" for e in python_trace)
    assert python_trace[0]["state"]["elapsed"] == 0.0
    assert python_trace[3]["state"]["elapsed"] == DT + DT + DT  # each later frame adds its dt
    assert python_trace[3]["state"]["dt"] == DT
    assert [e["state"]["frame"] for e in python_trace] == list(range(12))
    assert python_trace[-1]["state"]["seen"] == list(range(12))  # handlers in registration order


def test_emits_delivered_in_the_same_frame_are_a_mismatch(tmp_path, python_trace):
    out = tmp_path / "out"
    assert export_web(None, str(EVENTS), out, []) == 0
    js = out / "manifoldx.js"
    text = js.read_text()
    step = "  step(app, dt);\n  engine.frame += 1;\n"
    assert text.count(step) == 1
    # drain again after the systems: tick's ping reaches on_ping in the frame that emitted it
    drain = "  engine.bus.drain((name, p) => deliver(app, name, p));\n"
    js.write_text(text.replace(step, "  step(app, dt);\n" + drain + "  engine.frame += 1;\n"))
    assert compare(python_trace, replay_node(out, FRAMES, [])) is not None


def test_a_handler_payload_without_a_type_asks_for_one(tmp_path, capsys):
    app = tmp_path / "untyped.py"
    app.write_text(
        "import manifoldx as mx\n"
        "engine = mx.Engine('u', width=64, height=64)\n"
        "engine.state['n'] = 0\n"
        "\n"
        "def on_ping(p):\n"
        "    engine.state['n'] = p['n']\n"
        "\n"
        "engine.on('ping')(on_ping)\n"
        "engine.run()\n"
    )
    assert export_web(None, str(app), tmp_path / "out", []) == 1
    assert not (tmp_path / "out").exists()
    assert (
        'untyped.py:5: the payload of on("ping") on_ping, with no type '
        "(annotate it: def on_ping(p: dict)), is outside the subset"
    ) in capsys.readouterr().err
