"""The desktop's input events in the replay frame: the page records key, pointer, wheel and
resize events in a frame's "events", and both replays deliver them as the desktop's classes
(KeyEvent, PointerEvent, WheelEvent, ResizeEvent) to the app's handlers."""

from pathlib import Path

import pytest

from manifoldx.input import KeyEvent, PointerEvent, ResizeEvent, WheelEvent
from manifoldx.web.export import export_web
from manifoldx.web.replay import compare, event_of, replay_node, replay_python

EVENTS = Path(__file__).parent / "apps" / "events.py"


def _frame(
    events=(),
    keys=(),
    just=(),
    up=(),
    buttons=(),
    clicks=(),
    released=(),
    pos=(0.0, 0.0),
    wheel=(0.0, 0.0),
    vp=(320, 200),
    mods=(),
):
    return {
        "keys": list(keys),
        "just": list(just),
        "d": [0.0, 0.0],
        "buttons": list(buttons),
        "clicks": list(clicks),
        "vp": list(vp),
        "up": list(up),
        "released": list(released),
        "pos": list(pos),
        "wheel": list(wheel),
        "mods": list(mods),
        "events": [list(e) for e in events],
    }


def _pointer(phase, x, y, button, buttons, dx=0.0, dy=0.0, mods=(), pointer_id=0):
    return [
        f"pointer_{phase}",
        {
            "x": x,
            "y": y,
            "dx": dx,
            "dy": dy,
            "button": button,
            "buttons": list(buttons),
            "modifiers": list(mods),
            "phase": phase,
            "pointer_id": pointer_id,
        },
    ]


def _key(name, key, mods=()):
    return [name, {"key": key, "modifiers": list(mods), "is_down": name == "key_down"}]


def _script():
    """40 frames in the page's form: a key down and up, a click with moves between, a wheel,
    Shift held over a move, a resize at frame 30."""
    frames = [_frame() for _ in range(40)]
    frames[2] = _frame([_key("key_down", "a")], keys=["a"], just=["a"])
    frames[3] = _frame(keys=["a"])
    frames[5] = _frame([_key("key_up", "a")], up=["a"])
    frames[8] = _frame(
        [_pointer("down", 40.0, 50.0, 1, [1])], buttons=[1], clicks=[1], pos=(40, 50)
    )
    frames[9] = _frame(
        [_pointer("move", 43.5, 48.0, 0, [1], dx=3.5, dy=-2.0)], buttons=[1], pos=(43.5, 48)
    )
    frames[10] = _frame(
        [
            _pointer("move", 45.0, 52.0, 0, [1], dx=1.5, dy=4.0),
            _pointer("move", 46.0, 52.0, 0, [1], dx=1.0),
        ],
        buttons=[1],
        pos=(46, 52),
    )
    frames[12] = _frame([_pointer("up", 46.0, 52.0, 1, [])], released=[1], pos=(46, 52))
    frames[15] = _frame(
        [["wheel", {"dx": 0.0, "dy": 50.0, "x": 46.0, "y": 52.0, "buttons": [], "modifiers": []}]],
        wheel=(0, 50),
        pos=(46, 52),
    )
    frames[20] = _frame(
        [
            _key("key_down", "Shift", ["Shift"]),
            _pointer("move", 60.0, 70.0, 0, [], dx=14.0, dy=18.0, mods=["Shift"]),
        ],
        keys=["Shift"],
        just=["Shift"],
        pos=(60, 70),
        mods=["Shift"],
    )
    frames[21] = _frame([_key("key_up", "Shift")], up=["Shift"], pos=(60, 70))
    frames[24] = _frame(
        [
            _pointer("down", 10.0, 11.0, 1, [1], pointer_id=3),
            _pointer("up", 10.0, 11.0, 1, [], pointer_id=3),
        ],
        clicks=[1],
        released=[1],
        pos=(10, 11),
    )
    for i in range(30, 40):
        frames[i] = _frame(vp=(400, 300))
    frames[30]["events"] = [["resize", {"width": 400, "height": 300, "pixel_ratio": 2.0}]]
    return frames


FRAMES = _script()


@pytest.fixture(scope="module")
def traces(tmp_path_factory):
    out = tmp_path_factory.mktemp("input_events") / "out"
    assert export_web(None, str(EVENTS), out, []) == 0
    return replay_python(None, str(EVENTS), FRAMES, []), replay_node(out, FRAMES, [])


def test_input_events_replay(traces):
    python, node = traces
    assert compare(python, node) is None
    assert python[10]["events"] == ["frame", "pointer_move", "pointer_move"]
    assert python[30]["events"] == ["frame", "resize"]


def test_python_builds_the_desktops_classes():
    assert event_of(
        "pointer_move",
        {
            "x": 1,
            "y": 2,
            "dx": 0.5,
            "dy": 0,
            "button": 0,
            "buttons": [1],
            "modifiers": ["Shift"],
            "phase": "move",
            "pointer_id": 0,
        },
    ) == PointerEvent(1, 2, 0.5, 0, 0, (1,), ("Shift",), "move", 0)
    assert event_of("key_up", {"key": "a", "modifiers": [], "is_down": False}) == KeyEvent(
        "a", (), False
    )
    assert event_of(
        "wheel", {"dx": 0, "dy": 5, "x": 1, "y": 2, "buttons": [], "modifiers": []}
    ) == WheelEvent(0, 5, 1, 2, (), ())
    assert event_of("resize", {"width": 4, "height": 3, "pixel_ratio": 1.0}) == ResizeEvent(
        4, 3, 1.0
    )
    assert event_of("ping", {"n": 1}) == {"n": 1}


def test_a_pointer_handler_reads_the_fields(traces):
    python, node = traces
    for trace in (python, node):
        assert trace[8]["state"]["pointer_down"] == [40, 50, 1, [1], "down"]
        assert trace[8]["state"]["mouse_at"] == [40, 50]
        assert trace[10]["state"]["pointer_move"] == [46, 52, 1, 0, [], 0]
        assert trace[20]["state"]["pointer_move"] == [60, 70, 14, 18, ["Shift"], 0]
        assert trace[24]["state"]["pointer_up"] == [10, 11, 1, [], "up", 3]
        assert trace[15]["state"]["wheel"] == [0, 50, 46, 52, [], []]
        assert trace[21]["state"]["key_up"] == ["Shift", [], False]
        assert trace[30]["state"]["resize"] == [400, 300, 2.0]
