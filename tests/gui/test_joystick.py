"""Tests for manifoldx.gui.joystick — a value in the unit disc, y up, back to (0, 0) on release."""

import math

from manifoldx.gui import Button, Joystick, Panel, style
from manifoldx.gui.bridge import _GuiBridge
from manifoldx.gui.layout import LayoutBox


def setup_function(_):
    style.reset()


def _fake_pointer(x, y, phase="down", pointer_id=0):
    from manifoldx.input import PointerEvent

    return PointerEvent(
        x=x,
        y=y,
        dx=0.0,
        dy=0.0,
        button=1,
        buttons=(1,),
        modifiers=(),
        phase=phase,
        pointer_id=pointer_id,
    )


def _fake_engine(sink):
    class _E:
        def emit(self, name, payload):
            sink.append((name, payload))

    return _E()


def test_joystick_construction():
    j = Joystick(name="wind")
    assert j.name == "wind"
    assert j.value == (0.0, 0.0)
    assert j._is_gui_interactive is True
    assert j._gui_captures_pointer is True
    assert j.intrinsic_size() == (96.0, 96.0)


def test_the_value_is_the_pointer_from_the_centre_over_the_radius_y_up():
    received = []
    eng = _fake_engine(received)
    j = Joystick(name="wind")
    j._layout_box = LayoutBox(10.0, 20.0, 100.0, 100.0)
    j._on_pointer_down(_fake_pointer(85.0, 45.0), eng)  # 25 right, 25 up of the centre (60, 70)
    assert j.value == (0.5, 0.5)
    assert received[-1] == ("gui:joystick:wind:change", {"value": (0.5, 0.5)})


def test_past_the_rim_the_value_is_clipped_to_the_unit_circle():
    eng = _fake_engine([])
    j = Joystick(name="wind")
    j._layout_box = LayoutBox(10.0, 20.0, 100.0, 100.0)
    j._on_pointer_down(_fake_pointer(160.0, 70.0), eng)
    assert j.value == (1.0, 0.0)
    j._on_pointer_move(_fake_pointer(110.0, 20.0, "move"), eng)
    x = y = 1.0
    n = math.sqrt(x * x + y * y)
    assert j.value == (x / n, y / n)


def test_release_returns_to_the_centre_and_says_so():
    received = []
    eng = _fake_engine(received)
    j = Joystick(name="wind")
    j._layout_box = LayoutBox(0.0, 0.0, 100.0, 100.0)
    j._on_pointer_down(_fake_pointer(60.0, 50.0), eng)
    j._on_pointer_move(_fake_pointer(70.0, 30.0, "move"), eng)
    assert j.value == (0.4, 0.4)
    j._on_pointer_up(_fake_pointer(500.0, 500.0, "up"), eng)
    assert j.value == (0.0, 0.0)
    assert received[-1] == ("gui:joystick:wind:change", {"value": (0.0, 0.0)})


def test_a_non_square_box_uses_the_shorter_half_side():
    eng = _fake_engine([])
    j = Joystick(name="wind")
    j._layout_box = LayoutBox(0.0, 0.0, 200.0, 100.0)
    j._on_pointer_down(_fake_pointer(150.0, 50.0), eng)
    assert j.value == (1.0, 0.0)  # (150 - 100) / 50


def test_no_box_or_no_width_leaves_the_value():
    eng = _fake_engine([])
    j = Joystick(name="wind")
    j._update_from(_fake_pointer(10.0, 10.0))
    assert j.value == (0.0, 0.0)
    j._layout_box = LayoutBox(0.0, 0.0, 0.0, 100.0)
    j._on_pointer_down(_fake_pointer(10.0, 10.0), eng)
    assert j.value == (0.0, 0.0)


class _BusEngine:
    def __init__(self):
        from manifoldx.gui.widgets import _GuiRoot

        self.gui = _GuiRoot()
        self.w, self.h = 400, 300
        self.input = None
        self._handlers = {}
        self.received = []

    def on(self, name):
        def deco(fn):
            self._handlers.setdefault(name, []).append(fn)
            return fn

        return deco

    def emit(self, name, payload=None):
        self.received.append((name, payload))

    def fire(self, name, ev):
        for fn in self._handlers.get(name, []):
            fn(ev)


def test_joystick_and_button_at_once_through_the_bridge():
    eng = _BusEngine()
    j = Joystick(name="wind", style_overrides={"width": 100, "height": 100})
    b = Button(name="b", label="B", style_overrides={"width": 100, "height": 20})
    eng.gui.append(
        Panel(
            children=[j], offset=(0, 0), style_overrides={"width": 100, "height": 100, "padding": 0}
        )
    )
    eng.gui.append(
        Panel(
            children=[b],
            offset=(0, 200),
            style_overrides={"width": 100, "height": 20, "padding": 0},
        )
    )
    _GuiBridge(eng)
    eng.fire("pointer_down", _fake_pointer(50.0, 50.0, "down", 0))
    eng.fire("pointer_move", _fake_pointer(75.0, 50.0, "move", 0))
    assert j.value == (0.5, 0.0)
    eng.received.clear()
    eng.fire("pointer_down", _fake_pointer(50.0, 210.0, "down", 3))
    eng.fire("pointer_up", _fake_pointer(50.0, 210.0, "up", 3))
    assert eng.received == [("gui:button:b:click", {})]
    eng.received.clear()
    eng.fire("pointer_move", _fake_pointer(50.0, 25.0, "move", 0))
    assert eng.received == [("gui:joystick:wind:change", {"value": (0.0, 0.5)})]
    eng.fire("pointer_up", _fake_pointer(50.0, 25.0, "up", 0))
    assert j.value == (0.0, 0.0)
    assert eng.received[-1] == ("gui:joystick:wind:change", {"value": (0.0, 0.0)})
