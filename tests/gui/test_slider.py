"""Tests for manifoldx.gui.slider — drag capture + change/commit events."""

from manifoldx.gui import Slider, style
from manifoldx.gui.layout import LayoutBox


def setup_function(_):
    style.reset()


def test_slider_construction_validates_args():
    s = Slider(name="G", min=0.1, max=10.0, value=1.0, label="G")
    assert s.name == "G"
    assert s.min == 0.1
    assert s.max == 10.0
    assert s.value == 1.0
    assert s._is_gui_interactive is True
    assert s._gui_captures_pointer is True


def test_slider_pointer_down_sets_value_proportional_to_x_in_box():
    received = []
    eng = _fake_engine(received)
    s = Slider(name="G", min=0.0, max=100.0, value=0.0, label="G")
    s._layout_box = LayoutBox(10.0, 20.0, 200.0, 16.0)
    ev = _fake_pointer(110.0, 28.0)
    s._on_pointer_down(ev, eng)
    assert 49.0 < s.value < 51.0
    assert any(name == "gui:slider:G:change" for name, _ in received)


def test_slider_pointer_move_updates_value_and_emits_change():
    received = []
    eng = _fake_engine(received)
    s = Slider(name="G", min=0.0, max=10.0, value=0.0, label="G")
    s._layout_box = LayoutBox(0.0, 0.0, 100.0, 16.0)
    s._on_pointer_down(_fake_pointer(0.0, 8.0), eng)
    received.clear()
    s._on_pointer_move(_fake_pointer(50.0, 8.0), eng)
    assert s.value == 5.0
    assert received == [("gui:slider:G:change", {"value": 5.0})]


def test_slider_pointer_up_emits_commit():
    received = []
    eng = _fake_engine(received)
    s = Slider(name="G", min=0.0, max=10.0, value=0.0, label="G")
    s._layout_box = LayoutBox(0.0, 0.0, 100.0, 16.0)
    s._on_pointer_down(_fake_pointer(50.0, 8.0), eng)
    received.clear()
    s._on_pointer_up(_fake_pointer(50.0, 8.0), eng)
    assert ("gui:slider:G:commit", {"value": 5.0}) in received


def test_slider_value_clamped_outside_box():
    s = Slider(name="G", min=0.0, max=10.0, value=0.0, label="G")
    s._layout_box = LayoutBox(0.0, 0.0, 100.0, 16.0)
    s._on_pointer_down(_fake_pointer(500.0, 8.0), _fake_engine([]))
    assert s.value == 10.0
    s._on_pointer_move(_fake_pointer(-50.0, 8.0), _fake_engine([]))
    assert s.value == 0.0


def test_on_step_moves_a_twentieth_of_the_range():
    received = []
    s = Slider(name="G", min=2.0, max=4.0, value=3.0)
    s._on_step(1, _fake_engine(received))
    assert s.value == 3 + 1 * (4 - 2) / 20
    assert received == [
        ("gui:slider:G:change", {"value": s.value}),
        ("gui:slider:G:commit", {"value": s.value}),
    ]


def _fake_pointer(x, y):
    from manifoldx.input import PointerEvent
    return PointerEvent(x=x, y=y, dx=0, dy=0, button=1, buttons=(1,),
                        modifiers=(), phase="down")


def _fake_engine(sink):
    class _E:
        def emit(self, name, payload):
            sink.append((name, payload))
    return _E()
