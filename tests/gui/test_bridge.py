"""Tests for manifoldx.gui.bridge — pointer routing skeleton."""

import pytest

from manifoldx.gui import Panel, style
from manifoldx.gui.bridge import _GuiBridge


def setup_function(_):
    style.reset()


def test_bridge_constructs_with_engine_and_subscribes_to_pointer_events():
    received = []

    class _Stub:
        def __init__(self):
            self.gui = _GuiRootStub()
            self._handlers = {}
        def on(self, event_name):
            def decorator(fn):
                self._handlers.setdefault(event_name, []).append(fn)
                return fn
            return decorator
        def emit(self, name, payload):
            received.append((name, payload))

    class _GuiRootStub:
        def __init__(self):
            self.pointer_over_gui = False
            self._panels = []
        def __iter__(self):
            return iter(self._panels)
        def __len__(self):
            return len(self._panels)

    eng = _Stub()
    bridge = _GuiBridge(eng)
    assert "pointer_down" in eng._handlers
    assert "pointer_move" in eng._handlers
    assert "pointer_up" in eng._handlers


def test_bridge_begin_frame_clears_pointer_over_gui_flag():
    class _G:
        pointer_over_gui = True
        def __iter__(self):
            return iter([])
        def __len__(self):
            return 0

    class _Stub:
        def __init__(self):
            self.gui = _G()
            self._handlers = {}
        def on(self, event_name):
            def deco(fn):
                self._handlers.setdefault(event_name, []).append(fn)
                return fn
            return deco
        def emit(self, *a, **k):
            pass
    eng = _Stub()
    bridge = _GuiBridge(eng)
    eng.gui.pointer_over_gui = True
    bridge.begin_frame()
    assert eng.gui.pointer_over_gui is False


def test_pointer_event_over_interactive_widget_sets_flag():
    import pytest
    try:
        from manifoldx.backends import get_offscreen_canvas
        canvas = get_offscreen_canvas(width=128, height=128)
    except Exception as e:
        pytest.skip(f"offscreen canvas unavailable: {e}")
    import manifoldx as mx
    from manifoldx.gui.widgets import Widget

    class _Marker(Widget):
        _is_gui_interactive = True
        def intrinsic_size(self):
            return (10.0, 10.0)

    engine = mx.Engine("test", width=128, height=128)
    engine._init_canvas(canvas)
    engine._running = True

    panel = Panel(
        children=[_Marker()],
        anchor="top-left",
        offset=(0, 0),
        style_overrides={"width": 50, "height": 50, "padding": 0},
    )
    engine.gui.append(panel)

    engine._render_canvas.submit_event({
        "event_type": "pointer_down",
        "x": 10.0, "y": 10.0,
        "button": 1, "buttons": (1,),
        "modifiers": (), "ntouches": 0, "touches": {},
    })
    engine._render_canvas._process_events()
    engine._draw_frame()
    assert engine.gui.pointer_over_gui is True


def test_pointer_move_without_capture_does_not_dispatch_to_widget():
    """A raw pointer_move (no prior down) must not call widget event hooks."""
    try:
        from manifoldx.backends import get_offscreen_canvas
        canvas = get_offscreen_canvas(width=128, height=128)
    except Exception as e:
        pytest.skip(f"offscreen canvas unavailable: {e}")
    import manifoldx as mx
    from manifoldx.gui.widgets import Widget

    dispatched = []

    class _Marker(Widget):
        _is_gui_interactive = True
        _gui_captures_pointer = True

        def intrinsic_size(self):
            return (10.0, 10.0)

        def _on_pointer_move(self, ev, engine):
            dispatched.append("move")

    engine = mx.Engine("test", width=128, height=128)
    engine._init_canvas(canvas)
    engine._running = True

    panel = Panel(
        children=[_Marker()],
        anchor="top-left",
        offset=(0, 0),
        style_overrides={"width": 50, "height": 50, "padding": 0},
    )
    engine.gui.append(panel)

    # Submit move event without any prior down — should not dispatch.
    engine._render_canvas.submit_event({
        "event_type": "pointer_move",
        "x": 10.0, "y": 10.0,
        "button": 0, "buttons": (0,),
        "modifiers": (), "ntouches": 0, "touches": {},
    })
    engine._render_canvas._process_events()
    engine._draw_frame()
    assert dispatched == []


# --- a capture per pointer, the _locate seam, arrow keys (plan 5b, task 1) ---


class _BusEngine:
    """An engine stub with a real GUI root and the bus's on/emit, no canvas."""

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


def _pointer(phase, x, y, pointer_id=0):
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


def _key(key):
    from manifoldx.input import KeyEvent

    return KeyEvent(key=key, modifiers=(), is_down=True)


def _slider_and_button(value=5.0):
    """A slider in a panel at (0, 0) and a button in a panel at (0, 100), each 100x20."""
    from manifoldx.gui import Button, Slider

    eng = _BusEngine()
    slider = Slider(
        name="g", min=0.0, max=10.0, value=value, style_overrides={"width": 100, "height": 20}
    )
    button = Button(name="b", label="B", style_overrides={"width": 100, "height": 20})
    for i, w in enumerate((slider, button)):
        eng.gui.append(
            Panel(
                children=[w],
                anchor="top-left",
                offset=(0, 100 * i),
                style_overrides={"width": 100, "height": 20, "padding": 0},
            )
        )
    bridge = _GuiBridge(eng)
    return eng, bridge, slider, button


def test_a_capture_is_per_pointer():
    eng, bridge, slider, button = _slider_and_button()
    eng.fire("pointer_down", _pointer("down", 50.0, 10.0, 0))
    assert bridge._captured == {0: slider}
    eng.received.clear()
    eng.fire("pointer_down", _pointer("down", 50.0, 110.0, 7))
    eng.fire("pointer_up", _pointer("up", 50.0, 110.0, 7))
    assert [n for n, _ in eng.received] == ["gui:button:b:click"]
    eng.received.clear()
    eng.fire("pointer_move", _pointer("move", 80.0, 10.0, 0))
    assert slider.value == 8.0
    assert eng.received == [("gui:slider:g:change", {"value": 8.0})]
    assert bridge._captured == {0: slider}


def test_a_new_down_of_the_same_pointer_ends_its_capture():
    eng, bridge, slider, button = _slider_and_button()
    eng.fire("pointer_down", _pointer("down", 50.0, 10.0, 0))
    eng.fire("pointer_down", _pointer("down", 50.0, 110.0, 0))
    assert bridge._captured == {}
    eng.received.clear()
    eng.fire("pointer_move", _pointer("move", 80.0, 10.0, 0))
    assert slider.value == 5.0
    assert eng.received == []


def test_locate_is_the_seam():
    from manifoldx.gui.layout import LayoutBox

    eng, bridge, slider, button = _slider_and_button()
    bridge._locate = lambda ev: (ev, (button, LayoutBox(0.0, 0.0, 1.0, 1.0)))
    eng.fire("pointer_down", _pointer("down", 390.0, 290.0))
    assert eng.received == [("gui:button:b:click", {})]
    assert eng.gui.pointer_over_gui is True


def test_arrow_keys_step_the_focused_slider():
    eng, bridge, slider, button = _slider_and_button()
    eng.fire("pointer_down", _pointer("down", 50.0, 10.0))
    eng.fire("pointer_up", _pointer("up", 50.0, 10.0))
    eng.received.clear()
    eng.fire("key_down", _key("ArrowRight"))
    assert slider.value == 5.5
    assert eng.received == [
        ("gui:slider:g:change", {"value": 5.5}),
        ("gui:slider:g:commit", {"value": 5.5}),
    ]
    eng.fire("key_down", _key("ArrowDown"))
    eng.fire("key_down", _key("ArrowDown"))
    assert slider.value == 4.5
    eng.fire("key_down", _key("a"))
    assert slider.value == 4.5
    eng.fire("pointer_down", _pointer("down", 300.0, 250.0))
    eng.received.clear()
    eng.fire("key_down", _key("ArrowRight"))
    assert slider.value == 4.5
    assert eng.received == []


def test_arrow_keys_clamp_at_the_ends():
    eng, bridge, slider, button = _slider_and_button()
    eng.fire("pointer_down", _pointer("down", 98.0, 10.0))
    assert slider.value == 9.8
    eng.fire("key_down", _key("ArrowUp"))
    assert slider.value == 10.0
    eng.fire("pointer_down", _pointer("down", 2.0, 10.0))
    assert slider.value == 0.2
    eng.fire("key_down", _key("ArrowLeft"))
    assert slider.value == 0.0
