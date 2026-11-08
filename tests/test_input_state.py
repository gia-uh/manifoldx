"""InputState + event dataclass unit tests (no engine, no canvas)."""
import pytest
from dataclasses import FrozenInstanceError

from manifoldx.input import (
    InputState,
    KeyEvent,
    PointerEvent,
    WheelEvent,
    ResizeEvent,
)


def test_key_event_is_frozen():
    ev = KeyEvent(key="a", modifiers=("Shift",), is_down=True)
    assert ev.key == "a"
    assert ev.modifiers == ("Shift",)
    assert ev.is_down is True
    with pytest.raises(FrozenInstanceError):
        ev.key = "b"


def test_pointer_event_fields():
    ev = PointerEvent(
        x=10.0, y=20.0, dx=1.0, dy=2.0,
        button=1, buttons=(1,), modifiers=(), phase="down",
    )
    assert ev.x == 10.0
    assert ev.dx == 1.0
    assert ev.phase == "down"
    with pytest.raises(FrozenInstanceError):
        ev.x = 99.0


def test_wheel_event_fields():
    ev = WheelEvent(dx=0.0, dy=100.0, x=5.0, y=5.0, buttons=(), modifiers=())
    assert ev.dy == 100.0
    with pytest.raises(FrozenInstanceError):
        ev.dy = 0.0


def test_resize_event_fields():
    ev = ResizeEvent(width=800, height=600, pixel_ratio=2.0)
    assert ev.width == 800
    assert ev.pixel_ratio == 2.0
    with pytest.raises(FrozenInstanceError):
        ev.width = 0



def test_is_pressed_reflects_live_state_without_frame_swap():
    state = InputState()
    assert not state.is_pressed("w")
    state._record_key_down("w", modifiers=())
    # Live: no _begin_frame call yet, but is_pressed still answers True.
    assert state.is_pressed("w")
    state._record_key_up("w", modifiers=())
    assert not state.is_pressed("w")


def test_just_pressed_visible_only_after_frame_swap():
    state = InputState()
    state._record_key_down("w", modifiers=())
    # Before the swap, just_pressed is empty.
    assert not state.just_pressed("w")
    state._begin_frame()
    assert state.just_pressed("w")
    assert state.is_pressed("w")


def test_just_pressed_clears_on_subsequent_frame():
    state = InputState()
    state._record_key_down("w", modifiers=())
    state._begin_frame()
    assert state.just_pressed("w")
    state._begin_frame()
    assert not state.just_pressed("w")
    assert state.is_pressed("w")  # still held


def test_just_released_visible_only_after_frame_swap():
    state = InputState()
    state._record_key_down("w", modifiers=())
    state._begin_frame()
    state._record_key_up("w", modifiers=())
    assert not state.just_released("w")
    state._begin_frame()
    assert state.just_released("w")
    assert not state.is_pressed("w")
    state._begin_frame()
    assert not state.just_released("w")


def test_pressed_keys_returns_immutable_frozenset():
    state = InputState()
    state._record_key_down("a", modifiers=())
    state._record_key_down("b", modifiers=())
    keys = state.pressed_keys
    assert isinstance(keys, frozenset)
    assert keys == frozenset({"a", "b"})


def test_modifiers_property_reflects_latest_event():
    state = InputState()
    state._record_key_down("a", modifiers=("Shift",))
    assert state.modifiers == ("Shift",)
    state._record_key_up("a", modifiers=())
    assert state.modifiers == ()



def _pmove(x: float, y: float, dx: float, dy: float) -> PointerEvent:
    return PointerEvent(
        x=x, y=y, dx=dx, dy=dy,
        button=0, buttons=(), modifiers=(), phase="move",
    )


def _pdown(x: float, y: float, button: int) -> PointerEvent:
    return PointerEvent(
        x=x, y=y, dx=0.0, dy=0.0,
        button=button, buttons=(button,), modifiers=(), phase="down",
    )


def _pup(x: float, y: float, button: int) -> PointerEvent:
    return PointerEvent(
        x=x, y=y, dx=0.0, dy=0.0,
        button=button, buttons=(), modifiers=(), phase="up",
    )


def test_mouse_pos_reflects_latest_move_without_swap():
    state = InputState()
    assert state.mouse_pos == (0.0, 0.0)
    state._record_pointer_move(_pmove(10.0, 20.0, 0.0, 0.0))
    assert state.mouse_pos == (10.0, 20.0)
    state._record_pointer_move(_pmove(15.0, 25.0, 5.0, 5.0))
    assert state.mouse_pos == (15.0, 25.0)


def test_mouse_delta_accumulates_across_events_then_resets():
    state = InputState()
    state._record_pointer_move(_pmove(10.0, 10.0, 2.0, 0.0))
    state._record_pointer_move(_pmove(15.0, 10.0, 5.0, 0.0))
    # Before swap: accumulator is internal; mouse_delta property reads the
    # last finalized value (still (0,0) since no _begin_frame has happened).
    assert state.mouse_delta == (0.0, 0.0)
    state._begin_frame()
    assert state.mouse_delta == (7.0, 0.0)
    # Next frame with no events: delta resets to zero.
    state._begin_frame()
    assert state.mouse_delta == (0.0, 0.0)


def test_wheel_delta_accumulates_then_resets():
    state = InputState()
    state._record_wheel(WheelEvent(dx=0.0, dy=100.0, x=0.0, y=0.0,
                                   buttons=(), modifiers=()))
    state._record_wheel(WheelEvent(dx=0.0, dy=-50.0, x=0.0, y=0.0,
                                   buttons=(), modifiers=()))
    assert state.wheel_delta == (0.0, 0.0)
    state._begin_frame()
    assert state.wheel_delta == (0.0, 50.0)
    state._begin_frame()
    assert state.wheel_delta == (0.0, 0.0)


def test_mouse_button_pressed_lifecycle():
    state = InputState()
    state._record_pointer_down(_pdown(0.0, 0.0, button=1))
    assert state.is_mouse_pressed(1)
    state._begin_frame()
    assert state.just_mouse_pressed(1)
    assert state.is_mouse_pressed(1)
    state._begin_frame()
    assert not state.just_mouse_pressed(1)
    assert state.is_mouse_pressed(1)
    state._record_pointer_up(_pup(0.0, 0.0, button=1))
    state._begin_frame()
    assert state.just_mouse_released(1)
    assert not state.is_mouse_pressed(1)


def test_pressed_buttons_is_sorted_tuple():
    state = InputState()
    state._record_pointer_down(_pdown(0.0, 0.0, button=2))
    state._record_pointer_down(_pdown(0.0, 0.0, button=1))
    assert state.pressed_buttons == (1, 2)


def test_resize_updates_viewport_size_immediately():
    state = InputState()
    assert state.viewport_size == (0, 0)
    state._record_resize(ResizeEvent(width=800, height=600, pixel_ratio=1.0))
    assert state.viewport_size == (800, 600)


def test_releasing_a_letter_after_shift_changed_its_case_releases_it():
    # rendercanvas's glfw backend names a letter upper case while Shift is held,
    # so "s" goes down and "S" comes up.
    state = InputState()
    state._record_key_down("s", modifiers=())
    state._record_key_up("S", modifiers=("Shift",))
    assert not state.is_pressed("s") and not state.is_pressed("S")
    state._begin_frame()
    assert state.just_released("s")
