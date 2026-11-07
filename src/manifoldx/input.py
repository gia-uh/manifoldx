"""Input layer: event dataclasses, polling state, rendercanvas bridge.

See .knowledge/analysis/2026-05-08-input-events-design.md for the design.
"""

from __future__ import annotations

from dataclasses import dataclass


def _key_name(key: str) -> str:
    """One name per physical key: letters lowercase (Shift is in the modifiers),
    and the space bar "Space", which rendercanvas reports as " "."""
    if key == " ":
        return "Space"
    return key.lower() if len(key) == 1 else key


def _glfw_set_capture(window, on: bool) -> None:
    """Hide and lock the cursor of a glfw window, with raw motion where supported."""
    import glfw

    glfw.set_input_mode(window, glfw.CURSOR, glfw.CURSOR_DISABLED if on else glfw.CURSOR_NORMAL)
    if glfw.raw_mouse_motion_supported():
        glfw.set_input_mode(window, glfw.RAW_MOUSE_MOTION, on)


@dataclass(frozen=True, slots=True)
class KeyEvent:
    """A keyboard event delivered on `key_down` or `key_up`."""

    key: str
    modifiers: tuple[str, ...]
    is_down: bool


@dataclass(frozen=True, slots=True)
class PointerEvent:
    """A pointer event delivered on `pointer_down` / `pointer_up` / `pointer_move`.

    `dx` / `dy` is the delta from the previous pointer position the bridge
    saw on a `pointer_move`; on `pointer_down` and `pointer_up` it is always
    0.0. The bridge re-anchors its "last position" tracker to `(x, y)` on
    every pointer event, so the first `pointer_move` after a click produces
    a delta measured from the click point — not from wherever the cursor
    last reported during a previous drag.
    """

    x: float
    y: float
    dx: float
    dy: float
    button: int
    buttons: tuple[int, ...]
    modifiers: tuple[str, ...]
    phase: str  # "down" | "up" | "move"
    pointer_id: int = 0  # which pointer; the desktop has one, 0


@dataclass(frozen=True, slots=True)
class WheelEvent:
    """A scroll-wheel event. `dy` is rendercanvas-scaled (~100 per notch on glfw)."""

    dx: float
    dy: float
    x: float
    y: float
    buttons: tuple[int, ...]
    modifiers: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ResizeEvent:
    """A canvas-resize event. Sizes are in logical pixels."""

    width: int
    height: int
    pixel_ratio: float


class InputState:
    """Per-engine input polling state. Read-only public surface; only the
    bridge mutates via `_record_*` and `_begin_frame`.

    Three flavors of state:

    - "Live" (e.g. `pressed_keys`, `mouse_pos`): mutated immediately on
      every bridge callback. Reads at any point in the frame see the
      freshest value.
    - "Frame-bound" (`just_pressed_keys`, `just_released_keys`,
      `just_pressed_buttons`, `just_released_buttons`): visible only
      between calls to `_begin_frame`. The pending buffer accumulates
      between swaps; `_begin_frame` moves pending → current and clears
      pending.
    - "Accumulators" (`mouse_delta`, `wheel_delta`): summed across all
      bridge callbacks since the last `_begin_frame`; finalized into the
      public `_mouse_delta` / `_wheel_delta` snapshots on swap.
    """

    def __init__(self) -> None:
        # Keyboard
        self._pressed_keys: set[str] = set()
        self._modifiers: tuple[str, ...] = ()
        self._pending_just_pressed_keys: set[str] = set()
        self._pending_just_released_keys: set[str] = set()
        self._just_pressed_keys: frozenset[str] = frozenset()
        self._just_released_keys: frozenset[str] = frozenset()

        # Mouse
        self._mouse_pos: tuple[float, float] = (0.0, 0.0)
        self._accum_mouse_dx: float = 0.0
        self._accum_mouse_dy: float = 0.0
        self._mouse_delta: tuple[float, float] = (0.0, 0.0)
        self._accum_wheel_dx: float = 0.0
        self._accum_wheel_dy: float = 0.0
        self._wheel_delta: tuple[float, float] = (0.0, 0.0)
        self._pressed_buttons: set[int] = set()
        self._pending_just_pressed_buttons: set[int] = set()
        self._pending_just_released_buttons: set[int] = set()
        self._just_pressed_buttons: frozenset[int] = frozenset()
        self._just_released_buttons: frozenset[int] = frozenset()

        # Window
        self._viewport_size: tuple[int, int] = (0, 0)
        self._pointer_captured: bool = False
        self._capture_fn = None  # set by the bridge while a canvas with a window is attached

    # ---- Keyboard public API ----

    def is_pressed(self, key: str) -> bool:
        return _key_name(key) in self._pressed_keys

    def just_pressed(self, key: str) -> bool:
        return _key_name(key) in self._just_pressed_keys

    def just_released(self, key: str) -> bool:
        return _key_name(key) in self._just_released_keys

    @property
    def pressed_keys(self) -> frozenset[str]:
        return frozenset(self._pressed_keys)

    @property
    def modifiers(self) -> tuple[str, ...]:
        return self._modifiers

    # ---- Mouse public API ----

    @property
    def mouse_pos(self) -> tuple[float, float]:
        return self._mouse_pos

    @property
    def mouse_delta(self) -> tuple[float, float]:
        return self._mouse_delta

    @property
    def wheel_delta(self) -> tuple[float, float]:
        return self._wheel_delta

    def is_mouse_pressed(self, button: int) -> bool:
        return button in self._pressed_buttons

    def just_mouse_pressed(self, button: int) -> bool:
        return button in self._just_pressed_buttons

    def just_mouse_released(self, button: int) -> bool:
        return button in self._just_released_buttons

    @property
    def pressed_buttons(self) -> tuple[int, ...]:
        return tuple(sorted(self._pressed_buttons))

    # ---- Window public API ----

    @property
    def viewport_size(self) -> tuple[int, int]:
        return self._viewport_size

    @property
    def pointer_captured(self) -> bool:
        return self._pointer_captured

    def capture_pointer(self, on: bool) -> bool:
        """Hide the cursor and lock it to the window, for mouse look: `mouse_delta`
        keeps counting motion while the cursor cannot reach the window edge, and
        the GUI ignores the pointer. Returns False, and does nothing, when the
        canvas has no window to capture in (offscreen, or a non-glfw backend)."""
        if self._capture_fn is None:
            return False
        self._capture_fn(bool(on))
        self._pointer_captured = bool(on)
        return True

    # ---- Bridge-only mutators ----

    def _record_key_down(self, key: str, modifiers: tuple[str, ...]) -> None:
        key = _key_name(key)
        if key not in self._pressed_keys:
            self._pending_just_pressed_keys.add(key)
        self._pressed_keys.add(key)
        self._modifiers = modifiers

    def _record_key_up(self, key: str, modifiers: tuple[str, ...]) -> None:
        key = _key_name(key)
        if key in self._pressed_keys:
            self._pending_just_released_keys.add(key)
        self._pressed_keys.discard(key)
        self._modifiers = modifiers

    def _record_pointer_down(self, ev: PointerEvent) -> None:
        if ev.button not in self._pressed_buttons:
            self._pending_just_pressed_buttons.add(ev.button)
        self._pressed_buttons.add(ev.button)
        self._mouse_pos = (ev.x, ev.y)
        self._modifiers = ev.modifiers

    def _record_pointer_up(self, ev: PointerEvent) -> None:
        if ev.button in self._pressed_buttons:
            self._pending_just_released_buttons.add(ev.button)
        self._pressed_buttons.discard(ev.button)
        self._mouse_pos = (ev.x, ev.y)
        self._modifiers = ev.modifiers

    def _record_pointer_move(self, ev: PointerEvent) -> None:
        self._accum_mouse_dx += ev.dx
        self._accum_mouse_dy += ev.dy
        self._mouse_pos = (ev.x, ev.y)
        self._modifiers = ev.modifiers

    def _record_wheel(self, ev: WheelEvent) -> None:
        self._accum_wheel_dx += ev.dx
        self._accum_wheel_dy += ev.dy
        self._modifiers = ev.modifiers

    def _record_resize(self, ev: ResizeEvent) -> None:
        self._viewport_size = (ev.width, ev.height)

    def _begin_frame(self) -> None:
        # Swap keyboard pending → current.
        self._just_pressed_keys = frozenset(self._pending_just_pressed_keys)
        self._just_released_keys = frozenset(self._pending_just_released_keys)
        self._pending_just_pressed_keys.clear()
        self._pending_just_released_keys.clear()

        # Swap mouse-button pending → current.
        self._just_pressed_buttons = frozenset(self._pending_just_pressed_buttons)
        self._just_released_buttons = frozenset(self._pending_just_released_buttons)
        self._pending_just_pressed_buttons.clear()
        self._pending_just_released_buttons.clear()

        # Finalize delta accumulators.
        self._mouse_delta = (self._accum_mouse_dx, self._accum_mouse_dy)
        self._wheel_delta = (self._accum_wheel_dx, self._accum_wheel_dy)
        self._accum_mouse_dx = 0.0
        self._accum_mouse_dy = 0.0
        self._accum_wheel_dx = 0.0
        self._accum_wheel_dy = 0.0


_BRIDGED_EVENTS = ("key_down", "key_up", "pointer_down", "pointer_up", "pointer_move", "wheel", "resize")


class _InputBridge:
    """Forward rendercanvas events to the engine event bus and the InputState.

    Attaches as a single `add_event_handler` callback for all input event
    types we care about. Translates the rendercanvas dict into the matching
    typed event dataclass, mutates `InputState` immediately (so live reads
    are fresh), then `engine.emit(event_name, typed_event)` so the typed
    event rides the standard one-frame-delayed bus queue.

    Threading: rendercanvas marshals callbacks to the main thread before
    invoking us, so `_on_event` always runs on the same thread that pumps
    `_draw_frame`. No locking.
    """

    def __init__(self, engine, state: InputState) -> None:
        self._engine = engine
        self._state = state
        self._last_pointer_pos: tuple[float, float] | None = None

    def attach(self, canvas) -> None:
        canvas.add_event_handler(self._on_event, *_BRIDGED_EVENTS)
        window = getattr(canvas, "_window", None)  # glfw only; private in rendercanvas
        if window is not None:
            self._state._capture_fn = lambda on: self._set_capture(window, on)

    def detach(self, canvas) -> None:
        canvas.remove_event_handler(self._on_event, *_BRIDGED_EVENTS)
        if self._state._pointer_captured:
            self._state.capture_pointer(False)
        self._state._capture_fn = None

    def _set_capture(self, window, on: bool) -> None:
        _glfw_set_capture(window, on)
        # glfw moves the virtual cursor when the mode changes; measure the next
        # move from where it lands, not from the last position seen before.
        self._last_pointer_pos = None

    def begin_frame(self) -> None:
        self._state._begin_frame()

    def _on_event(self, rc: dict) -> None:
        et = rc.get("event_type")
        if et == "key_down":
            ev = KeyEvent(key=_key_name(rc["key"]), modifiers=rc["modifiers"], is_down=True)
            self._state._record_key_down(ev.key, ev.modifiers)
            self._engine.emit("key_down", ev)
        elif et == "key_up":
            ev = KeyEvent(key=_key_name(rc["key"]), modifiers=rc["modifiers"], is_down=False)
            self._state._record_key_up(ev.key, ev.modifiers)
            self._engine.emit("key_up", ev)
        elif et in ("pointer_down", "pointer_up", "pointer_move"):
            ev = self._build_pointer_event(rc, et)
            if et == "pointer_down":
                self._state._record_pointer_down(ev)
                self._last_pointer_pos = (ev.x, ev.y)
            elif et == "pointer_up":
                self._state._record_pointer_up(ev)
                self._last_pointer_pos = (ev.x, ev.y)
            else:  # pointer_move
                self._state._record_pointer_move(ev)
                self._last_pointer_pos = (ev.x, ev.y)
            self._engine.emit(et, ev)
        elif et == "wheel":
            ev = WheelEvent(
                dx=rc["dx"], dy=rc["dy"],
                x=rc["x"], y=rc["y"],
                buttons=rc["buttons"], modifiers=rc["modifiers"],
            )
            self._state._record_wheel(ev)
            self._engine.emit("wheel", ev)
        elif et == "resize":
            ev = ResizeEvent(
                width=rc["width"], height=rc["height"],
                pixel_ratio=rc["pixel_ratio"],
            )
            self._state._record_resize(ev)
            on_resize = getattr(self._engine, "_on_canvas_resize", None)
            if on_resize is not None:
                on_resize(ev)
            self._engine.emit("resize", ev)
        # Other rendercanvas events (pointer_enter, pointer_leave,
        # double_click, char, before_draw, animate) are ignored in v1.

    def _build_pointer_event(self, rc: dict, et: str) -> PointerEvent:
        x, y = rc["x"], rc["y"]
        if et == "pointer_move" and self._last_pointer_pos is not None:
            dx = x - self._last_pointer_pos[0]
            dy = y - self._last_pointer_pos[1]
        else:
            dx = 0.0
            dy = 0.0
        phase = {
            "pointer_down": "down",
            "pointer_up": "up",
            "pointer_move": "move",
        }[et]
        return PointerEvent(
            x=x, y=y, dx=dx, dy=dy,
            button=rc.get("button", 0),
            buttons=rc["buttons"],
            modifiers=rc["modifiers"],
            phase=phase,
        )
