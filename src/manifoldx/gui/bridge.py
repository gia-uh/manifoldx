"""GUI input bridge — routes pointer events from the bus to widget state.

Subscribes to `pointer_down` / `pointer_move` / `pointer_up` and `key_down` via
`engine.on(...)`. A pointer is located (scaled to UI pixels and hit-tested) by
`_locate`, the one method a replay replaces; a widget that captures the pointer
keeps it, one capture per `pointer_id`, until that pointer goes up or down again.
The last interactive widget a pointer went down on has the focus: the arrow keys
step a focused Slider. `begin_frame` resets `engine.gui.pointer_over_gui`.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

# An arrow key's direction on a focused Slider.
STEPS = {"ArrowLeft": -1, "ArrowDown": -1, "ArrowRight": 1, "ArrowUp": 1}


class _GuiBridge:
    def __init__(self, engine: Any) -> None:
        self._engine = engine
        self._captured: dict[int, Any] = {}
        self._focused: Any | None = None

        @engine.on("pointer_down")
        def _on_down(ev):  # noqa: F841
            self._on_pointer(ev, "down")

        @engine.on("pointer_move")
        def _on_move(ev):  # noqa: F841
            self._on_pointer(ev, "move")

        @engine.on("pointer_up")
        def _on_up(ev):  # noqa: F841
            self._on_pointer(ev, "up")

        @engine.on("key_down")
        def _on_key(ev):  # noqa: F841
            self._on_key(ev)

    def begin_frame(self) -> None:
        """Reset per-frame state. Called by Engine._draw_frame after the
        input bridge's begin_frame."""
        self._engine.gui.pointer_over_gui = False

    def _on_pointer(self, ev: Any, phase: str) -> None:
        """Route a pointer event through hit-test to widget handlers.

        Widgets lay out in UI pixels, so the event is mapped from screen
        pixels by the GUI scale before hit-testing and dispatch.
        """
        # A captured pointer turns the view; its position means nothing to widgets.
        input_state = getattr(self._engine, "input", None)
        if input_state is not None and input_state.pointer_captured:
            return
        gui = self._engine.gui
        ev, result = self._locate(ev)

        # Captured drag has highest priority.
        captured = self._captured.get(ev.pointer_id)
        if captured is not None:
            if phase == "down":
                # New down while captured: end old capture, fall through to hit-test.
                del self._captured[ev.pointer_id]
            else:
                self._dispatch(captured, ev, phase)
                gui.pointer_over_gui = True
                if phase == "up":
                    del self._captured[ev.pointer_id]
                return

        if result is None:
            if phase == "down":
                self._focused = None
            return
        widget, box = result
        gui.pointer_over_gui = True
        # Stash layout box for widgets that need it (e.g. Slider).
        widget._layout_box = box  # noqa: SLF001 — internal contract

        # Only dispatch down/up to widgets without capture — never raw move.
        # Moves without capture only set pointer_over_gui (done above) so that
        # the 3D camera knows the cursor is over the panel, but widgets do not
        # fire change events on hover. Hover opt-in can land in a future plan.
        if phase != "move":
            self._dispatch(widget, ev, phase)
            if phase == "down":
                if getattr(widget, "_gui_captures_pointer", False):
                    self._captured[ev.pointer_id] = widget
                self._focused = widget

    def _locate(self, ev: Any) -> tuple[Any, tuple[Any, Any] | None]:
        """The event in UI pixels, and the interactive (widget, box) under it or None."""
        gui = self._engine.gui
        scale = gui.effective_scale(self._engine.h)
        if scale != 1.0:
            ev = replace(ev, x=ev.x / scale, y=ev.y / scale, dx=ev.dx / scale, dy=ev.dy / scale)

        from manifoldx.gui.hit_test import hit_test
        from manifoldx.gui.layout import LayoutBox

        viewport = LayoutBox(0.0, 0.0, self._engine.w / scale, self._engine.h / scale)
        return ev, hit_test(list(gui), ev.x, ev.y, viewport=viewport)

    def _on_key(self, ev: Any) -> None:
        """Step the focused Slider when the key is an arrow."""
        from manifoldx.gui.slider import Slider

        sign = STEPS.get(ev.key)
        if sign is not None and isinstance(self._focused, Slider):
            self._focused._on_step(sign, self._engine)

    def _dispatch(self, widget: Any, ev: Any, phase: str) -> None:
        """Default dispatch — defer to widget if it implements the hook."""
        hook = getattr(widget, f"_on_pointer_{phase}", None)
        if hook is not None:
            hook(ev, self._engine)
