"""Joystick widget — a drag gives a value in the unit disc, y up; release returns to (0, 0).

Bridge contract: as for the Slider, the bridge sets `joystick._layout_box` to the
widget's pixel-space LayoutBox before dispatching the pointer down, and keeps the
pointer captured until it goes up.
"""

from __future__ import annotations

import math
from typing import Any

from manifoldx.gui.layout import LayoutBox
from manifoldx.gui.widgets import Widget


class Joystick(Widget):
    _is_gui_interactive = True
    _gui_captures_pointer = True

    def __init__(
        self,
        *,
        name: str,
        style: str | None = None,
        style_overrides: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(style=style, style_overrides=style_overrides)
        if not isinstance(name, str) or not name:
            raise ValueError("Joystick.name is required and must be a non-empty string")
        self.name = name
        self.value: tuple[float, float] = (0.0, 0.0)
        self._layout_box: LayoutBox | None = None

    def intrinsic_size(self) -> tuple[float, float]:
        return (96.0, 96.0)

    def _update_from(self, ev: Any) -> None:
        # The pointer from the centre over half the shorter side, y up, clipped
        # to the unit circle by sqrt (not hypot, whose last bit differs between libms).
        box = self._layout_box
        if box is None or box.w <= 0:
            return
        r = min(box.w, box.h) * 0.5
        x = (ev.x - (box.x + box.w * 0.5)) / r
        y = -(ev.y - (box.y + box.h * 0.5)) / r
        n = math.sqrt(x * x + y * y)
        if n > 1.0:
            x, y = x / n, y / n
        self.value = (x, y)

    def _on_pointer_down(self, ev: Any, engine: Any) -> None:
        self._update_from(ev)
        engine.emit(f"gui:joystick:{self.name}:change", {"value": self.value})

    def _on_pointer_move(self, ev: Any, engine: Any) -> None:
        self._update_from(ev)
        engine.emit(f"gui:joystick:{self.name}:change", {"value": self.value})

    def _on_pointer_up(self, ev: Any, engine: Any) -> None:
        self.value = (0.0, 0.0)
        engine.emit(f"gui:joystick:{self.name}:change", {"value": self.value})
