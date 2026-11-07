"""Slider widget — horizontal drag updates value, emits change + commit.

Bridge contract: the bridge must set `slider._layout_box` to the slider's
current pixel-space LayoutBox BEFORE dispatching pointer events. The
bridge does this automatically via the (widget, box) tuple from hit_test.
"""

from __future__ import annotations

from typing import Any

from manifoldx.gui.layout import LayoutBox
from manifoldx.gui.widgets import Widget

# An arrow key on a focused slider moves it by this fraction of its range.
SLIDER_STEPS = 20


class Slider(Widget):
    _is_gui_interactive = True
    _gui_captures_pointer = True

    def __init__(
        self,
        *,
        name: str,
        min: float,
        max: float,
        value: float,
        label: str | None = None,
        style: str | None = None,
        style_overrides: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(style=style, style_overrides=style_overrides)
        if not isinstance(name, str) or not name:
            raise ValueError("Slider.name is required and must be a non-empty string")
        if max <= min:
            raise ValueError(f"Slider max ({max}) must be > min ({min})")
        self.name = name
        self.min = float(min)
        self.max = float(max)
        self.value = float(value)
        self.label = label
        self._layout_box: LayoutBox | None = None

    def intrinsic_size(self) -> tuple[float, float]:
        return (160.0, 16.0)

    def _update_from_x(self, x: float) -> None:
        if self._layout_box is None:
            return
        box = self._layout_box
        if box.w <= 0:
            return
        t = (x - box.x) / box.w
        t = max(0.0, min(1.0, t))
        self.value = self.min + t * (self.max - self.min)

    def _on_pointer_down(self, ev: Any, engine: Any) -> None:
        self._update_from_x(ev.x)
        engine.emit(f"gui:slider:{self.name}:change", {"value": self.value})

    def _on_pointer_move(self, ev: Any, engine: Any) -> None:
        self._update_from_x(ev.x)
        engine.emit(f"gui:slider:{self.name}:change", {"value": self.value})

    def _on_pointer_up(self, ev: Any, engine: Any) -> None:
        self._update_from_x(ev.x)
        engine.emit(f"gui:slider:{self.name}:commit", {"value": self.value})

    def _on_step(self, sign: int, engine: Any) -> None:
        """Step by a twentieth of the range (an arrow key on the focused slider)."""
        step = sign * (self.max - self.min) / SLIDER_STEPS
        self.value = max(self.min, min(self.max, self.value + step))
        engine.emit(f"gui:slider:{self.name}:change", {"value": self.value})
        engine.emit(f"gui:slider:{self.name}:commit", {"value": self.value})
