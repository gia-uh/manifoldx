"""Topmost-interactive-widget lookup at a screen point.

A widget participates in hit testing only if `_is_gui_interactive == True`.
The lookup walks `panels` in REVERSE order (last appended = topmost), then
recurses into each panel's children depth-first, returning the deepest
interactive widget whose layout box contains `(x, y)`, or None.

`viewport` is the UI-pixel screen rect (LayoutBox(0, 0, w / scale, h / scale)). Root
panels' slots come from anchor + offset + explicit width/height; nested
widgets get their boxes from compute_layout.
"""

from __future__ import annotations

from typing import Iterable

from manifoldx.gui.layout import LayoutBox, anchored_slot, compute_layout
from manifoldx.gui.widgets import Panel, Widget

HitResult = tuple[Widget, LayoutBox]


def hit_test(
    panels: Iterable[Panel],
    x: float,
    y: float,
    viewport: LayoutBox,
) -> HitResult | None:
    """Return (widget, box) for the topmost interactive widget under (x, y),
    or None."""
    panels = list(panels)
    for panel in reversed(panels):
        spec = panel.build_layout_spec()
        slot = anchored_slot(panel.anchor, panel.offset, spec, viewport)
        boxes = compute_layout(spec, viewport=slot)
        hit = _walk(panel, spec, boxes, x, y)
        if hit is not None:
            return hit
    return None


def _walk(
    widget: Widget,
    spec: dict,
    boxes: dict,
    x: float,
    y: float,
) -> HitResult | None:
    box = boxes[id(spec)]
    if not _contains(box, x, y):
        return None
    if getattr(widget, "_is_gui_interactive", False):
        if isinstance(widget, Panel):
            for child, child_spec in zip(widget.children, spec["children"]):
                hit = _walk(child, child_spec, boxes, x, y)
                if hit is not None:
                    return hit
        return (widget, box)
    if isinstance(widget, Panel):
        for child, child_spec in zip(widget.children, spec["children"]):
            hit = _walk(child, child_spec, boxes, x, y)
            if hit is not None:
                return hit
    return None


def _contains(box: LayoutBox, x: float, y: float) -> bool:
    return box.x <= x <= box.x + box.w and box.y <= y <= box.y + box.h
