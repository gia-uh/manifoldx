"""engine.gui as the browser page draws it: panels, anchors, offsets and resolved styles.

gui_spec(engine) -> {"scale", "design_height", "panels": [node]}, where a node is
{"kind": "Panel", "anchor", "offset": [x, y], "style", "children": [node]},
{"kind": "Text", "text", "style"} or {"kind": "ValueDisplay", "value": k, "min_width", "style"},
k being the ValueDisplay's index among find_roots' getter roots (the same panel walk).
Styles are resolved (theme, then class, then overrides); colours stay sRGB hex, as CSS reads them.
"""

from __future__ import annotations

from typing import Any


class GuiUnsupported(Exception):
    """A widget the page cannot draw yet: Button, Slider, Toggle, Image."""


def gui_spec(engine) -> dict[str, Any]:
    gui = engine.gui
    counter = [0]
    panels = [_node(p, (f"panel {i}",), counter) for i, p in enumerate(gui._panels)]
    return {"scale": float(gui.scale), "design_height": gui.design_height, "panels": panels}


def _node(widget, where: tuple[str, ...], counter: list[int]) -> dict[str, Any]:
    from manifoldx.gui import Panel, Text, ValueDisplay

    style = widget.effective_style()
    if isinstance(widget, Panel):
        return {
            "kind": "Panel",
            "anchor": widget.anchor,
            "offset": list(widget.offset),
            "style": style,
            "children": [
                _node(c, (*where, f"child {i}"), counter) for i, c in enumerate(widget.children)
            ],
        }
    if isinstance(widget, ValueDisplay):
        counter[0] += 1
        return {
            "kind": "ValueDisplay",
            "value": counter[0] - 1,
            "min_width": widget._min_width,
            "style": style,
        }
    if isinstance(widget, Text):
        return {"kind": "Text", "text": widget.text, "style": style}
    raise GuiUnsupported(
        f"the GUI widget {type(widget).__name__} is not supported by export --web yet"
        f" ({', '.join(where)})"
    )
