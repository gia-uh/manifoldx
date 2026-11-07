"""engine.gui as the browser page draws it: panels, anchors, offsets and resolved styles.

gui_spec(engine) -> {"scale", "design_height", "panels": [node]}, where a node is
{"kind": "Panel", "anchor", "offset": [x, y], "children": [node], ...},
{"kind": "Text", "text", ...} or {"kind": "ValueDisplay", "value": k, "min_width", ...},
k being the ValueDisplay's index among find_roots' getter roots (the same panel walk). Every
node also has "widget", its index in the snapshot's walk of the widgets ({"__widget": n}),
"style", resolved (theme, then class, then overrides), and the two parts the page resolves
again when reached code writes style_overrides: "base" (theme, then class) and "overrides".
Colours stay sRGB hex, as CSS reads them.
"""

from __future__ import annotations

from typing import Any


class GuiUnsupported(Exception):
    """A widget the page cannot draw yet: Button, Slider, Toggle, Image."""


def gui_spec(engine) -> dict[str, Any]:
    gui = engine.gui
    counter, widgets = [0], [0]
    panels = [_node(p, (f"panel {i}",), counter, widgets) for i, p in enumerate(gui._panels)]
    return {"scale": float(gui.scale), "design_height": gui.design_height, "panels": panels}


def _node(widget, where: tuple[str, ...], counter: list[int], widgets: list[int]):
    from manifoldx.gui import Panel, Text, ValueDisplay, style

    common = {
        "widget": widgets[0],
        "style": widget.effective_style(),
        "base": style.resolve(widget.style, None),
        "overrides": dict(widget.style_overrides),
    }
    widgets[0] += 1
    if isinstance(widget, Panel):
        return {
            "kind": "Panel",
            "anchor": widget.anchor,
            "offset": list(widget.offset),
            **common,
            "children": [
                _node(c, (*where, f"child {i}"), counter, widgets)
                for i, c in enumerate(widget.children)
            ],
        }
    if isinstance(widget, ValueDisplay):
        counter[0] += 1
        return {
            "kind": "ValueDisplay",
            "value": counter[0] - 1,
            "min_width": widget._min_width,
            **common,
        }
    if isinstance(widget, Text):
        return {"kind": "Text", "text": widget.text, **common}
    raise GuiUnsupported(
        f"the GUI widget {type(widget).__name__} is not supported by export --web yet"
        f" ({', '.join(where)})"
    )
