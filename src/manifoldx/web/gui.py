"""engine.gui as the browser page draws it: panels, anchors, offsets and resolved styles.

gui_spec(engine) -> {"scale", "design_height", "panels": [node]}, where a node is
{"kind": "Panel", "anchor", "offset": [x, y], "children": [node], ...},
{"kind": "Text", "text", ...}, {"kind": "ValueDisplay", "value": k, "min_width", ...}, k being
the ValueDisplay's index among find_roots' getter roots (the same panel walk), or one of the
five interactive and drawn widgets: {"kind": "Button", "name", "label"}, {"kind": "Slider",
"name", "min", "max", "value", "label"}, {"kind": "Toggle", "name", "value", "label"},
{"kind": "Joystick", "name", "value": [x, y]} and {"kind": "Image", "svg": the SVG text or
null (a texture), "size": [w, h]}. Every
node also has "widget", its index in the snapshot's walk of the widgets ({"__widget": n}),
"style", resolved (theme, then class, then overrides), and the two parts the page resolves
again when reached code writes style_overrides: "base" (theme, then class) and "overrides".
Colours stay sRGB hex, as CSS reads them.
"""

from __future__ import annotations

from typing import Any


class GuiUnsupported(Exception):
    """A widget the page does not have: any but Panel, Text, ValueDisplay, Button, Slider,
    Toggle, Joystick and Image."""


def gui_spec(engine) -> dict[str, Any]:
    gui = engine.gui
    counter, widgets = [0], [0]
    panels = [_node(p, (f"panel {i}",), counter, widgets) for i, p in enumerate(gui._panels)]
    return {"scale": float(gui.scale), "design_height": gui.design_height, "panels": panels}


def widget_list(engine) -> list:
    """engine.gui's widgets by their "widget" index: the snapshot's walk of the panels."""
    from manifoldx.gltf.engine_export import _widgets

    return [w for panel in engine.gui._panels for w in _widgets(panel)]


def _node(widget, where: tuple[str, ...], counter: list[int], widgets: list[int]):
    from manifoldx.gui import Button, Image, Joystick, Panel, Slider, Text, Toggle, ValueDisplay
    from manifoldx.gui import style

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
    if isinstance(widget, Button):
        return {"kind": "Button", "name": widget.name, "label": widget.label, **common}
    if isinstance(widget, Slider):
        return {
            "kind": "Slider",
            "name": widget.name,
            "min": widget.min,
            "max": widget.max,
            "value": widget.value,
            "label": widget.label,
            **common,
        }
    if isinstance(widget, Toggle):
        return {
            "kind": "Toggle",
            "name": widget.name,
            "value": widget.value,
            "label": widget.label,
            **common,
        }
    if isinstance(widget, Joystick):
        return {"kind": "Joystick", "name": widget.name, "value": list(widget.value), **common}
    if isinstance(widget, Image):
        svg = None if widget.svg_data is None else widget.svg_data.decode("utf-8")
        return {"kind": "Image", "svg": svg, "size": list(widget.svg_size), **common}
    raise GuiUnsupported(
        f"the GUI widget {type(widget).__name__} is not supported by export --web yet"
        f" ({', '.join(where)})"
    )
