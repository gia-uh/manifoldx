"""Widget classes for the GUI layer.

This module defines the non-interactive widgets — interactive ones
(Button/Slider/Toggle) land in Plan 2.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from manifoldx import _deferred
from manifoldx.gui import style
from manifoldx.gui.style import parse_padding

# Anchors are corners on the viewport; offset is added in pixels.
_VALID_ANCHORS = frozenset({
    "top-left", "top-right", "top-center",
    "bottom-left", "bottom-right", "bottom-center",
    "center-left", "center-right", "center",
})


class Widget(ABC):
    """Base class for all GUI widgets.

    A widget owns its style references (`style` is a class name, `style_overrides`
    is a per-instance dict) and exposes the surface the layout, painter, and
    bridge need.

    A write to a field of a widget that is in the tree defers while its engine
    defers, so the render never reads a field a system is changing. A widget
    built inside a system is not in the tree yet, so its constructor acts at once.
    """

    def __init__(
        self,
        *,
        style: str | None = None,
        style_overrides: dict[str, Any] | None = None,
    ) -> None:
        self.style = style
        self.style_overrides = dict(style_overrides) if style_overrides else {}

    def __setattr__(self, name, value):
        engine = _deferred.current()
        if self._defer_write(engine, name):
            _deferred.defer(engine, object.__setattr__, self, name, value)
            return
        object.__setattr__(self, name, value)

    def _defer_write(self, engine, name) -> bool:
        return (
            engine is not None
            and engine._deferring()
            and not name.startswith("_")
            and self.__dict__.get("_gui_attached", False)
        )

    def effective_style(self) -> dict[str, Any]:
        return _style_resolve(self.style, self.style_overrides)

    @abstractmethod
    def intrinsic_size(self) -> tuple[float, float]:
        """(width, height) in pixels — used when no explicit size or flex."""


def _style_resolve(class_name: str | None, overrides: dict[str, Any]) -> dict[str, Any]:
    # Indirection so tests can monkeypatch resolution if needed.
    return style.resolve(class_name, overrides)


class Panel(Widget):
    """A container of widgets. Lays out children in a stack (vertical or
    horizontal) per its style `direction`, with `padding` around and `gap`
    between children.

    `anchor` and `offset` apply only to root panels (those added directly to
    `engine.gui`). Nested panels are positioned by the parent layout and
    ignore these fields.
    """

    def __init__(
        self,
        children: list[Widget] | None = None,
        *,
        anchor: str = "top-left",
        offset: tuple[int, int] = (0, 0),
        style: str | None = None,
        style_overrides: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(style=style, style_overrides=style_overrides)
        if anchor not in _VALID_ANCHORS:
            raise ValueError(
                f"invalid anchor {anchor!r}; expected one of {sorted(_VALID_ANCHORS)}"
            )
        self._children: list[Widget] = list(children or [])
        self.anchor = anchor
        self.offset = tuple(offset)

    @property
    def children(self) -> list[Widget]:
        """The panel's children; mutated through a queued view while the engine defers."""
        engine = _deferred.current()
        return _deferred.deferring_list(engine, self._children) if engine is not None else self._children

    @children.setter
    def children(self, value) -> None:
        value = list(value)
        engine = _deferred.current()
        if engine is not None and engine._deferring():
            _deferred.defer(engine, object.__setattr__, self, "_children", value)
            return
        self._children = value

    def intrinsic_size(self) -> tuple[float, float]:
        # Containers without explicit size grow to fit their viewport slot;
        # the intrinsic value is a fallback for nested-without-flex cases.
        return (0.0, 0.0)

    def build_layout_spec(self) -> dict[str, Any]:
        s = self.effective_style()
        return {
            "direction": s["direction"],
            "padding": parse_padding(s["padding"]),
            "gap": int(s["gap"]),
            "width": s.get("width"),
            "height": s.get("height"),
            "flex": s.get("flex"),
            "intrinsic": self.intrinsic_size(),
            "children": [_child_spec(c) for c in self.children],
        }


def _child_spec(widget: Widget) -> dict[str, Any]:
    """Convert a widget into a layout spec dict. Defers to Panel for nested
    containers; leaves use the widget's intrinsic_size + style."""
    if isinstance(widget, Panel):
        return widget.build_layout_spec()
    s = widget.effective_style()
    return {
        "direction": "v",  # ignored for leaves
        "padding": (0, 0, 0, 0),
        "gap": 0,
        "width": s.get("width"),
        "height": s.get("height"),
        "flex": s.get("flex"),
        "intrinsic": widget.intrinsic_size(),
        "children": [],
    }


class Text(Widget):
    """A static text label. Intrinsic size is derived from rasterized glyph
    extents at the effective font size; cached on construction (re-measured
    if you mutate `.text`)."""

    def __init__(
        self,
        text: str,
        *,
        style: str | None = None,
        style_overrides: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(style=style, style_overrides=style_overrides)
        self._text = text
        self._cached_intrinsic: tuple[float, float] | None = None

    @property
    def text(self) -> str:
        return self._text

    @text.setter
    def text(self, value: str) -> None:
        if value != self._text:
            self._cached_intrinsic = None
        self._text = value

    def intrinsic_size(self) -> tuple[float, float]:
        if self._cached_intrinsic is None:
            s = self.effective_style()
            self._cached_intrinsic = _measure_text(
                self._text, int(s.get("font_size", 12)), s.get("font")
            )
        return self._cached_intrinsic


class Image(Widget):
    """An SVG drawn at its on-screen pixel size, so it stays crisp at any GUI
    scale or window size. `svg` is a path, SVG markup or bytes, or a
    TextureHandle (such as engine.render_to_texture's), drawn as it is.

    Style: `width` and `height` in UI pixels (one alone keeps the SVG's aspect;
    neither uses the SVG's own size), `tint` (hex, multiplies the art like a
    text's fg), `opacity` (0..1) and `radius` (rounds the corners; half the
    size makes a circle), and `uv` = (u0, v0, u1, v1), the part of the source
    drawn into the box (default the whole; change it every frame to pan a map).
    The art keeps its aspect ratio, centred in its layout box.
    """

    def __init__(
        self,
        svg,
        *,
        style: str | None = None,
        style_overrides: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(style=style, style_overrides=style_overrides)
        self.svg = svg

    @property
    def svg(self):
        return self._svg

    @svg.setter
    def svg(self, source) -> None:
        from manifoldx import svg as _svg
        from manifoldx.textures import TextureHandle

        self._svg = source
        if isinstance(source, TextureHandle):
            self.svg_data = self.svg_digest = None
            self.svg_size = tuple(float(v) for v in source.size)
            return
        self.svg_data = _svg.svg_bytes(source)
        self.svg_digest = _svg.svg_digest(self.svg_data)
        self.svg_size = _svg.svg_size(self.svg_data)

    def intrinsic_size(self) -> tuple[float, float]:
        s = self.effective_style()
        sw, sh = self.svg_size
        w, h = s.get("width"), s.get("height")
        if w is not None and h is not None:
            return (float(w), float(h))
        if w is not None:
            return (float(w), float(w) * sh / sw)
        if h is not None:
            return (float(h) * sw / sh, float(h))
        return (sw, sh)


def _measure_text(text: str, font_size: int, font: str | None = None) -> tuple[float, float]:
    """Pixel extents of `text` rasterized at `font_size`.

    Delegates to the label-atlas PIL rasterizer for accuracy so that layout
    boxes match the actual glyph footprint painted by the GPU pass. With a
    `font` file, the line as the image path draws it.
    """
    if font is not None:
        from manifoldx.gui.font_text import measure

        return measure(text, str(font), font_size)
    from manifoldx.viz.text import LabelTextureAtlas
    w, h, *_ = LabelTextureAtlas.measure_string(text, font_size=font_size)
    return (float(w), float(h))


def _set_attached(widget: Widget, attached: bool) -> None:
    """Mark a widget and its descendants as part of (or out of) the rendered tree."""
    widget.__dict__["_gui_attached"] = attached
    for child in getattr(widget, "children", ()) or ():
        _set_attached(child, attached)


class _GuiRoot:
    """Container exposed as `engine.gui`. List-like over root panels, plus
    a `pointer_over_gui` flag the bridge will toggle (Plan 2).

    Widget sizes, paddings, radii, borders, font sizes and offsets are in UI
    pixels; on screen they are multiplied by `effective_scale(engine.h)`.
    Set `scale` for a fixed factor, or `design_height` to scale the UI with
    the window height (`engine.h / design_height`, which wins over `scale`).

    Colours (`bg`, `border_color`, `fg`, `tint`) are sRGB, like material colours
    and SVG fills: `#888888` shows as #888888. Set `exact_colors = False` for the
    look before 0.11, where a hex was drawn as linear light and showed lighter.
    """

    def __init__(self) -> None:
        self._panels: list[Panel] = []
        self.pointer_over_gui: bool = False
        self.scale: float = 1.0
        self.design_height: float | None = None
        self.exact_colors: bool = True

    def effective_scale(self, screen_height: float) -> float:
        """Screen pixels per UI pixel for a screen `screen_height` pixels tall."""
        if self.design_height:
            return float(screen_height) / float(self.design_height)
        return float(self.scale)

    def __setattr__(self, name, value):
        engine = _deferred.current()
        if (
            engine is not None
            and engine._deferring()
            and not name.startswith("_")
            and name in self.__dict__
        ):
            _deferred.defer(engine, object.__setattr__, self, name, value)
            return
        object.__setattr__(self, name, value)

    @_deferred.deferred_in_pipeline
    def append(self, panel: Panel) -> None:
        self._panels.append(panel)
        _set_attached(panel, True)

    @_deferred.deferred_in_pipeline
    def remove(self, panel: Panel) -> None:
        self._panels.remove(panel)
        _set_attached(panel, False)

    @_deferred.deferred_in_pipeline
    def clear(self) -> None:
        for panel in self._panels:
            _set_attached(panel, False)
        self._panels.clear()

    def __iter__(self):
        return iter(self._panels)

    def __len__(self) -> int:
        return len(self._panels)

    def __getitem__(self, i: int) -> Panel:
        return self._panels[i]
