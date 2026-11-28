"""Widget declarations and Python callbacks for the native desktop GUI."""

from dataclasses import replace
from importlib.resources import files
from pathlib import Path

from manifoldx.gui.bridge import STEPS
from manifoldx.gui.button import Button
from manifoldx.gui.joystick import Joystick
from manifoldx.gui.layout import AXES, LayoutBox
from manifoldx.gui.slider import Slider
from manifoldx.gui.style import parse_color, parse_padding
from manifoldx.gui.toggle import Toggle
from manifoldx.gui.value_display import ValueDisplay
from manifoldx.gui.widgets import Image, Panel, Text
from manifoldx.textures import TextureHandle

_ANCHOR = {"start": -1, "center": 0, "end": 1}
_TRANSPARENT = (0, 0, 0, 0)
_WHITE = (1, 1, 1, 1)


class _NativeGui:
    def __init__(self, engine):
        self._engine = engine
        self._widgets = {}
        self._fonts = set()
        self._images = {}
        self.scale = 1.0
        self.dpi = 1.0
        self.getter_calls = 0
        self.getter_total = 0
        self.last_target = None
        self._hud_panel = None
        self._hud_texts = []

    def sync(self, supersample=1, *, refresh=False):
        engine = self._engine
        scale = engine.gui.effective_scale(engine.h) if engine.gui.design_height else engine.gui.scale * self.dpi
        self.scale = scale
        declarations = []
        widgets = {}
        used_images = set()
        self.getter_calls = 0

        def add(widget, parent):
            style = widget.effective_style()
            anchor = tuple(_ANCHOR[axis] for axis in AXES[widget.anchor]) if parent is None else (0, 0)
            offset = widget.offset if parent is None else (0, 0)
            size = (style.get("width"), style.get("height"))
            kind, label, value, stick, image = 0, None, 0.0, (0.0, 0.0), None
            bg = parse_color(style["bg"]) if "bg" in style else _TRANSPARENT
            fg = parse_color(style["fg"]) if "fg" in style else _WHITE
            border_color = parse_color(style["border_color"]) if "border_color" in style else _TRANSPARENT
            border, radius = float(style.get("border") or 0), float(style.get("radius") or 0)
            if isinstance(widget, ValueDisplay):
                if refresh:
                    widget.refresh()
                    self.getter_calls += 1
                kind, label, value = 1, self._label(widget.text, style), widget._min_width or 0.0
            elif isinstance(widget, Text):
                kind, label = 1, self._label(widget.text, style)
            elif isinstance(widget, Button):
                kind, label = 2, self._label(widget.label, style)
            elif isinstance(widget, Slider):
                kind = 3
                if widget.max > widget.min:
                    value = (widget.value - widget.min) / (widget.max - widget.min)
            elif isinstance(widget, Toggle):
                kind, label, value = 4, self._label(widget.label, style), float(widget.value)
            elif isinstance(widget, Joystick):
                kind, stick = 5, widget.value
            elif isinstance(widget, Image):
                kind = 6
                fg = parse_color(style["tint"]) if style.get("tint") else _WHITE
                fg = (*fg[:3], fg[3] * float(style.get("opacity", 1.0)))
                image_size = widget.intrinsic_size()
                texture = self._texture(widget, image_size, scale * supersample, used_images)
                image = (
                    texture.id, image_size, style.get("flex") is not None,
                    style.get("uv") or (0, 0, 1, 1),
                )
            elif not isinstance(widget, Panel):
                raise NotImplementedError(f"native GUI widget {type(widget).__name__}")
            index = len(declarations)
            identifier = id(widget)
            widgets[identifier] = widget
            declarations.append((
                identifier, parent,
                (anchor, offset, style.get("direction") == "h",
                 parse_padding(style.get("padding", 0)), style.get("gap", 0), size, style.get("flex")),
                (bg, fg, border_color, border, radius),
                (kind, label, value, stick, image),
            ))
            if isinstance(widget, Panel):
                for child in widget.children:
                    add(child, index)

        for panel in engine.gui:
            add(panel, None)
        if getattr(engine, "_debug_hud", False):
            add(self._hud(engine), None)
        self._images = {key: handle for key, handle in self._images.items() if key in used_images}
        engine._graphics.gui_tree(
            declarations, (engine.w * supersample, engine.h * supersample),
            scale * supersample, engine.gui.exact_colors,
        )
        self._widgets = widgets
        self.getter_total += self.getter_calls

    def _hud(self, engine):
        """The `--debug` overlay: native GUI paint and font, outside engine.gui."""
        if self._hud_panel is None:
            self._hud_texts = [
                Text("", style_overrides={"font_size": 11, "fg": "#f2f2f2"})
                for _ in range(6)
            ]
            self._hud_panel = Panel(
                self._hud_texts,
                anchor="top-left",
                offset=(8, 8),
                style_overrides={
                    "bg": "#101014e6",
                    "padding": 6,
                    "gap": 2,
                    "radius": 4,
                },
            )
        lines = engine._hud_lines()
        for text, value in zip(self._hud_texts, lines):
            text.text = value
        for text in self._hud_texts[len(lines):]:
            text.text = ""
        # The border is the one glanceable marker: red while a synchronous
        # callback is over 1 ms, green when none is measured.
        slow = bool(engine.over_budget)
        self._hud_panel.style_overrides["border_color"] = "#e84848" if slow else "#3a8a4a"
        self._hud_panel.style_overrides["border"] = 2 if slow else 1
        return self._hud_panel

    def _label(self, text, style):
        font = style.get("font")
        key = str(font) if font else ""
        if key not in self._fonts:
            path = Path(font) if font else files("manifoldx.viz.assets").joinpath("DejaVuSansMono.ttf")
            self._engine._graphics.gui_font(key, path.read_bytes())
            self._fonts.add(key)
        return str(text), key, float(style.get("font_size", 12))

    def _texture(self, widget, size, scale, used):
        handle = widget.svg
        if isinstance(handle, TextureHandle):
            if not self._engine._owns_texture(handle):
                raise ValueError("GUI texture belongs to a different or closed graphics owner")
            return handle
        from PIL import Image as PillowImage
        from manifoldx.svg import rasterize_digest

        pixels = tuple(min(4096, max(1, round(v * scale))) for v in size)
        key = (widget.svg_digest, *pixels)
        used.add(key)
        if key not in self._images:
            rgba = rasterize_digest(widget.svg_data, widget.svg_digest, *pixels)
            self._images[key] = self._engine._upload_native_albedo(PillowImage.fromarray(rgba), None)
        return self._images[key]

    def pointer(self, event, phase):
        self.last_target = None
        engine = self._engine
        if engine.input.pointer_captured:
            return False
        if not self._widgets:
            return False
        scale = self.scale
        x, y = event.x / scale, event.y / scale
        consumed, hit = engine._graphics.gui_pointer(getattr(event, "pointer_id", 0), phase, (x, y))
        engine.gui.pointer_over_gui |= consumed
        if hit is not None:
            identifier, box = hit
            widget = self._widgets[identifier]
            self.last_target = getattr(widget, "name", None)
            widget._layout_box = LayoutBox(*box)
            hook = getattr(widget, f"_on_pointer_{phase}", None)
            if hook is not None:
                hook(replace(event, x=x, y=y, dx=event.dx / scale, dy=event.dy / scale), engine)
        return consumed

    def key(self, event):
        self.last_target = None
        sign = STEPS.get(event.key)
        widget = self._widgets.get(self._engine._graphics.gui_focus()) if sign is not None else None
        if isinstance(widget, Slider):
            self.last_target = widget.name
            widget._on_step(sign, self._engine)
            return True
        return False

    def reset(self):
        self._engine._graphics.gui_reset()
        self._engine.gui.pointer_over_gui = False
        self.last_target = None
