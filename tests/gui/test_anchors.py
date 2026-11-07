"""Root panels honour all nine anchors on the desktop, as the page places them.

The expected rect comes from the page's rule (src/manifoldx/web/js/gui.js, panelCss): an
offset measured inward from the anchored edges (`left`/`right`, `top`/`bottom`), and on a
centred axis `left: 50%` (or `top`) with `translate(-50%) translate(offset)`, so the offset
moves the panel right or down. CSS works in screen pixels, sizes and offsets times the
GUI's scale.
"""

import pytest

import manifoldx as mx
from manifoldx.gui import Button, Panel, style
from manifoldx.gui.layout import LayoutBox
from manifoldx.gui.widgets import _VALID_ANCHORS
from manifoldx.input import PointerEvent
from manifoldx.render.passes.gui import layout_gui

# gui.js's AXES: the anchor's horizontal and vertical placement.
AXES = {
    "top-left": ("start", "start"),
    "top-center": ("center", "start"),
    "top-right": ("end", "start"),
    "center-left": ("start", "center"),
    "center": ("center", "center"),
    "center-right": ("end", "center"),
    "bottom-left": ("start", "end"),
    "bottom-center": ("center", "end"),
    "bottom-right": ("end", "end"),
}

W, H = 64, 30  # the panel's size in UI pixels
OFFSET = (12, 7)


def setup_function(_):
    style.reset()


def _page_axis(place: str, viewport: float, size: float, offset: float) -> float:
    """Where the CSS of panelCss puts a box's near edge on one axis of the viewport."""
    if place == "start":  # left: ox
        return offset
    if place == "end":  # right: ox
        return viewport - size - offset
    return viewport * 0.5 - size * 0.5 + offset  # left: 50%, translate(-50%) translate(ox)


def page_rect(anchor, offset, size, screen, scale=1.0) -> LayoutBox:
    h, v = AXES[anchor]
    w, ht = size[0] * scale, size[1] * scale
    ox, oy = offset[0] * scale, offset[1] * scale
    return LayoutBox(_page_axis(h, screen[0], w, ox), _page_axis(v, screen[1], ht, oy), w, ht)


def _engine(anchor, w=800, h=600):
    engine = mx.Engine("t", width=w, height=h)
    button = Button(name="b", label="b", style_overrides={"flex": 1})
    engine.gui.append(
        Panel(
            children=[button],
            anchor=anchor,
            offset=OFFSET,
            style_overrides={"width": W, "height": H, "padding": 0},
        )
    )
    return engine, button


def _painted(engine) -> LayoutBox:
    return layout_gui(engine).rect_ops[0].box


def _hit_box(engine, x, y):
    """The (widget, box) the desktop's bridge finds at a screen point, the box in screen pixels."""
    ev = PointerEvent(x=x, y=y, dx=0.0, dy=0.0, button=1, buttons=(1,), modifiers=(), phase="down")
    _, hit = engine._gui_bridge._locate(ev)
    if hit is None:
        return None
    widget, box = hit
    s = engine.gui.effective_scale(engine.h)
    return widget, LayoutBox(box.x * s, box.y * s, box.w * s, box.h * s)


def test_the_table_covers_every_anchor_a_panel_accepts():
    assert set(AXES) == set(_VALID_ANCHORS)


@pytest.mark.parametrize("anchor", sorted(AXES))
def test_paint_and_hit_test_place_the_panel_as_the_page(anchor):
    engine, button = _engine(anchor)
    want = page_rect(anchor, OFFSET, (W, H), (800, 600))
    assert _painted(engine) == want
    cx, cy = want.x + want.w / 2, want.y + want.h / 2
    assert _hit_box(engine, cx, cy) == (button, want)


@pytest.mark.parametrize("anchor", sorted(set(AXES) - {"top-left"}))
def test_the_top_left_corner_is_not_the_panel(anchor):
    engine, _ = _engine(anchor)
    assert _hit_box(engine, OFFSET[0] + 1, OFFSET[1] + 1) is None


@pytest.mark.parametrize("anchor", sorted(AXES))
def test_a_resize_moves_the_panel_with_its_edges(anchor):
    engine, button = _engine(anchor)
    engine._on_canvas_resize(mx.input.ResizeEvent(width=1024, height=480, pixel_ratio=1.0))
    assert (engine.w, engine.h) == (1024, 480)
    want = page_rect(anchor, OFFSET, (W, H), (1024, 480))
    assert _painted(engine) == want
    assert _hit_box(engine, want.x + 1, want.y + 1) == (button, want)


@pytest.mark.parametrize("anchor", sorted(AXES))
def test_the_rule_holds_under_the_gui_scale(anchor):
    engine, button = _engine(anchor, w=1920, h=1080)
    engine.gui.design_height = 540  # scale 2
    want = page_rect(anchor, OFFSET, (W, H), (1920, 1080), scale=2.0)
    assert _painted(engine) == want
    assert _hit_box(engine, want.x + want.w - 1, want.y + want.h - 1) == (button, want)


@pytest.mark.parametrize("anchor", ["bottom-right", "center", "top-center"])
def test_a_panel_without_a_size_fits_its_content_on_a_placed_axis(anchor):
    """The page sizes an absolute panel with no width or height to its content; the desktop
    does the same for every anchor but top-left (which fills the window, as it always has),
    so the panel touches its edges instead of filling the window from them."""
    engine = mx.Engine("t", width=800, height=600)
    button = Button(name="b", label="b", style_overrides={"width": 50, "height": 20})
    engine.gui.append(
        Panel(children=[button], anchor=anchor, offset=OFFSET, style_overrides={"padding": 5})
    )
    want = page_rect(anchor, OFFSET, (60, 30), (800, 600))
    assert _painted(engine) == want
