"""Tests for manifoldx.gui.painter — rect + text op accumulation."""

from manifoldx.gui import style
from manifoldx.gui.layout import LayoutBox, compute_layout
from manifoldx.gui.painter import Painter, paint
from manifoldx.gui.widgets import Panel, Text


def setup_function(_):
    style.reset()


def test_painter_starts_empty():
    p = Painter()
    assert p.rect_ops == []
    assert p.text_ops == []


def test_painter_draw_rect_records_op():
    p = Painter()
    p.draw_rect(
        box=LayoutBox(10, 20, 100, 50),
        fill=(1, 0, 0, 1),
        border_color=(0, 1, 0, 1),
        border=2.0,
        radius=4.0,
    )
    assert len(p.rect_ops) == 1
    op = p.rect_ops[0]
    assert op.box == LayoutBox(10, 20, 100, 50)
    assert op.fill == (1, 0, 0, 1)
    assert op.border == 2.0
    assert op.radius == 4.0


def test_painter_draw_text_records_op():
    p = Painter()
    p.draw_text(box=LayoutBox(0, 0, 50, 12), text="hi", font_size=12, fg=(1, 1, 1, 1))
    assert len(p.text_ops) == 1
    op = p.text_ops[0]
    assert op.text == "hi"
    assert op.font_size == 12


def test_paint_walks_panel_emitting_rect_then_children():
    style.set_theme({"bg": "#222", "padding": 0, "gap": 0})
    style.define("filled", {"bg": "#ff0000"})
    panel = Panel(children=[Text("hi")], style="filled")
    spec = panel.build_layout_spec()
    boxes = compute_layout(spec, viewport=LayoutBox(0, 0, 100, 100))
    p = Painter()
    paint(panel, spec, boxes, p)
    # Panel emits a rect with its bg.
    assert len(p.rect_ops) == 1
    assert p.rect_ops[0].fill[0] == 1.0  # red channel from #ff0000
    # Child Text emits exactly one text op.
    assert len(p.text_ops) == 1
    assert p.text_ops[0].text == "hi"


def test_paint_nested_panels_walks_depth_first():
    inner = Panel(children=[Text("inner")])
    outer = Panel(children=[Text("outer"), inner])
    spec = outer.build_layout_spec()
    boxes = compute_layout(spec, viewport=LayoutBox(0, 0, 200, 200))
    p = Painter()
    paint(outer, spec, boxes, p)
    # Two panels → 2 rects; two text widgets → 2 text ops.
    assert len(p.rect_ops) == 2
    assert {op.text for op in p.text_ops} == {"inner", "outer"}


def test_paint_value_display_emits_text_op():
    from manifoldx.gui import ValueDisplay
    vd = ValueDisplay(getter=lambda: "fps: 60")
    vd.refresh()
    panel = Panel(children=[vd])
    spec = panel.build_layout_spec()
    boxes = compute_layout(spec, viewport=LayoutBox(0, 0, 100, 50))
    p = Painter()
    paint(panel, spec, boxes, p)
    assert any(op.text == "fps: 60" for op in p.text_ops)


def test_joystick_paints_a_pad_and_a_knob():
    from manifoldx.gui import Joystick
    from manifoldx.gui.style import parse_color

    j = Joystick(name="wind")
    j.value = (0.5, 0.5)
    spec = {}
    box = LayoutBox(10.0, 20.0, 100.0, 100.0)
    p = Painter()
    paint(j, spec, {id(spec): box}, p)
    s = j.effective_style()
    assert len(p.rect_ops) == 2
    pad, knob = p.rect_ops
    assert pad.box == box
    assert pad.radius == box.w / 2
    assert pad.fill == parse_color(s["bg"])
    k = 0.4 * box.w
    r = box.w / 2
    cx = box.x + r + 0.5 * (r - k / 2)
    cy = box.y + r - 0.5 * (r - k / 2)  # y down on screen for a positive value[1]
    assert knob.box == LayoutBox(cx - k / 2, cy - k / 2, k, k)
    assert knob.radius == k / 2
    assert knob.fill == parse_color(s["fg"])
