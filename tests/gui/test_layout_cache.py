"""layout_gui lays out again only the root panels that changed since the last frame."""

import pytest

import manifoldx as mx
from manifoldx.gui import Panel, Text
from manifoldx.gui import layout as layout_mod
from manifoldx.render.passes import gui as gui_pass
from manifoldx.render.passes.gui import layout_gui


def _engine():
    engine = mx.Engine("cache", width=400, height=300)
    a = Text("10:00", style_overrides={"font_size": 20, "fg": "#ffffff"})
    b = Text("0", style_overrides={"font_size": 30, "fg": "#ffffff"})
    left = Panel([a], anchor="top-left", offset=(8, 8), style_overrides={"bg": "#000000"})
    right = Panel([b], anchor="bottom-right", offset=(8, 8), style_overrides={"bg": "#000000"})
    engine.gui.append(left)
    engine.gui.append(right)
    return engine, left, right, a, b


@pytest.fixture
def layouts(monkeypatch):
    """Counts compute_layout calls, one per root panel laid out."""
    calls = []
    real = layout_mod.compute_layout

    def counted(*args, **kwargs):
        calls.append(1)
        return real(*args, **kwargs)

    monkeypatch.setattr(gui_pass, "compute_layout", counted)
    return calls


def _ops(p):
    return (p.rect_ops, p.text_ops, p.image_ops, p.order)


def test_unchanged_gui_is_not_laid_out_again(layouts):
    engine, *_ = _engine()
    first = layout_gui(engine)
    n = len(layouts)
    second = layout_gui(engine)
    assert len(layouts) == n
    assert _ops(second) == _ops(first)


def test_only_the_changed_panel_is_laid_out_again(layouts):
    engine, left, right, a, b = _engine()
    layout_gui(engine)
    n = len(layouts)
    b.text = "88"
    p = layout_gui(engine)
    assert len(layouts) == n + 1
    assert [op.text for op in p.text_ops] == ["10:00", "88"]


@pytest.mark.parametrize("change", [
    lambda left, a: setattr(a, "text", "10:01"),
    lambda left, a: setattr(left, "offset", (20, 8)),
    lambda left, a: a.style_overrides.update(fg="#ff0000"),  # a dict changed in place
    lambda left, a: setattr(a, "style_overrides", {**a.style_overrides, "font_size": 24}),
    lambda left, a: left.children.append(Text("x", style_overrides={"font_size": 10})),
])
def test_any_change_shows_on_the_next_layout(change):
    engine, left, right, a, b = _engine()
    fresh_engine, fresh_left, _, fresh_a, _ = _engine()
    layout_gui(engine)
    change(left, a)
    change(fresh_left, fresh_a)
    assert _ops(layout_gui(engine)) == _ops(layout_gui(fresh_engine))


def test_a_new_window_size_lays_out_again():
    engine, *_ = _engine()
    before = layout_gui(engine).rect_ops[1].box
    engine.w = 800
    after = layout_gui(engine).rect_ops[1].box
    assert after.x > before.x


def test_the_order_indexes_the_merged_ops():
    engine, *_ = _engine()
    layout_gui(engine)
    p = layout_gui(engine)
    assert p.order == [("rect", 0), ("text", 0), ("rect", 1), ("text", 1)]


def test_a_removed_panel_is_dropped():
    engine, left, right, a, b = _engine()
    layout_gui(engine)
    engine.gui.remove(right)
    p = layout_gui(engine)
    assert [op.text for op in p.text_ops] == ["10:00"]


def test_a_new_theme_lays_out_again():
    from manifoldx.gui import style

    engine, *_ = _engine()
    layout_gui(engine)
    try:
        style.set_theme({"border": 3})
        assert layout_gui(engine).rect_ops[0].border == 3
    finally:
        style.reset()
