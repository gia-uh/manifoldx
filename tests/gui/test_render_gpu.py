"""GPU-gated tests for the gui render pass.

These tests need an offscreen wgpu device; they pytest.skip on machines
without one (matches the rest of the suite's gating).
"""

import pytest

import numpy as np

from manifoldx.gui import Panel, style


def _make_offscreen_engine(width: int = 128, height: int = 128):
    try:
        from manifoldx.backends import get_offscreen_canvas
        canvas = get_offscreen_canvas(width=width, height=height)
    except Exception as e:
        pytest.skip(f"offscreen canvas unavailable: {e}")
    import manifoldx as mx
    engine = mx.Engine("test", width=width, height=height)
    engine._init_canvas(canvas)
    engine._running = True
    return engine, canvas


def setup_function(_):
    style.reset()


def test_engine_exposes_gui_root_listlike_with_pointer_flag():
    engine, _ = _make_offscreen_engine()
    assert hasattr(engine, "gui")
    assert len(engine.gui) == 0
    p = Panel(children=[])
    engine.gui.append(p)
    assert engine.gui[0] is p
    assert engine.gui.pointer_over_gui is False


def test_gui_pass_renders_panel_rect_into_framebuffer():
    style.set_theme({"bg": "#ff0000ff", "padding": 4, "gap": 0})
    engine, canvas = _make_offscreen_engine(width=128, height=128)
    panel = Panel(
        children=[],
        anchor="top-left",
        offset=(10, 20),
        style_overrides={"width": 40, "height": 30},
    )
    engine.gui.append(panel)
    engine._draw_frame()
    frame = np.asarray(canvas.draw())  # (H, W, 4) uint8 RGBA
    # Inside the panel rect (offset 10,20 size 40x30): expect ~red.
    inside = frame[30, 25]  # y=30 is inside [20,50], x=25 is inside [10,50]
    assert inside[0] > 200, f"expected red inside panel, got {inside.tolist()}"
    assert inside[1] < 80
    assert inside[2] < 80
    # Outside the panel: should be the default clear colour (dark-blue
    # background), which is distinctly different from the red panel.
    # The clear colour is (0.1, 0.1, 0.2) linear → sRGB ≈ (89, 89, 124).
    # We just check the red channel is significantly lower than inside the
    # panel and that it looks dark-blue (G ≈ B, not saturated-red).
    outside = frame[100, 100]
    assert inside[0] - outside[0] > 100, (
        f"outside pixel should be much less red than inside: inside={inside.tolist()}, "
        f"outside={outside.tolist()}"
    )


def test_gui_pass_renders_text_widget():
    from manifoldx.gui import Text
    style.set_theme({"bg": "#000000ff", "fg": "#ffffffff", "font_size": 16,
                     "padding": 2, "gap": 0})
    engine, canvas = _make_offscreen_engine(width=160, height=64)
    panel = Panel(
        children=[Text("hello")],
        anchor="top-left",
        offset=(0, 0),
        style_overrides={"width": 100, "height": 40},
    )
    engine.gui.append(panel)
    engine._draw_frame()
    frame = np.asarray(canvas.draw())
    # Sum brightness over the text region — non-empty means glyphs drew.
    region = frame[6:30, 4:96, :3].astype(np.int32).sum()
    assert region > 5000, (
        f"expected glyphs to brighten the panel region; got region sum={region}"
    )


def test_gui_pass_drives_value_display_each_frame():
    from manifoldx.gui import ValueDisplay
    style.set_theme({"bg": "#000000ff", "fg": "#ffffffff", "font_size": 14,
                     "padding": 2, "gap": 0})
    engine, canvas = _make_offscreen_engine(width=160, height=64)
    calls = []
    def getter():
        calls.append(1)
        return f"n={len(calls)}"
    vd = ValueDisplay(getter=getter, min_width=80)
    panel = Panel(
        children=[vd],
        anchor="top-left",
        offset=(0, 0),
        style_overrides={"width": 100, "height": 40},
    )
    engine.gui.append(panel)
    engine._draw_frame()
    engine._draw_frame()
    engine._draw_frame()
    assert len(calls) == 3


def test_a_still_gui_makes_no_bind_group_after_its_first_frames():
    """Image runs reuse their bind group: making one per draw cost a HUD about 2 ms a frame."""
    pytest.importorskip("PIL")
    from importlib.resources import files

    from manifoldx.gui import Text

    font = str(files("manifoldx.viz.assets").joinpath("DejaVuSansMono.ttf"))
    engine, _ = _make_offscreen_engine()
    for corner in ("top-left", "bottom-right"):
        engine.gui.append(Panel([Text("12", style_overrides={"font": font, "font_size": 16})],
                                anchor=corner, style_overrides={"bg": "#202020"}))
    engine.render_frame(supersample=1)
    engine.render_frame(supersample=1)
    device = engine._device
    made = []
    real = device.create_bind_group
    device.create_bind_group = lambda **kw: made.append(1) or real(**kw)
    try:
        engine.render_frame(supersample=1)
    finally:
        del device.create_bind_group
    assert made == []
