"""Teardown for one-engine, many-screens apps: remove_system, off, clear_scene."""
import numpy as np
import pytest

import manifoldx as mx


def _engine():
    return mx.Engine("teardown")


# --- remove_system --------------------------------------------------------------


def test_remove_system_stops_it_running():
    engine = _engine()
    calls = []

    @engine.system
    def counter(query: mx.Query[mx.Transform], dt: float):
        calls.append(dt)

    engine.systems.run_all(engine, 0.1)
    engine.remove_system(counter)
    engine.systems.run_all(engine, 0.1)
    assert len(calls) == 1


def test_remove_system_from_inside_itself_keeps_the_rest_running():
    engine = _engine()
    seen = []

    @engine.system
    def once(query: mx.Query[mx.Transform], dt: float):
        seen.append("once")
        engine.remove_system(once)

    @engine.system
    def always(query: mx.Query[mx.Transform], dt: float):
        seen.append("always")

    engine.systems.run_all(engine, 0.1)
    engine.systems.run_all(engine, 0.1)
    assert seen == ["once", "always", "always"]


def test_removing_a_later_system_mid_frame_skips_it_this_frame():
    engine = _engine()
    seen = []

    @engine.system
    def first(query: mx.Query[mx.Transform], dt: float):
        seen.append("first")
        engine.remove_system(second)

    @engine.system
    def second(query: mx.Query[mx.Transform], dt: float):
        seen.append("second")

    engine.systems.run_all(engine, 0.1)
    assert seen == ["first"]


def test_remove_unknown_system_raises():
    engine = _engine()
    with pytest.raises(ValueError):
        engine.remove_system(lambda q, dt: None)


# --- off ------------------------------------------------------------------------


def test_off_stops_a_handler():
    engine = _engine()
    seen = []

    @engine.on("ping")
    def handler(payload):
        seen.append(payload)

    engine.emit("ping", 1)
    engine._event_bus.dispatch_pending(engine)
    engine.off("ping", handler)
    engine.emit("ping", 2)
    engine._event_bus.dispatch_pending(engine)
    assert seen == [1]


def test_off_during_dispatch_skips_later_handlers_and_keeps_others():
    engine = _engine()
    seen = []

    @engine.on("ping")
    def first(payload):
        seen.append("first")
        engine.off("ping", first)
        engine.off("ping", second)

    @engine.on("ping")
    def second(payload):
        seen.append("second")

    @engine.on("ping")
    def third(payload):
        seen.append("third")

    engine.emit("ping")
    engine.emit("ping")
    engine._event_bus.dispatch_pending(engine)
    assert seen == ["first", "third", "third"]


def test_off_unknown_handler_raises():
    engine = _engine()
    with pytest.raises(ValueError):
        engine.off("ping", lambda payload: None)


# --- clear_scene ----------------------------------------------------------------


def _offscreen_engine():
    try:
        from manifoldx.backends import get_offscreen_canvas

        canvas = get_offscreen_canvas(width=64, height=64)
    except Exception as e:
        pytest.skip(f"offscreen canvas unavailable: {e}")
    engine = mx.Engine("teardown", width=64, height=64)
    engine._init_canvas(canvas)
    engine._running = True
    return engine


CUBE = mx.geometry.cube(1, 1, 1)
RED = mx.material.phong(mx.colors.RED)


def _spawn(engine, n=1):
    return engine.spawn(mx.Mesh(CUBE), mx.Material(RED), mx.Transform(pos=(0, 0, 0)), n=n)


def _alive(engine):
    return int(np.count_nonzero(engine.store._alive))


def test_clear_scene_from_a_system_destroys_entities_and_panels():
    from manifoldx.gui import Panel, Text

    engine = _offscreen_engine()
    _spawn(engine, n=3)
    engine.gui.append(Panel(children=[Text("hi")]))
    kept = []

    @engine.system
    def leave(query: mx.Query[mx.Transform], dt: float):
        engine.clear_scene()
        engine.remove_system(leave)

    @engine.system
    def keep(query: mx.Query[mx.Transform], dt: float):
        kept.append(dt)

    engine._draw_frame()
    assert _alive(engine) == 0
    assert len(engine.gui) == 0
    engine._draw_frame()
    assert len(kept) == 2  # systems stay


def test_entities_spawned_after_clear_scene_in_the_same_frame_survive():
    engine = _offscreen_engine()
    _spawn(engine, n=2)

    @engine.system
    def swap(query: mx.Query[mx.Transform], dt: float):
        engine.clear_scene()
        _spawn(engine, n=1)
        engine.remove_system(swap)

    engine._draw_frame()
    assert _alive(engine) == 1


def test_clear_scene_between_frames_is_applied_next_frame():
    # In run() mode an async handler resumed after `await engine.tick()` runs
    # between frames; its destroy commands must reach the next frame's flush.
    engine = _offscreen_engine()
    _spawn(engine, n=2)
    engine._draw_frame()
    engine.clear_scene()
    _spawn(engine, n=1)
    engine._draw_frame()
    assert _alive(engine) == 1


def test_clear_scene_with_a_pending_destroy_does_not_double_free():
    engine = _offscreen_engine()
    handle = _spawn(engine, n=2)
    engine.destroy([handle.index])
    engine.clear_scene()
    engine._draw_frame()
    a, b = _spawn(engine).index, _spawn(engine).index
    assert a != b
    assert _alive(engine) == 2


def test_clear_scene_keeps_camera_lights_and_background():
    engine = _offscreen_engine()
    from manifoldx.resources import PointLight

    light = PointLight("#ffffff", 1.0, (1, 2, 3))
    engine.add_light(light)
    engine.background_color = (0.3, 0.2, 0.1)
    camera = engine.camera
    engine.clear_scene()
    engine._draw_frame()
    assert engine._lights == [light]
    assert engine.camera is camera
    assert engine.background_color == (0.3, 0.2, 0.1)


def test_clear_scene_keeps_the_loading_spinner():
    engine = _offscreen_engine()
    with engine.loading("level"):
        engine._draw_frame()
        assert len(engine.gui) == 1
        engine.clear_scene()
        engine._draw_frame()
        assert list(engine.gui) == [engine._loading_panel]
    engine._draw_frame()  # closing the block removes the spinner without error
    assert len(engine.gui) == 0


def test_a_pointer_event_after_clear_scene_reaches_the_gui():
    """clear_scene left the GUI bridge's capture table as None, so the next
    pointer event raised AttributeError."""
    from manifoldx.input import PointerEvent

    engine = _engine()
    engine.clear_scene()
    ev = PointerEvent(x=5.0, y=5.0, dx=0.0, dy=0.0, button=1, buttons=(1,),
                      modifiers=(), phase="move", pointer_id=0)
    engine.emit("pointer_move", ev)
    engine.render_frame(supersample=1)  # handlers run as the frame drains its events
