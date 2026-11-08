"""Under a fixed timestep `engine.elapsed` is the sum of the fixed steps simulated so far.

A frame observes the clock before its step: 0 on the first frame, then one dt per
completed step. Fixed-dt mode must advance it as the wall-clock branch does, or
`engine.delay()`, `elapsed_at()` and frame code reading the clock stand still.
"""

import pytest


def _headless_engine(width=64, height=64):
    try:
        from manifoldx.backends import get_offscreen_canvas

        canvas = get_offscreen_canvas(width=width, height=height)
    except Exception as e:  # pragma: no cover - depends on the host GPU
        pytest.skip(f"offscreen canvas unavailable: {e}")
    import manifoldx as mx

    engine = mx.Engine("fixed clock", width=width, height=height)
    engine._init_canvas(canvas)
    engine._running = True
    return engine


def test_fixed_steps_advance_elapsed():
    engine = _headless_engine()
    dt, frames = 0.02, 5
    seen = []

    @engine.system
    def sample(query, step):
        seen.append(engine.elapsed)

    engine.set_fixed_timestep(dt)
    for _ in range(frames):
        engine._draw_frame()

    # Frame i sees the clock after i completed steps.
    assert seen == pytest.approx([i * dt for i in range(frames)])
    assert engine.elapsed == pytest.approx(frames * dt)


def test_the_frame_event_sees_the_same_clock_as_systems():
    engine = _headless_engine()
    dt = 0.05
    payloads = []

    def on_frame(p: dict):
        payloads.append(p["elapsed"])

    @engine.system
    def sample(query, step):
        assert engine.elapsed == payloads[-1]

    engine.on("frame")(on_frame)
    engine.set_fixed_timestep(dt)
    for _ in range(3):
        engine._draw_frame()

    assert payloads == pytest.approx([0.0, dt, 2 * dt])
