"""Async handlers, await engine.tick / delay / elapsed_at, error propagation."""
import pytest


def _make_offscreen_engine(width=64, height=64):
    try:
        from manifoldx.backends import get_offscreen_canvas
        canvas = get_offscreen_canvas(width=width, height=height)
    except Exception as e:
        pytest.skip(f"offscreen canvas unavailable: {e}")
    import manifoldx as mx
    engine = mx.Engine("test", width=width, height=height)
    engine._init_canvas(canvas)
    engine._running = True
    return engine


def test_async_handler_runs_to_first_await_on_dispatch_frame():
    engine = _make_offscreen_engine()
    log = []

    @engine.on("go")
    async def go(payload):
        log.append("before")
        await engine.tick()
        log.append("after")

    engine.emit("go")
    engine._draw_frame()
    assert log == ["before"]
    engine._draw_frame()
    assert log == ["before", "after"]


def test_await_delay_resolves_after_seconds_elapsed():
    """Drive frames manually with a fixed timestep, setting engine.elapsed before
    each frame to place the deadline.
    """
    engine = _make_offscreen_engine()
    engine.set_fixed_timestep(0.1)  # keep _compute_dt from overwriting the elapsed we set
    log = []

    @engine.on("start")
    async def start(payload):
        await engine.delay(0.5)
        log.append("done")

    engine.emit("start")
    engine.elapsed = 0.0
    engine._draw_frame()  # dispatch + first await; deadline set to 0.5
    assert log == []

    last_tick = 0
    for tick in range(1, 11):
        engine.elapsed = tick * 0.1
        engine._draw_frame()
        last_tick = tick
        if log:
            break

    assert log == ["done"]
    assert last_tick == 5


def test_await_elapsed_at_uses_absolute_target():
    engine = _make_offscreen_engine()
    engine.set_fixed_timestep(0.1)
    log = []

    @engine.on("start")
    async def start(payload):
        await engine.elapsed_at(0.3)
        log.append("done")

    engine.emit("start")
    engine.elapsed = 0.0
    engine._draw_frame()  # dispatch handler, register waiter

    last_tick = 0
    for tick in range(1, 11):
        engine.elapsed = tick * 0.1
        engine._draw_frame()
        last_tick = tick
        if log:
            break
    assert log == ["done"]
    assert last_tick == 3


def test_async_handler_exception_propagates():
    engine = _make_offscreen_engine()

    @engine.on("boom")
    async def boom(payload):
        raise RuntimeError("kaboom")

    engine.emit("boom")
    # Frame 1: dispatch creates the task; pump runs it; task raises;
    # done-callback spools the error onto _task_errors; pump re-raises.
    with pytest.raises(RuntimeError, match="kaboom"):
        engine._draw_frame()


def test_pump_under_running_loop_does_not_recurse():
    """Regression: in interactive run() mode, rendercanvas owns a running
    asyncio loop. _draw_frame -> _pump_aio_loop must not call
    run_until_complete on a different loop (RuntimeError) or recursively
    on the same one. It should detect the running loop and skip its own
    run_until_complete, leaving task scheduling to the host loop.
    """
    import asyncio

    engine = _make_offscreen_engine()
    log = []

    @engine.on("frame")
    def on_frame(payload):
        log.append(payload["frame"])

    async def driver():
        # Inside this coroutine an asyncio loop is running. Calling
        # _draw_frame should NOT raise "Cannot run the event loop while
        # another loop is running".
        engine._draw_frame()
        engine._draw_frame()

    asyncio.run(driver())
    assert log == [0, 1]


def test_sync_handler_exception_propagates_immediately():
    engine = _make_offscreen_engine()

    @engine.on("boom")
    def boom(payload):
        raise ValueError("sync-boom")

    engine.emit("boom")
    with pytest.raises(ValueError, match="sync-boom"):
        engine._draw_frame()
