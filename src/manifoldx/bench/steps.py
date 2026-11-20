"""The engine's steps: camera poses and paths, waiting, phases, shots, state and events.

None of them knows what an app is about. Positions are world coordinates; a camera far from the
engine's floating origin moves the origin first, so what is drawn stays near it. An app adds its
own steps with `manifoldx.bench.step`.
"""

from __future__ import annotations

import math
import time

import numpy as np

from manifoldx.bench.scenario import Ctx, StepCall, engine_step

FAR = 2000.0  # metres from the origin past which a camera step moves the origin


def _pose(ctx: Ctx, pos, look_at) -> None:
    engine = ctx.engine
    pos, look_at = np.asarray(pos, float), np.asarray(look_at, float)
    ox, oz = engine.origin
    if math.hypot(pos[0] - ox, pos[2] - oz) > FAR:
        engine.set_origin(float(pos[0]), float(pos[2]))
        ox, oz = engine.origin
    shift = np.array([ox, 0.0, oz])
    engine.camera.set_pose(pos - shift, look_at - shift)


def _wait(ctx: Ctx, until, max_s):
    """Yield frames until `until()` is true; past max_s seconds, time out."""
    t0 = time.monotonic()
    while not until():
        if time.monotonic() - t0 > max_s:
            ctx.timeout(max_s)
        yield


# --- what a scenario writes ----------------------------------------------------------------------


def camera(pos, look_at) -> StepCall:
    """Put the camera at `pos`, looking at `look_at`, both in world coordinates. Takes no frame."""
    return StepCall("camera", (tuple(pos), tuple(look_at)))


def fly(frm, to, speed: float) -> StepCall:
    """Fly the camera in a straight line from `frm` to `to` at `speed` m/s, looking ahead at `to`."""
    return StepCall("fly", (tuple(frm), tuple(to), float(speed)))


def zoom(h0: float, h1: float, *, at=(0.0, 0.0), seconds: float | None = None, steps: int | None = None,
         hold=None, max_s: float = 30.0) -> StepCall:
    """Move the camera from height h0 to h1 over the point `at` (x, z), geometrically: over `seconds`
    of frames, or in `steps` heights, each held until `hold()` is true (at most max_s seconds each)."""
    return StepCall("zoom", (float(h0), float(h1)),
                    {"at": tuple(at), "seconds": seconds, "steps": steps, "hold": hold, "max_s": max_s})


def hold(*, seconds: float | None = None, frames: int | None = None, until=None, max_s: float = 60.0) -> StepCall:
    """Keep rendering: for `seconds`, for `frames`, or until `until()` is true (at most max_s seconds)."""
    return StepCall("hold", (), {"seconds": seconds, "frames": frames, "until": until, "max_s": max_s})


def each_frame(fn, *, frames: int | None = None, until=None, max_s: float = 60.0) -> StepCall:
    """Call `fn(app)` every frame, for `frames` frames or until `until()` is true (at most max_s seconds)."""
    return StepCall("each_frame", (fn,), {"frames": frames, "until": until, "max_s": max_s})


def set(patch: dict) -> StepCall:  # noqa: A001 - the agent loop's verb
    """Apply a JSON merge patch to engine.state, as `manifoldx set` does. Takes no frame."""
    return StepCall("set", (patch,))


def emit(event: str, payload=None) -> StepCall:
    """Emit an event the app handles, as `manifoldx emit` does. Takes no frame."""
    return StepCall("emit", (event, payload))


def mark(name: str) -> StepCall:
    """Start a phase: every probe reports per phase. Takes no frame."""
    return StepCall("mark", (name,))


def shot(name: str) -> StepCall:
    """Save the next rendered frame as shots/<name>.png."""
    return StepCall("shot", (name,))


def settled(streamer):
    """A condition for hold/zoom: the streamer has drawn everything the camera needs."""
    return lambda: streamer.settled


def drawn(streamer, node):
    """A condition for hold/zoom: `node` (a cell or a quadtree node) is drawn."""
    return lambda: node in streamer.drawn()


# --- how they run --------------------------------------------------------------------------------


@engine_step("camera")
def _camera(ctx, pos, look_at):
    _pose(ctx, pos, look_at)


@engine_step("fly")
def _fly(ctx, frm, to, speed):
    a, b = np.asarray(frm, float), np.asarray(to, float)
    length = float(np.linalg.norm(b - a))
    done = 0.0
    while done < length:
        p = a + (b - a) * (done / length)
        _pose(ctx, p, b)
        yield
        done += speed * ctx.dt
    _pose(ctx, b, b + (b - a) / max(length, 1e-9))


@engine_step("zoom")
def _zoom(ctx, h0, h1, *, at, seconds, steps, hold, max_s):
    if (seconds is None) == (steps is None):
        raise ValueError("zoom takes seconds or steps, one of them")
    x, z = at

    def place(h):
        _pose(ctx, (x, h, z - 0.3 * h), (x, 0.0, z))

    if steps is not None:
        for k in range(steps):
            place(h0 * (h1 / h0) ** (k / max(steps - 1, 1)))
            yield
            if hold is not None:
                yield from _wait(ctx, hold, max_s)
        return
    t0 = time.monotonic()
    while (t := (time.monotonic() - t0) / seconds) < 1.0:
        place(h0 * (h1 / h0) ** t)
        yield
    place(h1)


@engine_step("hold")
def _hold(ctx, *, seconds, frames, until, max_s):
    if frames is not None:
        for _ in range(frames):
            yield
    elif seconds is not None:
        t0 = time.monotonic()
        while time.monotonic() - t0 < seconds:
            yield
    elif until is not None:
        yield from _wait(ctx, until, max_s)
    else:
        raise ValueError("hold takes seconds, frames or until")


@engine_step("each_frame")
def _each_frame(ctx, fn, *, frames, until, max_s):
    t0, n = time.monotonic(), 0
    while (frames is None or n < frames) and not (until is not None and until()):
        if frames is None and time.monotonic() - t0 > max_s:
            ctx.timeout(max_s)
        fn(ctx.app)
        yield
        n += 1


@engine_step("set")
def _set(ctx, patch):
    from manifoldx.devchannel import merge_patch

    merge_patch(ctx.engine.state, patch)


@engine_step("emit")
def _emit(ctx, event, payload=None):
    ctx.engine.emit(event, payload)


@engine_step("mark")
def _mark(ctx, name):
    ctx.run.mark(name)


@engine_step("shot")
def _shot(ctx, name):
    ctx.run.shot(name)
