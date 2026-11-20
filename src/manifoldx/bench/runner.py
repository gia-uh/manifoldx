"""The frame loop of `manifoldx bench run`: build the app, walk its steps, render, record.

Frames are drawn offscreen with `Engine._draw_frame` and `_render_canvas.draw()`, as the frame
benchmarks do. `loop="pipelined"` runs next frame's systems while this one renders (on a
free-threaded build); `loop="check"` adds MANIFOLDX_PIPELINE=check, which raises when a system
writes what the render reads. A run always ends with its numbers, even when a step fails or the
person presses Ctrl-C.
"""

from __future__ import annotations

import importlib.metadata
import os
import time
import traceback
from dataclasses import dataclass, field
from datetime import UTC, datetime

import numpy as np

from manifoldx.bench.probes import Frames, Host, Split, phase_summary
from manifoldx.bench.scenario import Ctx, Params, Scenario, iterate

LOOPS = ("serial", "pipelined", "check")


@dataclass
class Run:
    """What a run recorded: the summary (run.json), per-frame arrays, events, shots and the error."""

    summary: dict
    frames: dict = field(default_factory=dict)
    events: list = field(default_factory=list)
    shots: dict = field(default_factory=dict)
    error: str | None = None


class _Recorder:
    """The `run` an engine step sees: phases, shots and events."""

    def __init__(self, frame):
        self.phases, self.phase = ["main"], 0
        self.events, self.shots, self._frame = [], {}, frame
        self._t0 = time.monotonic()

    def event(self, kind, **kw):
        self.events.append({"t": round(time.monotonic() - self._t0, 4), "kind": kind, **kw})

    def mark(self, name):
        if name not in self.phases:
            self.phases.append(name)
        self.phase = self.phases.index(name)
        self.event("mark", name=name)

    def shot(self, name):
        self.shots[name] = self._frame(keep=True)
        self.event("shot", name=name)


def _version(dist):
    if not dist:
        return None
    try:
        return importlib.metadata.version(dist)
    except importlib.metadata.PackageNotFoundError:
        return None


def run_scenario(sc: Scenario, *, size=(640, 360), loop: str = "serial", step_dt: float | None = None,
                 overrides: dict | None = None, smoke: bool = False, label: str | None = None,
                 app_name: str | None = None, progress=None) -> Run:
    """Run one scenario and return what it recorded. `progress(frames, phase)` is called once a frame."""
    import manifoldx as mx

    if loop not in LOOPS:
        raise ValueError(f"loop must be one of {', '.join(LOOPS)}, not {loop!r}")
    params = Params(sc.params, overrides)
    w, h = size
    host, frames, split, systems = Host(), Frames(), Split(), Split()
    started, t_start = datetime.now(UTC), time.monotonic()
    saved_env = os.environ.get("MANIFOLDX_PIPELINE")
    if loop == "check":
        os.environ["MANIFOLDX_PIPELINE"] = "check"
    engine = app = None
    state = {"last": None}
    error = None

    def frame(keep=False):
        t0 = time.perf_counter()
        engine._draw_frame()
        cpu = time.perf_counter() - t0
        img = engine._render_canvas.draw()
        end = time.perf_counter()
        wall = end - (state["last"] if state["last"] is not None else t0)
        state["last"] = end
        ctx.dt = step_dt or wall
        frames.add(1000.0 * wall, 1000.0 * cpu, rec.phase)
        split.add(engine.frame_split)
        systems.add(engine.system_split)
        host.tick()
        if progress:
            progress(len(frames.wall_ms), rec.phases[rec.phase])
        if keep:
            from manifoldx.engine import _drop_alpha

            return _drop_alpha(np.asarray(img)).copy()
        return None

    rec = _Recorder(frame)
    ctx = None
    try:
        engine = mx.Engine(sc.name, width=w, height=h)
        app, steps = sc.fn(engine, params)
        engine._ensure_offscreen(1)
        host.read_adapter(engine)
        if step_dt:
            engine.set_fixed_timestep(step_dt)
        if loop != "serial":
            engine._pipeline_on = True
            engine._pipeline_frames = 0
        engine._fire_startup()
        ctx = Ctx(engine, app, rec)
        cap = sc.sizes.get("smoke") if smoke else None
        for i, call in enumerate(steps):
            ctx.index = i
            rec.event("step", name=call.name, index=i)
            n = 0
            gen = iterate(call, ctx)
            for _ in gen:
                frame()
                n += 1
                if cap is not None and n >= cap:
                    gen.close()
                    break
            rec.event("step_end", name=call.name, index=i, frames=n)
    except KeyboardInterrupt:
        error = "interrupted"
    except Exception as e:
        error = f"{type(e).__name__}: {e}"
        rec.event("error", name=error, traceback=traceback.format_exc(limit=8))
    finally:
        _close(app)
        if engine is not None:
            engine.close()  # fires 'shutdown' and frees the engine's pools and GPU device
        if loop == "check":
            if saved_env is None:
                os.environ.pop("MANIFOLDX_PIPELINE", None)
            else:
                os.environ["MANIFOLDX_PIPELINE"] = saved_env
    if error:
        rec.event("error", name=error)

    arrays = frames.arrays()
    phases = {}
    for k, name in enumerate(rec.phases):
        rows = arrays["phase"] == k
        if rows.any():
            phases[name] = phase_summary(arrays["wall_ms"], arrays["cpu_ms"], split, systems, rows)
    for key, probe in (("split", split), ("systems", systems)):
        arrays[f"{key}_names"] = np.array(probe.names(), dtype=str)
        arrays[f"{key}_ms"] = probe.matrix()
    summary = {
        "scenario": sc.name, "params": params.as_dict(), "mode": "offscreen", "loop": loop,
        "step": step_dt, "size": [w, h], "smoke": smoke, "label": label,
        "versions": {"manifoldx": _version("manifold-gfx"), "app": _version(app_name)},
        "host": host.summary(), "phases": phases, "phase_order": rec.phases, "error": error,
        "started": started.isoformat(timespec="seconds"), "seconds": round(time.monotonic() - t_start, 3),
        "app": {},
    }
    return Run(summary, arrays, rec.events, rec.shots, error)


def _close(app):
    """Close the app and the streamers it holds, so their worker pools end with the run."""
    from manifoldx.streaming import QuadStreamer, Streamer

    if app is None:
        return
    seen = [app, *(v for v in getattr(app, "__dict__", {}).values() if isinstance(v, (Streamer, QuadStreamer)))]
    for obj in seen:
        close = getattr(obj, "close", None)
        if callable(close):
            try:
                close()
            except Exception:
                traceback.print_exc(limit=3)
