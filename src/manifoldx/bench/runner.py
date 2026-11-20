"""The frame loop of `manifoldx bench run`: build the app, walk its steps, render, record.

Frames are drawn offscreen with `Engine._draw_frame` and `_render_canvas.draw()`, as the frame
benchmarks do. `loop="pipelined"` runs next frame's systems while this one renders (on a
free-threaded build); `loop="check"` adds MANIFOLDX_PIPELINE=check, which raises when a system
writes what the render reads. A run always ends with its numbers, even when a step fails or the
person presses Ctrl-C.
"""

from __future__ import annotations

import importlib.metadata
import math
import os
import tempfile
import time
import traceback
from dataclasses import dataclass, field
from datetime import UTC, datetime

import numpy as np

from manifoldx import counters
from manifoldx.bench.probes import (Columns, Frames, Host, Memory, jobs_summary, last_metrics, origin_distance,
                                   phase_summary, stream_summary)
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
        self.spans = [(0, 0.0)]  # (phase index, start time): the run's stretches, one per mark

    def event(self, kind, **kw):
        self.events.append({"t": round(counters.now(), 5), "kind": kind, **kw})

    def mark(self, name):
        if name not in self.phases:
            self.phases.append(name)
        self.phase = self.phases.index(name)
        self.spans.append((self.phase, counters.now()))
        self.event("mark", name=name)

    def intervals(self, end: float) -> dict:
        """Phase index -> the [start, end) times it ran over."""
        out: dict = {}
        for (k, t0), nxt in zip(self.spans, [*self.spans[1:], (None, end)]):
            out.setdefault(k, []).append((t0, nxt[1]))
        return out

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
                 app_name: str | None = None, progress=None, folders: dict | None = None) -> Run:
    """Run one scenario and return what it recorded. `progress(frames, phase)` is called once a frame.
    `folders` maps names to folders the app fills (caches, exports), sized at the start and the end."""
    import manifoldx as mx

    if loop not in LOOPS:
        raise ValueError(f"loop must be one of {', '.join(LOOPS)}, not {loop!r}")
    params = Params(sc.params, overrides)
    w, h = size
    host, frames = Host(), Frames()
    settled, ends = [], []  # per frame: every streamer of the run settled; the frame's end time
    origin = []  # per frame: the farthest thing drawn from the origin, every 30 frames and at a phase's start
    memory = Memory(folders)
    streamers = []  # the streamers made during this run
    probes = {"split": Columns(scale=1000.0), "systems": Columns(scale=1000.0), "counters": Columns(),
              "spans": Columns(), "metrics": Columns(fill=None)}
    started, t_start = datetime.now(UTC), time.monotonic()
    saved_env = os.environ.get("MANIFOLDX_PIPELINE")
    if loop == "check":
        os.environ["MANIFOLDX_PIPELINE"] = "check"
    engine = app = None
    state = {"last": None}
    error = None

    def frame(keep=False):
        counters.frame = len(frames)
        t0, c0 = time.perf_counter(), time.process_time()
        engine._draw_frame()
        draw = time.perf_counter() - t0
        img = engine._render_canvas.draw()
        end, cpu = time.perf_counter(), time.process_time() - c0
        wall = end - (state["last"] if state["last"] is not None else t0)
        state["last"] = end
        ctx.dt = step_dt or wall
        took = counters.take()
        ready = all(s.settled for s in streamers) if streamers else False
        t_end = counters.now()
        k = len(frames)
        far = origin_distance(engine) if k % 30 == 0 or (k and frames.phase[-1] != rec.phase) else math.nan
        # Everything is measured before anything is stored; a Ctrl-C between two stores is cut at the end.
        frames.add(1000.0 * wall, 1000.0 * draw, 1000.0 * cpu, rec.phase)
        probes["split"].add(engine.frame_split)
        probes["systems"].add(engine.system_split)
        probes["counters"].add(took.counts)
        probes["spans"].add(took.spans)
        probes["metrics"].add(took.metrics)
        settled.append(ready)
        ends.append(t_end)
        origin.append(far)
        memory.frame()
        host.tick()
        if progress:
            progress(len(frames.wall_ms), rec.phases[rec.phase])
        if keep:
            from manifoldx.engine import _drop_alpha

            return _drop_alpha(np.asarray(img)).copy()
        return None

    rec = _Recorder(frame)
    ctx = None
    from manifoldx import streaming

    old = {id(s) for s in streaming.live()}
    counters.start(tempfile.mkdtemp(prefix="manifoldx-bench-"))  # worker processes inherit its path
    try:
        engine = mx.Engine(sc.name, width=w, height=h)
        app, steps = sc.fn(engine, params)
        streamers[:] = [s for s in streaming.live() if id(s) not in old]
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
        _close(app, [s for s in streaming.live() if id(s) not in old])
        ended = counters.now()
        nodes = [e for e in counters.events() if e["kind"] == "node"]
        jobs = [{"kind": "job", **{("job" if k == "kind" else k): v for k, v in j.items()}} for j in counters.jobs()]
        counters.stop()
        memory.stop()
        if engine is not None:
            engine.close()  # fires 'shutdown' and frees the engine's pools and GPU device
        if loop == "check":
            if saved_env is None:
                os.environ.pop("MANIFOLDX_PIPELINE", None)
            else:
                os.environ["MANIFOLDX_PIPELINE"] = saved_env
    if error == "interrupted":
        rec.event("error", name=error)

    n = min(len(frames), len(settled), len(ends), len(origin), len(memory), *(len(pr) for pr in probes.values()))
    frames.cut(n)
    memory.cut(n)
    for pr in probes.values():
        pr.cut(n)
    del settled[n:], ends[n:], origin[n:]
    arrays = {**frames.arrays(), **memory.arrays(), "origin_m": np.array(origin)}
    intervals = rec.intervals(ended)
    phases = {}
    for k, name in enumerate(rec.phases):
        rows = arrays["phase"] == k
        if rows.any():
            phases[name] = phase_summary(arrays, probes, rows)
            far = arrays["origin_m"][rows]
            phases[name]["origin_max_m"] = None if np.isnan(far).all() else round(float(np.nanmax(far)), 2)
            phases[name]["gc_ms"] = round(float(arrays["gc_ms"][rows].sum()), 3)
            sampled = arrays["rss_mb"][rows]
            phases[name]["rss_max_mb"] = None if np.isnan(sampled).all() else round(float(np.nanmax(sampled)), 1)
            if streamers or nodes:
                phases[name]["streamer"] = stream_summary(nodes, intervals[k], ends, settled, rows)
            if jobs:
                phases[name]["workers"] = jobs_summary(jobs, intervals[k])
    rec.events = sorted([*rec.events, *nodes, *jobs], key=lambda e: e["t"])
    for key, names, array, dtype in (("split", "split_names", "split_ms", float),
                                     ("systems", "systems_names", "systems_ms", float),
                                     ("counters", "counter_names", "counters", np.int64),
                                     ("spans", "span_names", "spans_ms", float),
                                     ("metrics", "metric_names", "metrics", float)):
        arrays[names] = np.array(probes[key].names(), dtype=str)
        arrays[array] = probes[key].matrix(dtype)
    summary = {
        "scenario": sc.name, "params": params.as_dict(), "mode": "offscreen", "loop": loop,
        "step": step_dt, "size": [w, h], "smoke": smoke, "label": label,
        "versions": {"manifoldx": _version("manifold-gfx"), "app": _version(app_name)},
        "host": host.summary(), "phases": phases, "phase_order": rec.phases, "error": error,
        "started": started.isoformat(timespec="seconds"), "seconds": round(time.monotonic() - t_start, 3),
        "app": last_metrics(probes["metrics"]), "counters_exact": loop == "serial", "memory": memory.summary(),
    }
    return Run(summary, arrays, rec.events, rec.shots, error)


def _close(app, streamers):
    """Close the app, then every streamer the run made that is still open, so their worker pools end
    with the run, wherever the app kept them."""
    from manifoldx.streaming import live

    for k, obj in enumerate([app, *streamers]):
        if obj is None or (k and obj not in live()):  # closed by the app already
            continue
        close = getattr(obj, "close", None)
        if callable(close):
            try:
                close()
            except Exception:
                traceback.print_exc(limit=3)
