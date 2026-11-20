"""What a run records each frame, and the summaries made from it.

frames: wall time (start of one frame to the next), the engine's frame (`draw`), the process's CPU
time over the frame (every thread), the phase.
split: the engine's frame_split (update, acquire, render, overlap) in ms.
systems: the engine's system_split, each system's ms.
counters: what manifoldx.counters counted in the frame (draws, bytes uploaded, ...).
spans: ms of each bench.span, named <system>/<span>.
metrics: the app's bench.metric values.
memory: RSS and GPU memory once a second, garbage collections per frame, the app's folders at start and end.
origin: the farthest anything drawn is from the floating origin, every 30 frames and at each phase's start.
host: CPUs, the load over the run, the GPU adapter and whether it is a software rasterizer, Python.
"""

from __future__ import annotations

import gc
import math
import os
import platform
import re
import sys
import time

from pathlib import Path

import numpy as np

from manifoldx.devchannel import _rss_mb as rss_mb

SOFTWARE = re.compile(r"llvmpipe|lavapipe|swiftshader|softpipe", re.I)


class Frames:
    def __init__(self):
        self.wall_ms, self.draw_ms, self.cpu_ms, self.phase = [], [], [], []

    def add(self, wall_ms: float, draw_ms: float, cpu_ms: float, phase: int):
        self.wall_ms.append(wall_ms)
        self.draw_ms.append(draw_ms)
        self.cpu_ms.append(cpu_ms)
        self.phase.append(phase)

    def __len__(self):
        return min(len(self.wall_ms), len(self.draw_ms), len(self.cpu_ms), len(self.phase))

    def cut(self, n: int):
        for col in (self.wall_ms, self.draw_ms, self.cpu_ms, self.phase):
            del col[n:]

    def arrays(self) -> dict:
        return {"wall_ms": np.array(self.wall_ms), "draw_ms": np.array(self.draw_ms),
                "cpu_ms": np.array(self.cpu_ms), "phase": np.array(self.phase, dtype=np.int32)}


class Columns:
    """Per-frame dicts of name -> number, kept as columns; a name missing in a frame counts `fill`."""

    def __init__(self, fill=0.0, scale=1.0):
        self.rows: list[dict] = []
        self.fill, self.scale = fill, scale

    def add(self, row: dict):
        self.rows.append({k: v * self.scale if self.scale != 1.0 else v for k, v in (row or {}).items()})

    def __len__(self):
        return len(self.rows)

    def cut(self, n: int):
        del self.rows[n:]

    def names(self) -> list[str]:
        out: dict[str, None] = {}
        for r in self.rows:
            out.update(dict.fromkeys(r))
        return list(out)

    def matrix(self, dtype=float) -> np.ndarray:
        names = self.names()
        return np.array([[_number(r.get(n, self.fill)) for n in names] for r in self.rows],
                        dtype=dtype).reshape(len(self.rows), len(names))


def _number(v):
    return float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else math.nan


class Host:
    def __init__(self):
        self.load_before = _load()
        self.loads = [] if self.load_before is None else [self.load_before]
        self._sampled = time.monotonic()
        self.adapter, self.adapter_type = None, None

    def tick(self):
        """Called each frame; samples the load once a second."""
        now = time.monotonic()
        if now - self._sampled >= 1.0:
            self._sampled = now
            if (x := _load()) is not None:
                self.loads.append(x)

    def read_adapter(self, engine):
        info = getattr(getattr(engine, "_adapter", None), "info", None) or {}
        self.adapter = info.get("device") or info.get("description") or None
        self.adapter_type = info.get("adapter_type") or None

    def summary(self) -> dict:
        after = _load()
        software = (str(self.adapter_type or "").upper() == "CPU") or bool(SOFTWARE.search(self.adapter or ""))
        return {"cpus": os.cpu_count(), "load_before": self.load_before,
                "load_max": max(self.loads + ([after] if after is not None else []), default=None),
                "load_after": after, "adapter": self.adapter, "adapter_type": self.adapter_type,
                "software": software, "python": platform.python_version(),
                "free_threaded": not sys._is_gil_enabled()}


def _load():
    try:
        return round(os.getloadavg()[0], 2)
    except OSError:  # not on this platform
        return None


def phase_summary(arrays: dict, probes: dict, rows: np.ndarray) -> dict:
    """The numbers of one phase from the frames in `rows` (a boolean mask).

    `probes` holds the Columns probes by name: split, systems, counters, spans, metrics."""
    w = arrays["wall_ms"][rows]
    med = float(np.median(w))
    out = {
        "frames": int(rows.sum()),
        "p50_ms": round(med, 3),
        "p95_ms": round(float(np.percentile(w, 95)), 3),
        "p99_ms": round(float(np.percentile(w, 99)), 3),
        "max_ms": round(float(w.max()), 3),
        "over_1_5x": int((w > 1.5 * med).sum()),
        "over_3x": int((w > 3.0 * med).sum()),
        "draw_p50_ms": round(float(np.median(arrays["draw_ms"][rows])), 3),
        "cpu_p50_ms": round(float(np.median(arrays["cpu_ms"][rows])), 3),
    }
    idx = np.flatnonzero(rows)
    for key, name, top in (("split_ms", "split", None), ("systems_ms", "systems", 10), ("spans_ms", "spans", None)):
        probe = probes[name]
        names, m = probe.names(), probe.matrix()
        means = {n: round(float(m[idx, k].mean()), 3) for k, n in enumerate(names)} if len(names) else {}
        if top:
            means = dict(sorted(means.items(), key=lambda kv: -kv[1])[:top])
        out[key] = means
    names, m = probes["counters"].names(), probes["counters"].matrix()
    out["counters"] = {n: {"mean": round(float(m[idx, k].mean()), 3), "max": int(m[idx, k].max())}
                       for k, n in enumerate(names)}
    out["app"] = _metrics_summary(probes["metrics"], idx)
    return out


def _metrics_summary(probe: Columns, idx: np.ndarray) -> dict:
    """Per metric set in these frames: last, and mean and max when it is a number."""
    out = {}
    rows = [probe.rows[i] for i in idx]
    for name in probe.names():
        vals = [r[name] for r in rows if name in r]
        if not vals:
            continue
        nums = [v for v in vals if isinstance(v, (int, float)) and not isinstance(v, bool)]
        out[name] = {"last": vals[-1]}
        if len(nums) == len(vals):
            out[name].update(mean=round(sum(nums) / len(nums), 3), max=max(nums))
    return out


def last_metrics(probe: Columns) -> dict:
    """The last value each metric was set to over the run."""
    out = {}
    for r in probe.rows:
        out.update(r)
    return out


def _inside(t: float, intervals) -> bool:
    return any(a <= t < b for a, b in intervals)


def stream_summary(nodes: list, intervals, ends: list, settled: list, rows: np.ndarray) -> dict:
    """What the streamers did in one phase: node events by kind and level, the time from the phase's start
    to the first node drawn, and to the frame where every streamer settled for the last time in the phase:
    the first settled frame after the last one that was not (None when the phase ends unsettled)."""
    start = intervals[0][0]
    counts: dict = {}
    first_drawn = None
    for e in nodes:
        if not _inside(e["t"], intervals):
            continue
        by_level = counts.setdefault(e["ev"], {})
        by_level[e["level"]] = by_level.get(e["level"], 0) + 1
        if e["ev"] == "drawn" and first_drawn is None:
            first_drawn = round(e["t"] - start, 4)
    idx = np.flatnonzero(rows)
    busy = [i for i in idx if not settled[i]]
    after = idx[idx > busy[-1]] if busy else idx
    done = ends[after[0]] if len(after) else None
    return {**counts, "first_drawn_s": first_drawn,
            "settled_s": None if done is None else round(done - start, 4)}


def jobs_summary(jobs: list, intervals) -> dict:
    """Worker jobs that started in one phase, by kind: how many, their p50 and max ms, how many processes."""
    out = {}
    by_kind: dict = {}
    for j in jobs:
        if _inside(j["t"], intervals):
            by_kind.setdefault(j["job"], []).append(j)
    for kind, js in by_kind.items():
        ms = np.array([j["ms"] for j in js])
        out[kind] = {"jobs": len(js), "p50_ms": round(float(np.median(ms)), 3), "max_ms": round(float(ms.max()), 3),
                     "processes": len({j["pid"] for j in js})}
    return out


def gpu_mb() -> float | None:
    """What this process allocated through wgpu, in MB, or None when wgpu does not say."""
    try:
        import wgpu

        counts = wgpu.diagnostics.object_counts.get_dict()
    except Exception:
        return None
    mem = [v["resource_mem"] for v in counts.values() if isinstance(v, dict) and "resource_mem" in v]
    return sum(mem) / 2**20 if mem else None


def folder_mb(path: Path) -> float:
    total = 0
    for root, _dirs, files in os.walk(path):
        for name in files:
            try:
                total += os.stat(os.path.join(root, name)).st_size
            except OSError:
                pass
    return total / 2**20


class Memory:
    """RSS and GPU memory sampled once a second, garbage collections timed as they happen, folder sizes."""

    def __init__(self, folders: dict | None = None):
        self.folders = {k: Path(v) for k, v in (folders or {}).items()}
        self.start_sizes = {k: folder_mb(v) for k, v in self.folders.items()}
        self.rss, self.gpu = [rss_mb()], [gpu_mb()]
        self.rss_mb, self.gpu_mb, self.gc_ms = [], [], []  # per frame; NaN where not sampled
        self.runs, self.total, self.longest = [0, 0, 0], 0.0, 0.0
        self._at, self._frame_ms, self._sampled = None, 0.0, time.monotonic()
        gc.callbacks.append(self._gc)

    def _gc(self, phase, info):
        if phase == "start":
            self._at = time.perf_counter()
        elif self._at is not None:
            ms, self._at = 1000.0 * (time.perf_counter() - self._at), None
            self.runs[info["generation"]] += 1
            self.total += ms
            self.longest = max(self.longest, ms)
            self._frame_ms += ms

    def frame(self):
        """After each frame: the frame's collections, and a sample when a second has passed."""
        self.gc_ms.append(self._frame_ms)
        self._frame_ms = 0.0
        now = time.monotonic()
        if now - self._sampled >= 1.0 or len(self.gc_ms) == 1:
            self._sampled = now
            r, g = rss_mb(), gpu_mb()
            self.rss.append(r)
            self.gpu.append(g)
            self.rss_mb.append(r)
            self.gpu_mb.append(math.nan if g is None else g)
        else:
            self.rss_mb.append(math.nan)
            self.gpu_mb.append(math.nan)

    def __len__(self):
        return min(len(self.gc_ms), len(self.rss_mb), len(self.gpu_mb))

    def cut(self, n: int):
        del self.gc_ms[n:], self.rss_mb[n:], self.gpu_mb[n:]

    def stop(self):
        if self._gc in gc.callbacks:
            gc.callbacks.remove(self._gc)
        self.rss.append(rss_mb())
        self.gpu.append(gpu_mb())

    def summary(self) -> dict:
        gpus = [g for g in self.gpu if g is not None]
        return {
            "rss_mb": {"start": round(self.rss[0], 1), "max": round(max(self.rss), 1), "end": round(self.rss[-1], 1)},
            "gc": {"runs": list(self.runs), "ms": round(self.total, 3), "max_ms": round(self.longest, 3)},
            "gpu_mb": None if not gpus else {"start": round(gpus[0], 2), "max": round(max(gpus), 2),
                                             "end": round(gpus[-1], 2)},
            "folders_mb": {k: {"start": round(self.start_sizes[k], 3), "end": round(folder_mb(v), 3)}
                           for k, v in self.folders.items()},
        }

    def arrays(self) -> dict:
        return {"rss_mb": np.array(self.rss_mb), "gpu_mb": np.array(self.gpu_mb), "gc_ms": np.array(self.gc_ms)}


def origin_distance(engine) -> float:
    """The largest distance from the floating origin of an entity drawn with a mesh, in metres.

    Transform rows already hold positions relative to the origin."""
    store = engine.store
    if "Transform" not in store._components or "Mesh" not in store._components:
        return 0.0
    alive = np.flatnonzero(store._alive)
    if not len(alive):
        return 0.0
    mesh = store.get_component_data("Mesh", alive)
    rows = alive[mesh[:, 0] != 0] if mesh is not None else alive[:0]
    if not len(rows):
        return 0.0
    return float(np.linalg.norm(store._components["Transform"][rows, 0:3].astype(np.float64), axis=1).max())
