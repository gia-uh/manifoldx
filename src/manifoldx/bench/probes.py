"""What a run records each frame, and the summaries made from it.

frames: wall time (start of one frame to the next), CPU time of the engine's frame, the phase.
split: the engine's frame_split (update, acquire, render, overlap) in ms.
systems: the engine's system_split, each system's ms.
host: CPUs, the load over the run, the GPU adapter and whether it is a software rasterizer.
"""

from __future__ import annotations

import os
import re
import time

import numpy as np

SOFTWARE = re.compile(r"llvmpipe|lavapipe|swiftshader|softpipe", re.I)


class Frames:
    def __init__(self):
        self.wall_ms, self.cpu_ms, self.phase = [], [], []

    def add(self, wall_ms: float, cpu_ms: float, phase: int):
        self.wall_ms.append(wall_ms)
        self.cpu_ms.append(cpu_ms)
        self.phase.append(phase)

    def arrays(self) -> dict:
        return {"wall_ms": np.array(self.wall_ms), "cpu_ms": np.array(self.cpu_ms),
                "phase": np.array(self.phase, dtype=np.int32)}


class Split:
    """Per-frame dicts of name -> ms, kept as columns (a name missing in a frame counts 0)."""

    def __init__(self):
        self.rows: list[dict] = []

    def add(self, seconds: dict):
        self.rows.append({k: 1000.0 * float(v) for k, v in (seconds or {}).items()})

    def names(self) -> list[str]:
        out: dict[str, None] = {}
        for r in self.rows:
            out.update(dict.fromkeys(r))
        return list(out)

    def matrix(self) -> np.ndarray:
        names = self.names()
        return np.array([[r.get(n, 0.0) for n in names] for r in self.rows]).reshape(len(self.rows), len(names))


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
                "software": software}


def _load():
    try:
        return round(os.getloadavg()[0], 2)
    except OSError:  # not on this platform
        return None


def phase_summary(wall: np.ndarray, cpu: np.ndarray, split: Split, systems: Split, rows: np.ndarray) -> dict:
    """The numbers of one phase from the frames in `rows` (a boolean mask)."""
    w, c = wall[rows], cpu[rows]
    med = float(np.median(w))
    out = {
        "frames": int(rows.sum()),
        "p50_ms": round(med, 3),
        "p95_ms": round(float(np.percentile(w, 95)), 3),
        "p99_ms": round(float(np.percentile(w, 99)), 3),
        "max_ms": round(float(w.max()), 3),
        "over_1_5x": int((w > 1.5 * med).sum()),
        "over_3x": int((w > 3.0 * med).sum()),
        "cpu_p50_ms": round(float(np.median(c)), 3),
    }
    idx = np.flatnonzero(rows)
    for key, probe, top in (("split_ms", split, None), ("systems_ms", systems, 10)):
        names, m = probe.names(), probe.matrix()
        means = {n: round(float(m[idx, k].mean()), 3) for k, n in enumerate(names)} if len(names) else {}
        if top:
            means = dict(sorted(means.items(), key=lambda kv: -kv[1])[:top])
        out[key] = means
    return out
