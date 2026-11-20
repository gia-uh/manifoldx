"""Checks: what must hold for a run to pass, evaluated after the run has closed.

A check is a set of budgets, each an upper bound on one number of a phase summary, or a function of
the run that returns true when it holds. A scenario declares them with `name.check(...)`, a project
adds them in its budgets file, and `evaluate` turns them into one result each:

    @bench.scenario("far_walk")
    def far_walk(engine, p): ...

    far_walk.check(origin_max_m=20_000, phase="far")
    far_walk.check(frames_p95_ms=40, on="gpu")      # skipped on a software adapter

A budget name is a key of the phase summary (`p95_ms`, `origin_max_m`, `rss_max_mb`), or a probe and a
key: `frames_p95_ms`, `counters_draws` (its mean), `streamer_settled_s`, `systems_<name>`, `spans_<name>`,
`split_<name>`, `app_<metric>` (its mean), `workers_<kind>` (its p50 ms).
"""

from __future__ import annotations

import tomllib

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np

# The phase keys measured per frame: a budget on one of them names the slowest frame of the phase.
FRAME_KEYS = {"p50_ms", "p95_ms", "p99_ms", "max_ms", "over_1_5x", "over_3x", "draw_p50_ms", "cpu_p50_ms"}


@dataclass
class Check:
    """Budgets (name -> upper bound) or a function `fn(RunView) -> bool`, in one phase or in every one.

    `on="gpu"` skips the check on a software adapter, where timings say nothing about a GPU."""

    name: str
    budgets: dict = field(default_factory=dict)
    fn: Callable | None = None
    phase: str | None = None
    on: str | None = None


@dataclass
class RunView:
    """What a function check sees: the summary (run.json), the per-frame arrays and the events."""

    summary: dict
    frames: dict
    events: list

    @property
    def phases(self) -> dict:
        return self.summary["phases"]


def _number(v):
    return isinstance(v, (int, float, np.integer, np.floating)) and not isinstance(v, bool)


def lookup(ph: dict, name: str) -> float | None:
    """The number a budget name reads in one phase summary, or None when the phase has no such number."""
    if _number(ph.get(name)):
        return float(ph[name])
    probe, _, key = name.partition("_")
    try:
        if probe == "frames":
            v = ph[key]
        elif probe == "counters":
            v = ph["counters"][key]["mean"]
        elif probe == "streamer":
            v = ph["streamer"][key]
        elif probe in ("systems", "spans", "split"):
            v = ph[f"{probe}_ms"][key]
        elif probe == "app":
            v = ph["app"][key]["mean"]
        elif probe == "workers":
            v = ph["workers"][key]["p50_ms"]
        else:
            return None
    except (KeyError, TypeError):
        return None
    return float(v) if _number(v) else None


def names(ph: dict) -> list[str]:
    """Every budget name that reads a number in this phase."""
    out = [k for k, v in ph.items() if _number(v)]
    for probe, key, inner in (("counters", "counters", "mean"), ("app", "app", "mean"), ("workers", "workers", "p50_ms")):
        out += [f"{probe}_{k}" for k, v in (ph.get(key) or {}).items() if isinstance(v, dict) and _number(v.get(inner))]
    out += [f"streamer_{k}" for k, v in (ph.get("streamer") or {}).items() if _number(v)]
    for probe in ("systems", "spans", "split"):
        out += [f"{probe}_{k}" for k in ph.get(f"{probe}_ms") or {}]
    return out


def _frame(run, phase_index: int, name: str) -> int | None:
    """The global index of the frame that set the number: the slowest one, or the farthest origin sample."""
    rows = np.flatnonzero(np.asarray(run.frames.get("phase", [])) == phase_index)
    key = name.removeprefix("frames_")
    if not len(rows):
        return None
    if key in FRAME_KEYS:
        return int(rows[np.argmax(np.asarray(run.frames["wall_ms"])[rows])])
    if key == "origin_max_m" and "origin_m" in run.frames:
        far = np.asarray(run.frames["origin_m"], dtype=float)[rows]
        return None if np.isnan(far).all() else int(rows[np.nanargmax(far)])
    return None


def _result(check, phase, ok, message, value=None, budget=None, frame=None, skipped=False) -> dict:
    return {"check": check.name, "phase": phase, "ok": ok, "skipped": skipped, "value": value, "budget": budget,
            "frame": frame, "message": message}


def evaluate(checks: list[Check], run) -> list[dict]:
    """One result per budget and phase, and one per function check. Never raises: a check that does is a
    failed result with what it raised."""
    out = []
    for check in checks:
        try:
            out += _evaluate(check, run)
        except Exception as e:
            out.append(_result(check, check.phase, False, f"check {check.name}: raised {type(e).__name__}: {e}"))
    return out


def _evaluate(check: Check, run) -> list[dict]:
    summary = run.summary
    phases = summary.get("phases", {})
    host = summary.get("host", {})
    if check.on == "gpu" and host.get("software"):
        return [_result(check, check.phase, True, f"check {check.name}: skipped on a software adapter "
                                                  f"({host.get('adapter')})", skipped=True)]
    if check.phase is not None and check.phase not in phases:
        return [_result(check, check.phase, False, f"check {check.name}: no phase {check.phase}; "
                                                   f"this run has {', '.join(phases) or 'none'}")]
    out = []
    if check.fn is not None:
        try:
            ok = bool(check.fn(RunView(summary, run.frames, run.events)))
            message = f"check {check.name}: " + ("holds" if ok else "returned false")
        except Exception as e:
            ok, message = False, f"check {check.name}: raised {type(e).__name__}: {e}"
        out.append(_result(check, check.phase, ok, message))
    order = summary.get("phase_order", list(phases))
    for phase in [check.phase] if check.phase is not None else list(phases):
        ph = phases[phase]
        for name, budget in check.budgets.items():
            value = lookup(ph, name)
            if value is None:
                why = "was not measured" if name in ph else f"is not a number of this phase; this phase has " \
                                                            f"{', '.join(names(ph))}"
                out.append(_result(check, phase, False, f"check {check.name}: {name} {why} (phase {phase})",
                                   budget=budget))
                continue
            frame = _frame(run, order.index(phase), name) if phase in order else None
            ok = value <= budget
            where = f"in phase {phase}" + (f" (frame {frame})" if frame is not None else "")
            out.append(_result(check, phase, ok, f"check {check.name}: {name} {value:g} {'<=' if ok else '>'} "
                                                 f"{budget:g} {where}", value, budget, frame))
    return out


def check_name(fn: Callable | None, budgets: dict) -> str:
    """A check is named after its budgets, or after its function."""
    return ", ".join(budgets) if budgets else getattr(fn, "__name__", "check")



def read_budgets(path: Path) -> dict[str, list[Check]]:
    """A budgets file: one TOML table per scenario; its keys are budgets, besides `phase` and `on`."""
    data = tomllib.loads(Path(path).read_text())
    out = {}
    for scenario, table in data.items():
        if not isinstance(table, dict):
            raise ValueError(f"{path}: {scenario} must be a table, as [{scenario}]")
        budgets = {k: v for k, v in table.items() if k not in ("phase", "on")}
        bad = [k for k, v in budgets.items() if not _number(v)]
        if bad:
            raise ValueError(f"{path}: [{scenario}] {', '.join(bad)} must be numbers")
        out[scenario] = [Check(check_name(None, budgets), budgets, None, table.get("phase"), table.get("on"))]
    return out
