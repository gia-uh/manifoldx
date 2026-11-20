"""Micro cases: functions of one size `n`, timed at several sizes, for manifoldx and for apps.

Timing helpers and the case registry shared by every suite.

A case is a function of one size `n` that returns milliseconds, or a dict with
"ms" and any extra numbers worth printing (calls per frame, bytes moved). Cases
register with `@case(suite, name, sizes=..., smoke=...)`; `smoke` is the tiny
size the test suite runs so cases do not rot.
"""

from __future__ import annotations

import cProfile
import io
import pstats
import time
import timeit
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

CASES: dict[str, list["Case"]] = {}


@dataclass
class Case:
    suite: str
    name: str
    fn: Callable[[int], float | dict]
    sizes: list[int]
    smoke: int
    doc: str = ""


@dataclass
class Result:
    suite: str
    case: str
    n: int
    ms: float | None
    extra: dict = field(default_factory=dict)
    error: str | None = None


def case(suite: str, name: str, *, sizes: list[int], smoke: int):
    """Register `fn(n)` as case `name` of `suite`, timed at each of `sizes`."""

    def wrap(fn):
        CASES.setdefault(suite, []).append(
            Case(suite, name, fn, list(sizes), smoke, (fn.__doc__ or "").strip().split("\n")[0])
        )
        return fn

    return wrap


# Repeats of `measure`; the test suite lowers them to 1 so smoke runs stay quick.
REPEAT = 5


def measure(fn: Callable[[], object], repeat: int | None = None, number: int | None = None) -> float:
    """Best of `repeat` runs, in ms per call. Each run makes `number` calls, picked so
    a run lasts at least 0.2 s (at most 20 calls) when not given."""
    repeat = REPEAT if repeat is None else repeat
    if number is None:
        number = max(1, min(timeit.Timer(fn).autorange()[0], 20)) if REPEAT > 1 else 1
    return 1e3 * min(timeit.repeat(fn, number=number, repeat=repeat)) / number


def headless_engine(max_entities: int, size: int = 64):
    """An Engine for timing: small target, so the software rasterizer of a GPU-less
    machine adds little to the CPU-side cost being measured."""
    import manifoldx as mx

    return mx.Engine("bench", width=size, height=size, max_entities=max(max_entities, 16))


def start(engine) -> None:
    """Give a populated engine its offscreen canvas and a fixed timestep."""
    engine._ensure_offscreen(1)
    engine.set_fixed_timestep(1 / 30)


def frame_ms(engine, frames: int = 5, warmup: int = 2) -> float:
    """Median CPU time of Engine._draw_frame (simulation, render prep, encoding,
    submit) in ms. canvas.draw(), which waits for the GPU and reads back, runs
    between frames and is not counted."""
    for _ in range(warmup):
        engine._draw_frame()
        engine._render_canvas.draw()
    times = []
    for _ in range(frames if REPEAT > 1 else 1):
        t0 = time.perf_counter()
        engine._draw_frame()
        times.append(time.perf_counter() - t0)
        engine._render_canvas.draw()
    return 1e3 * float(np.median(times))


@contextmanager
def count_calls(owner, name: str):
    """Count calls of `owner.name` (a function or method) inside the block.
    Yields a dict with "calls"; a missing attribute counts nothing."""
    counter = {"calls": 0}
    original = getattr(owner, name, None)
    if original is None:
        yield counter
        return

    def counted(*args, **kwargs):
        counter["calls"] += 1
        return original(*args, **kwargs)

    setattr(owner, name, counted)
    try:
        yield counter
    finally:
        setattr(owner, name, original)


def profile(fn: Callable[[], object], top: int = 15) -> str:
    """cProfile one call of `fn`; the `top` functions by own time."""
    pr = cProfile.Profile()
    pr.enable()
    fn()
    pr.disable()
    out = io.StringIO()
    pstats.Stats(pr, stream=out).sort_stats("tottime").print_stats(top)
    return out.getvalue()


# --- running cases -------------------------------------------------------------------------------


def load(modules: list[str]) -> list[Case]:
    """Import each module (its @case functions register) and return their cases, in module order."""
    import importlib

    for m in modules:
        importlib.import_module(m)
    every = [c for cases in CASES.values() for c in cases]
    return [c for m in modules for c in every if c.fn.__module__ == m]


def run(cases, sizes=None, profile=False, smoke=False):
    """Time each case at its sizes (or `sizes`, or its smoke size); a case that raises is an error row."""
    import sys
    import traceback

    results = []
    for c in cases:
        for n in [c.smoke] if smoke else (sizes or c.sizes):
            try:
                out = c.fn(n)
                ms, extra = (out.pop("ms"), out) if isinstance(out, dict) else (out, {})
                r = Result(c.suite, c.name, n, float(ms), extra)
            except Exception as e:  # a case the code under test cannot run (an API it lacks)
                r = Result(c.suite, c.name, n, None, error=f"{type(e).__name__}: {e}")
                if not smoke:
                    traceback.print_exc(limit=3, file=sys.stderr)
            results.append(r)
            print(_row(r), flush=True)
        if profile:
            n = (sizes or c.sizes)[-1]
            print(f"--- profile {c.suite}.{c.name} n={n} (setup included)")
            print(globals()["profile"](lambda: c.fn(n)))
    return results


def _row(r):
    if r.error:
        return f"{r.suite:<9} {r.case:<22} n={r.n:<9} ERROR {r.error}"
    extra = "  ".join(f"{k}={v}" for k, v in r.extra.items())
    return f"{r.suite:<9} {r.case:<22} n={r.n:<9} {r.ms:12.3f} ms  {extra}"


def _fmt(ms):
    return "error" if ms is None else (f"{ms:.3f}" if ms < 1 else f"{ms:.1f}")


def compare(base, head, label):
    """Markdown table of base vs head results, matched on (suite, case, n)."""
    by_key = {(r["suite"], r["case"], r["n"]): r for r in base["results"]}
    lines = [
        f"| case | n | {label} ms | this copy ms | speed-up | |",
        "|---|---|---|---|---|---|",
    ]
    for r in head["results"]:
        b = by_key.get((r["suite"], r["case"], r["n"]))
        b_ms = b["ms"] if b else None
        ratio = f"{b_ms / r['ms']:.1f}×" if b_ms and r["ms"] else "–"
        extra = ", ".join(f"{k} {v}" for k, v in r["extra"].items())
        if b and b["extra"] != r["extra"] and b["extra"]:
            extra += " (was " + ", ".join(f"{k} {v}" for k, v in b["extra"].items()) + ")"
        lines.append(
            f"| {r['suite']}.{r['case']} | {r['n']} | {_fmt(b_ms) if b else 'n/a'} | {_fmt(r['ms'])} | {ratio} | {extra} |"
        )
    return "\n".join(lines)


# --- another copy of manifoldx -------------------------------------------------------------------


def _parse(argv):
    import argparse

    p = argparse.ArgumentParser(prog="manifoldx bench micro", add_help=False)
    p.add_argument("--case", action="append")
    p.add_argument("--sizes", nargs="+", type=int)
    p.add_argument("--smoke", action="store_true")
    return p.parse_args(argv)


def _child(modules, src, argv, out, cwd):
    """Run the cases in a fresh process that imports manifoldx from `src` and this file as the harness."""
    import json
    import os
    import subprocess
    import sys
    from pathlib import Path

    cmd = [sys.executable, str(Path(__file__).resolve()), "--child", str(out), *modules, "--", *argv]
    env = {**os.environ, "PYTHONPATH": os.pathsep.join([str(src), str(cwd)])}
    subprocess.run(cmd, cwd=cwd, env=env, check=True)
    data = json.loads(Path(out).read_text())
    imported = Path(data["manifoldx"]).resolve()
    if not imported.is_relative_to(Path(src).resolve()):  # the run must have measured the code asked for
        raise SystemExit(f"manifoldx bench micro: expected manifoldx from {src}, the run imported {imported}")
    return data


def against(modules: list[str], other, argv: list[str], cwd=None) -> str:
    """Run the cases on the manifoldx in `other` (a project directory holding src/manifoldx) and on this
    one, each in a fresh process, and return a Markdown table of both."""
    import tempfile
    from pathlib import Path

    other = Path(other)
    if not (other / "src" / "manifoldx" / "__init__.py").exists():
        raise SystemExit(
            f"--against needs a directory holding another manifoldx project (src/manifoldx); {other} has none. "
            "Make one with your version control, e.g. a worktree."
        )
    import manifoldx

    here = Path(manifoldx.__file__).resolve().parent.parent
    cwd = Path(cwd or Path.cwd())
    with tempfile.TemporaryDirectory(prefix="mx-micro-") as tmp:
        print(f"== {other} ==", flush=True)
        base = _child(modules, other / "src", argv, Path(tmp) / "base.json", cwd)
        print("== this copy ==", flush=True)
        head = _child(modules, here, argv, Path(tmp) / "head.json", cwd)
    return compare(base, head, other.name)


def _child_main(out, modules, argv):
    """In the child: make this file the harness the cases import, whatever manifoldx is on the path."""
    import json
    import os
    import platform
    import sys
    import types
    from dataclasses import asdict

    me = sys.modules[__name__]
    sys.modules["manifoldx.bench.micro"] = me
    import manifoldx

    if "manifoldx.bench" not in sys.modules:
        try:
            import manifoldx.bench  # noqa: F401
        except ImportError:  # an older manifoldx without this package
            pkg = types.ModuleType("manifoldx.bench")
            pkg.micro, pkg.case = me, case
            sys.modules["manifoldx.bench"] = pkg
    sys.modules["manifoldx.bench"].micro = me
    a = _parse(argv)
    cases = [c for c in load(modules) if not a.case or c.name in a.case]
    results = run(cases, a.sizes, smoke=a.smoke)
    with open(out, "w") as f:
        json.dump({
            "manifoldx": manifoldx.__file__,
            "machine": f"{platform.machine()} · {os.cpu_count()} cpus · Python {platform.python_version()}",
            "results": [asdict(r) for r in results],
        }, f, indent=1)


if __name__ == "__main__":
    import sys

    _, flag, out, *rest = sys.argv
    split = rest.index("--")
    _child_main(out, rest[:split], rest[split + 1:])
