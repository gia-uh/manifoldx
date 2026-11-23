"""S-a: does a slow async system move the frame?

Compares two paced 60 Hz loops under the same Python load:

- ``rust``: the spike loop in ``crates/spike-s-a``. It releases the GIL and calls
  one no-op Python sync system per frame. Async systems run on Rust threads and
  are never waited for.
- ``current``: the current engine, headless and pipelined (``_draw_frame`` with
  ``_pipeline_on``, the path ``run()`` takes on a free-threaded build), with one
  no-op system. Async work is a Python thread, as an app would write it today.

Each is measured three ways: baseline, with a CPU-bound 300 ms async job at 2 Hz,
and with that job plus a forced GC pause of about 100 ms every second. The
question is the ratio between a mode and its own baseline, so the render cost
does not decide it.

    uv run python benchmarks/spikes/s_a_frame.py --build
    uv run python benchmarks/spikes/s_a_frame.py --seconds 6 --repeats 3
"""

from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import os
import shutil
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CRATE = ROOT / "crates" / "spike-s-a"
SO = CRATE / "target" / "release" / "lib_spike_sa.so"
OUT = Path(__file__).with_name("s_a_results.json")

FPS = 60.0
BUDGET_MS = 1000.0 / FPS
ASYNC_HZ = 2.0
ASYNC_S = 0.3
BLOCK_MB = 50
GC_EVERY_S = 1.0
GC_CYCLES = 400_000


def cargo() -> str:
    found = shutil.which("cargo") or Path.home() / ".cargo" / "bin" / "cargo"
    return str(found)


def build() -> None:
    env = os.environ.copy()
    env["PYO3_PYTHON"] = sys.executable
    subprocess.run(
        [cargo(), "build", "--release", "--manifest-path", str(CRATE / "Cargo.toml")],
        check=True,
        env=env,
    )


def load_rust():
    if not SO.exists():
        raise SystemExit(f"{SO} is missing; run with --build")
    spec = importlib.util.spec_from_file_location("_spike_sa", SO)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.SpikeLoop


def slow_async(seconds: float = ASYNC_S) -> None:
    """About 50 MB and 300 ms of pure Python, no sleep: a system that holds the interpreter."""
    block = [0.0] * (BLOCK_MB * 1024 * 1024 // 8)
    end = time.perf_counter() + seconds
    junk: list[list[float]] = []
    total = 0.0
    while time.perf_counter() < end:
        for i in range(2000):
            total += i * 0.5
        junk.append([0.0] * 128)
        if len(junk) > 4000:
            junk.clear()
    junk.clear()
    del block


def async_worker(stop: threading.Event) -> None:
    period = 1.0 / ASYNC_HZ
    next_t = time.perf_counter()
    while not stop.is_set():
        slow_async()
        next_t += period
        delay = next_t - time.perf_counter()
        if delay > 0:
            stop.wait(delay)


def gc_pause(cycles: int = GC_CYCLES) -> float:
    """Allocate cyclic garbage, then force a collection; returns the pause in ms."""
    junk = []
    for i in range(cycles):
        a = [i]
        a.append(a)
        junk.append(a)
    del a, junk
    t0 = time.perf_counter()
    gc.collect()
    return (time.perf_counter() - t0) * 1000.0


def gc_worker(stop: threading.Event, pauses: list[float]) -> None:
    while not stop.wait(GC_EVERY_S):
        pauses.append(gc_pause())


def stats(periods: list[float], works: list[float], over: int) -> dict:
    ordered = sorted(periods)

    def pct(q: float) -> float:
        return ordered[min(len(ordered) - 1, int(q * len(ordered)))]

    return {
        "frames": len(periods),
        "p50": statistics.median(periods),
        "p99": pct(0.99),
        "p999": pct(0.999),
        "max": max(periods),
        "over33": over,
        "over50": sum(1 for p in periods if p > 50.0),
        "work_p50": statistics.median(works),
        "work_p99": sorted(works)[min(len(works) - 1, int(0.99 * len(works)))],
    }


def run_current(seconds: float, mode: str) -> dict:
    import manifoldx as mx

    engine = mx.Engine("s-a", width=64, height=64, max_entities=16, pipelined=True)
    engine._ensure_offscreen(1)
    engine.set_fixed_timestep(1.0 / FPS)
    engine._pipeline_on = True  # what run() does on the free-threaded build

    @engine.system
    def noop(query, dt):
        pass

    canvas = engine._render_canvas
    for _ in range(10):
        engine._draw_frame()
        canvas.draw()

    stop = threading.Event()
    pauses: list[float] = []
    threads = []
    if mode in ("async", "async+gc"):
        threads.append(threading.Thread(target=async_worker, args=(stop,), daemon=True))
    if mode == "async+gc":
        threads.append(threading.Thread(target=gc_worker, args=(stop, pauses), daemon=True))
    for t in threads:
        t.start()

    frames = int(seconds * FPS)
    periods, works = [], []
    over = 0
    next_t = time.perf_counter()
    prev = next_t
    for _ in range(frames):
        t0 = time.perf_counter()
        engine._draw_frame()
        canvas.draw()
        works.append((time.perf_counter() - t0) * 1000.0)
        next_t += 1.0 / FPS
        now = time.perf_counter()
        if next_t > now:
            time.sleep(next_t - now)
        else:
            next_t = now
        t1 = time.perf_counter()
        period = (t1 - prev) * 1000.0
        periods.append(period)
        if period > 33.0:
            over += 1
        prev = t1

    stop.set()
    for t in threads:
        t.join()
    result = stats(periods, works, over)
    result["gc_pauses_ms"] = pauses
    engine.close()
    return result


def run_rust(SpikeLoop, seconds: float, mode: str, sync: bool = True) -> dict:
    loop = SpikeLoop(FPS)
    calls = [0]

    def noop():
        calls[0] += 1

    if sync:
        loop.add_sync(noop)
    loop.run(10)  # warm up, no async yet
    if mode in ("async", "async+gc"):
        loop.add_async(slow_async, ASYNC_HZ)

    stop = threading.Event()
    pauses: list[float] = []
    gc_thread = None
    if mode == "async+gc":
        gc_thread = threading.Thread(target=gc_worker, args=(stop, pauses), daemon=True)
        gc_thread.start()

    frames = int(seconds * FPS)
    periods, works, over, errors = loop.run(frames)

    stop.set()
    if gc_thread is not None:
        gc_thread.join()
    result = stats(periods, works, over)
    result["gc_pauses_ms"] = pauses
    result["sync_calls"] = calls[0]
    result["callback_errors"] = errors
    return result


MODES = ("base", "async", "async+gc")
ENGINES = ("rust", "rust-nosync", "current")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=6.0)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()

    if args.build:
        build()
    SpikeLoop = load_rust()

    results: dict = {"config": {
        "fps": FPS, "seconds": args.seconds, "repeats": args.repeats,
        "async_hz": ASYNC_HZ, "async_s": ASYNC_S, "async_mb": BLOCK_MB,
        "gc_every_s": GC_EVERY_S, "gc_cycles": GC_CYCLES,
        "python": sys.version.split()[0], "free_threaded": not sys._is_gil_enabled(),
        "loadavg": os.getloadavg(),
    }, "runs": {}}

    for engine in ENGINES:
        for mode in MODES:
            rows = []
            for i in range(args.repeats):
                if engine == "current":
                    row = run_current(args.seconds, mode)
                else:
                    row = run_rust(SpikeLoop, args.seconds, mode, sync=engine == "rust")
                rows.append(row)
                print(f"{engine:7s} {mode:8s} run {i + 1}/{args.repeats}: "
                      f"p50 {row['p50']:.2f} p99 {row['p99']:.2f} max {row['max']:.2f} "
                      f"over33 {row['over33']} work {row['work_p50']:.2f}", flush=True)
            results["runs"][f"{engine}/{mode}"] = rows

    # Summaries: the median of each metric across repeats, and the ratio to base.
    summary = {}
    for engine in ENGINES:
        base = [r["p99"] for r in results["runs"][f"{engine}/base"]]
        base_p99 = statistics.median(base)
        for mode in MODES:
            rows = results["runs"][f"{engine}/{mode}"]
            p99 = statistics.median([r["p99"] for r in rows])
            summary[f"{engine}/{mode}"] = {
                "p50": statistics.median([r["p50"] for r in rows]),
                "p99": p99,
                "max": max(r["max"] for r in rows),
                "over33": sum(r["over33"] for r in rows),
                "ratio": p99 / base_p99 if base_p99 else float("nan"),
                "gc_ms": statistics.median(
                    [p for r in rows for p in r["gc_pauses_ms"]]
                ) if rows and rows[0]["gc_pauses_ms"] else 0.0,
            }
    results["summary"] = summary

    print("\n| loop | mode | p50 ms | p99 ms | max ms | frames >33 ms | p99 / base |")
    print("|---|---|---|---|---|---|---|")
    for key, row in summary.items():
        print(f"| {key} | | {row['p50']:.2f} | {row['p99']:.2f} | {row['max']:.1f} | "
              f"{row['over33']} | {row['ratio']:.2f} |")
    print(f"\nGC pause median: {statistics.median([p for r in results['runs']['rust/async+gc'] for p in r['gc_pauses_ms']]):.1f} ms"
          if results['runs']['rust/async+gc'][0]['gc_pauses_ms'] else "\nGC pause: none measured")

    args.out.write_text(json.dumps(results, indent=2))
    print(f"\nwrote {args.out} ({args.out.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
