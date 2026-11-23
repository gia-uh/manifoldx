"""S-d: selections and commands over 100k rows without reading rows into Python.

`crates/spike-s-d` holds the store. Python builds a selection (a spec in Rust),
enqueues a set and a spawn, and never touches a row. A Rust loop runs five
kernels, calls the Python system at 2 Hz, and applies its commands at the frame
boundary; the apply time is measured inside Rust.

    uv run python benchmarks/spikes/s_d_ir.py --build
    uv run python benchmarks/spikes/s_d_ir.py
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CRATE = ROOT / "crates" / "spike-s-d"
SO = CRATE / "target" / "release" / "lib_spike_sd.so"
OUT = Path(__file__).with_name("s_d_results.json")


def cargo() -> str:
    return str(shutil.which("cargo") or Path.home() / ".cargo" / "bin" / "cargo")


def build() -> None:
    env = os.environ.copy()
    env["PYO3_PYTHON"] = sys.executable
    subprocess.run([cargo(), "build", "--release", "--manifest-path", str(CRATE / "Cargo.toml")],
                   check=True, env=env)


def load():
    if not SO.exists():
        raise SystemExit(f"{SO} is missing; run with --build")
    spec = importlib.util.spec_from_file_location("_spike_sd", SO)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def percentile(values, q):
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(q * len(ordered)))]


def stats(values):
    return {
        "p50": statistics.median(values),
        "p99": percentile(values, 0.99),
        "max": max(values),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--frames", type=int, default=600)
    ap.add_argument("--every", type=int, default=30)
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()

    if args.build:
        build()
    module = load()
    world = module.World()
    world.populate(100_000)

    selection = world.select("speed", "<", 2.0)
    selected = world.count(selection)

    t0 = time.perf_counter()
    world.read_speeds()
    read_ms = (time.perf_counter() - t0) * 1000.0
    world.reset_rows_read()

    def traffic():
        sel = world.select("speed", "<", 2.0)
        world.set(sel, "target_speed", 8.0)
        world.spawn(40)

    started = time.perf_counter()
    out = world.run(args.frames, args.every, traffic)
    wall = time.perf_counter() - started

    periods = out["periods"]
    applies = out["applies"]
    batches = out["batches"]
    asyncs = out["asyncs"]
    result = {
        "config": {
            "python": sys.version.split()[0],
            "free_threaded": not sys._is_gil_enabled(),
            "cpus": os.cpu_count(), "loadavg": os.getloadavg(),
            "frames": args.frames, "every": args.every,
        },
        "selection": {"column": "speed", "op": "<", "value": 2.0, "matched": selected},
        "contrast": {"read_speeds_ms": read_ms},
        "periods": stats(periods),
        "periods_over_33": sum(value > 33.0 for value in periods),
        "applies": stats(applies),
        "batches": {"count": len(batches), **stats(batches)},
        "asyncs": stats(asyncs),
        "commands": {
            "sets": out["sets"], "spawns": out["spawns"],
            "affected": out["affected"], "rows_read": out["rows_read"],
            "entities": world.entity_count(),
        },
        "wall_s": wall,
    }
    result["passes"] = {
        "zero_rows_read": out["rows_read"] == 0,
        "apply_p99_under_1ms": result["batches"]["p99"] <= 1.0,
        "frame_p99_under_16_7ms": result["periods"]["p99"] <= 16.7,
    }

    print(f"selection: {selected} of 100,000 rows (speed < 2.0)")
    print(f"read_speeds (the naive path): {read_ms:.2f} ms")
    print(f"frame: p50 {result['periods']['p50']:.3f} ms, p99 {result['periods']['p99']:.3f} ms, "
          f"max {result['periods']['max']:.2f} ms, over33 {result['periods_over_33']}")
    print(f"apply: batches {len(batches)}, p50 {result['batches']['p50']:.3f} ms, "
          f"p99 {result['batches']['p99']:.3f} ms, max {result['batches']['max']:.3f} ms")
    print(f"async system: p50 {result['asyncs']['p50']:.3f} ms, p99 {result['asyncs']['p99']:.3f} ms")
    print(f"commands: {out['sets']} sets, {out['spawns']} spawns, {out['affected']} rows affected, "
          f"rows read {out['rows_read']}, entities {world.entity_count()}")

    print("\n| check | result |")
    print("|---|---|")
    for name, value in result["passes"].items():
        print(f"| {name} | {value} |")

    args.out.write_text(json.dumps(result, indent=2))
    print(f"\nwrote {args.out} ({args.out.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
