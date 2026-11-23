"""Full headless frames for the nbody and boids examples, numpy against native.

Each run is a fresh process, because the example picks its backend at import:
``NBODY_N``/``BOIDS_N`` set the size and ``MANIFOLDX_NUMPY=1`` forces the numpy
path. The frame is ``_draw_frame`` plus the offscreen canvas draw, so it includes
the render and the GPU wait; ``update`` is the systems' share.

    uv run python benchmarks/spikes/s_nbody_boids_frame.py --build
    uv run python benchmarks/spikes/s_nbody_boids_frame.py
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
CRATE = ROOT / "crates" / "spike-nbody-boids"
OUT = Path(__file__).with_name("s_nbody_boids_frame_results.json")
EXAMPLES = {"nbody": ROOT / "examples" / "nbody.py", "boids": ROOT / "examples" / "boids.py"}


def cargo() -> str:
    return str(shutil.which("cargo") or Path.home() / ".cargo" / "bin" / "cargo")


def build() -> None:
    env = os.environ.copy()
    env["PYO3_PYTHON"] = sys.executable
    subprocess.run([cargo(), "build", "--release", "--manifest-path", str(CRATE / "Cargo.toml")],
                   check=True, env=env)


def one(workload: str, n: int, frames: int) -> None:
    path = EXAMPLES[workload]
    spec = importlib.util.spec_from_file_location(f"{workload}_example", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    engine = module.engine
    engine._ensure_offscreen(1)
    engine.set_fixed_timestep(1 / 60)
    canvas = engine._render_canvas
    for _ in range(3):
        engine._draw_frame()
        canvas.draw()

    frame_ms, update_ms, render_ms = [], [], []
    for _ in range(frames):
        t0 = time.perf_counter()
        engine._draw_frame()
        canvas.draw()
        frame_ms.append((time.perf_counter() - t0) * 1000.0)
        update_ms.append(engine.frame_split["update"] * 1000.0)
        render_ms.append(engine.frame_split["render"] * 1000.0)
    result = {
        "workload": workload, "n": n,
        "backend": "native" if module.NATIVE is not None else "numpy",
        "frames": frames,
        "frame_p50": statistics.median(frame_ms),
        "frame_p99": sorted(frame_ms)[min(len(frame_ms) - 1, int(0.99 * len(frame_ms)))],
        "update_p50": statistics.median(update_ms),
        "update_p99": sorted(update_ms)[min(len(update_ms) - 1, int(0.99 * len(update_ms)))],
        "render_p50": statistics.median(render_ms),
    }
    engine.close()
    print(json.dumps(result))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--one", nargs=3, metavar=("WORKLOAD", "N", "FRAMES"))
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()

    if args.one:
        one(args.one[0], int(args.one[1]), int(args.one[2]))
        return 0
    if args.build:
        build()

    cases = [
        ("nbody", 500, 10),
        ("nbody", 2000, 5),
        ("boids", 300, 10),
        ("boids", 2000, 5),
    ]
    rows = []
    for workload, n, frames in cases:
        for backend in ("numpy", "native"):
            env = os.environ.copy()
            env["MANIFOLDX_NUMPY" if backend == "numpy" else "MANIFOLDX_NATIVE_SO"] = (
                "1" if backend == "numpy"
                else str(CRATE / "target" / "release" / "lib_spike_nb.so")
            )
            env["NBODY_N" if workload == "nbody" else "BOIDS_N"] = str(n)
            proc = subprocess.run(
                [sys.executable, str(Path(__file__)), "--one", workload, str(n), str(frames)],
                capture_output=True, text=True, cwd=ROOT, env=env,
            )
            if proc.returncode != 0:
                print(f"{workload} {n} {backend} failed:\n{proc.stdout}\n{proc.stderr}", file=sys.stderr)
                continue
            row = json.loads(proc.stdout.strip().splitlines()[-1])
            rows.append(row)
            print(f"{workload:6s} N={n:<5d} {row['backend']:6s} "
                  f"frame {row['frame_p50']:7.2f} ms  update {row['update_p50']:7.2f} ms  "
                  f"render {row['render_p50']:6.2f} ms", flush=True)

    print("\n| workload | N | backend | frame p50 | frame p99 | update p50 | render p50 |")
    print("|---|---:|---|---:|---:|---:|---:|")
    for row in rows:
        print(f"| {row['workload']} | {row['n']} | {row['backend']} | {row['frame_p50']:.2f} | "
              f"{row['frame_p99']:.2f} | {row['update_p50']:.2f} | {row['render_p50']:.2f} |")

    result = {
        "config": {"python": sys.version.split()[0], "free_threaded": not sys._is_gil_enabled(),
                   "cpus": os.cpu_count(), "loadavg": os.getloadavg()},
        "rows": rows,
    }
    args.out.write_text(json.dumps(result, indent=2))
    print(f"\nwrote {args.out} ({args.out.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
