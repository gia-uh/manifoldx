"""Native steps for the nbody and boids examples, against their numpy versions.

The numpy steps are the examples' own math, copied into functions so they can be
timed without an engine or a window. The native steps are in
``crates/spike-nbody-boids``: all-pairs gravity with the same
``max(dist, softening)`` and the boid rules over a uniform grid at the perception
radius. The examples run at 500 and 300 bodies; the spike also scales N to see
where a native step starts to win and where numpy is already enough.

    uv run python benchmarks/spikes/s_nbody_boids.py --build
    uv run python benchmarks/spikes/s_nbody_boids.py
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

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
CRATE = ROOT / "crates" / "spike-nbody-boids"
SO = CRATE / "target" / "release" / "lib_spike_nb.so"
OUT = Path(__file__).with_name("s_nbody_boids_results.json")

NBODY = dict(G=20.0, SOFTENING=0.05)
BOIDS = dict(
    PERCEPTION_SQ=16.0, FEAR_RADIUS=10.0, SEP_W=1.5, ALI_W=1.0, COH_W=0.8,
    FEAR_W=20.0, MAX_SPEED=10.0, PANIC_SPEED=15.0, MIN_SPEED=4.0,
    BOUND_RADIUS=12.0, PULL_STRENGTH=0.02, PRED_SPEED=3.0, PRED_TURN_RATE=1.0,
)


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
    spec = importlib.util.spec_from_file_location("_spike_nb", SO)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def nbody_numpy(pos, vel, masses, dt):
    import manifoldx as mx

    accel = mx.physics.gravity(pos, masses=masses, G=NBODY["G"], softening=NBODY["SOFTENING"])
    vel += accel * dt
    pos += vel * dt


def boids_numpy(pos, vel, pred_pos, pred_dir, dt, rng):
    diff = pos[np.newaxis, :, :] - pos[:, np.newaxis, :]
    dist_sq = (diff * diff).sum(axis=2)
    neighbors = (dist_sq < BOIDS["PERCEPTION_SQ"]) & (dist_sq > 1e-6)
    n_count = neighbors.sum(axis=1, keepdims=True).astype(np.float32)
    safe_count = np.maximum(n_count, 1.0)

    inv_dsq = np.zeros_like(dist_sq)
    inv_dsq[neighbors] = 1.0 / dist_sq[neighbors]
    sep = (-diff * (neighbors[:, :, np.newaxis] * inv_dsq[:, :, np.newaxis])).sum(axis=1)

    avg_vel = (vel[np.newaxis, :, :] * neighbors[:, :, np.newaxis]).sum(axis=1) / safe_count
    ali = avg_vel - vel
    center = (pos[np.newaxis, :, :] * neighbors[:, :, np.newaxis]).sum(axis=1) / safe_count
    coh = center - pos

    pred_diff = pos[:, np.newaxis, :] - pred_pos[np.newaxis, :, :]
    pred_dist = np.linalg.norm(pred_diff, axis=2)
    scared = pred_dist < BOIDS["FEAR_RADIUS"]
    scared_any = scared.any(axis=1)
    inv_dist = np.zeros_like(pred_dist)
    inv_dist[scared] = 1.0 / pred_dist[scared]
    fear = (pred_diff * (scared[:, :, np.newaxis] * inv_dist[:, :, np.newaxis])).sum(axis=1)

    pull = -pos * BOIDS["PULL_STRENGTH"]
    accel = (BOIDS["SEP_W"] * sep + BOIDS["ALI_W"] * ali + BOIDS["COH_W"] * coh
             + BOIDS["FEAR_W"] * fear + pull)
    vel += accel * dt

    cur_r = np.linalg.norm(pos, axis=1, keepdims=True)
    overshoot = np.maximum(cur_r - BOIDS["BOUND_RADIUS"], 0.0)
    inward = -pos / np.maximum(cur_r, 1e-9)
    vel += (inward * overshoot * 3.0).astype(np.float32)

    speed = np.linalg.norm(vel, axis=1, keepdims=True)
    max_s = np.where(scared_any[:, np.newaxis], BOIDS["PANIC_SPEED"], BOIDS["MAX_SPEED"])
    min_s = np.where(scared_any[:, np.newaxis], BOIDS["MIN_SPEED"] * 1.5, BOIDS["MIN_SPEED"])
    vel[:] = np.where(speed > max_s, vel * max_s / np.maximum(speed, 1e-9), vel)
    vel[:] = np.where((speed < min_s) & (speed > 1e-6), vel * min_s / np.maximum(speed, 1e-9), vel)

    pred_dir += rng.standard_normal(pred_dir.shape).astype(np.float32) * BOIDS["PRED_TURN_RATE"] * dt
    pred_dir /= np.maximum(np.linalg.norm(pred_dir, axis=1, keepdims=True), 1e-6)
    pred_r = np.linalg.norm(pred_pos, axis=1, keepdims=True)
    pred_over = np.maximum(pred_r - BOIDS["BOUND_RADIUS"] * 0.8, 0.0)
    pred_inward = -pred_pos / np.maximum(pred_r, 1e-6) * pred_over * 0.5
    pred_vel = pred_dir * BOIDS["PRED_SPEED"] + pred_inward
    pos += vel * dt
    pred_pos += pred_vel * dt


def make_nbody(n, seed=7):
    rng = np.random.default_rng(seed)
    pos = (rng.random((n, 3), dtype=np.float32) - 0.5) * np.float32(10.0 * n ** (1 / 3))
    vel = np.zeros((n, 3), np.float32)
    masses = rng.uniform(0.5, 3.0, n).astype(np.float32)
    return pos, vel, masses


def make_boids(n, p=3, seed=7):
    rng = np.random.default_rng(seed)
    pos = (rng.standard_normal((n, 3)) * 5.0).astype(np.float32)
    vel = (rng.standard_normal((n, 3)) * 6.0).astype(np.float32)
    pred_pos = (rng.random((p, 3)) * 10.0 - 5.0).astype(np.float32)
    pred_dir = rng.standard_normal((p, 3)).astype(np.float32)
    pred_dir /= np.maximum(np.linalg.norm(pred_dir, axis=1, keepdims=True), 1e-6)
    return pos, vel, pred_pos, pred_dir


def timed(fn, steps=10, warmup=2):
    for _ in range(warmup):
        fn()
    samples = []
    for _ in range(steps):
        t0 = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - t0) * 1000.0)
    return statistics.median(samples)


def correctness(module):
    pos, vel, masses = make_nbody(500)
    p0, v0 = pos.copy(), vel.copy()
    p1, v1 = pos.copy(), vel.copy()
    nbody_numpy(p0, v0, masses, 1 / 60)
    module.nbody_step(p1, v1, masses, 1 / 60, NBODY["G"], NBODY["SOFTENING"])
    nbody_ok = bool(np.allclose(p0, p1, rtol=1e-3, atol=1e-3) and np.allclose(v0, v1, rtol=1e-3, atol=1e-3))

    pos, vel, pred_pos, pred_dir = make_boids(300)
    pred_pos[:] = 1000.0  # far: no fear, so the boid math is the only thing compared
    p0, v0, pp0, pd0 = pos.copy(), vel.copy(), pred_pos.copy(), pred_dir.copy()
    p1, v1, pp1, pd1 = pos.copy(), vel.copy(), pred_pos.copy(), pred_dir.copy()
    boids_numpy(p0, v0, pp0, pd0, 1 / 60, np.random.default_rng(0))
    module.boids_step(p1, v1, pp1, pd1, 1 / 60)
    boids_ok = bool(np.allclose(p0, p1, rtol=1e-3, atol=1e-3) and np.allclose(v0, v1, rtol=1e-3, atol=1e-3))
    return {
        "nbody": nbody_ok,
        "boids": boids_ok,
        "nbody_error": float(np.max(np.abs(p0 - p1))),
        "boids_error": float(np.max(np.abs(p0 - p1))),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()

    if args.build:
        build()
    module = load()
    checks = correctness(module)
    print(f"correctness: nbody {checks['nbody']} (max {checks['nbody_error']:.2e}), "
          f"boids {checks['boids']} (max {checks['boids_error']:.2e})")

    rows = []
    nbody_sizes = [500, 1000, 2000, 5000, 10000]
    for n in nbody_sizes:
        pos, vel, masses = make_nbody(n)
        numpy_ms = None
        if n <= 2000:
            p, v = pos.copy(), vel.copy()
            numpy_ms = timed(lambda: nbody_numpy(p, v, masses, 1 / 60), args.steps)
        p, v = pos.copy(), vel.copy()
        native_ms = timed(lambda: module.nbody_step(p, v, masses, 1 / 60, NBODY["G"], NBODY["SOFTENING"]), args.steps)
        rows.append({
            "workload": "nbody", "n": n,
            "numpy_ms": numpy_ms, "native_ms": native_ms,
            "speedup": (numpy_ms / native_ms) if numpy_ms else None,
            "default": n == 500,
        })

    boids_sizes = [300, 1000, 2000, 5000, 10000]
    for n in boids_sizes:
        pos, vel, pred_pos, pred_dir = make_boids(n)
        numpy_ms = None
        if n <= 2000:
            p, v, pp, pd = pos.copy(), vel.copy(), pred_pos.copy(), pred_dir.copy()
            rng = np.random.default_rng(0)
            numpy_ms = timed(lambda: boids_numpy(p, v, pp, pd, 1 / 60, rng), args.steps)
        p, v, pp, pd = pos.copy(), vel.copy(), pred_pos.copy(), pred_dir.copy()
        native_ms = timed(lambda: module.boids_step(p, v, pp, pd, 1 / 60), args.steps)
        rows.append({
            "workload": "boids", "n": n,
            "numpy_ms": numpy_ms, "native_ms": native_ms,
            "speedup": (numpy_ms / native_ms) if numpy_ms else None,
            "default": n == 300,
        })

    print("\n| workload | N | numpy ms | native ms | speed-up | note |")
    print("|---|---:|---:|---:|---:|---|")
    for row in rows:
        numpy_ms = "—" if row["numpy_ms"] is None else f"{row['numpy_ms']:.2f}"
        speed = "—" if row["speedup"] is None else f"{row['speedup']:.1f}×"
        note = "example default" if row["default"] else ("numpy skipped (memory)" if row["numpy_ms"] is None else "")
        print(f"| {row['workload']} | {row['n']} | {numpy_ms} | {row['native_ms']:.2f} | {speed} | {note} |")

    result = {
        "config": {"python": sys.version.split()[0], "free_threaded": not sys._is_gil_enabled(),
                   "cpus": os.cpu_count(), "loadavg": os.getloadavg(), "steps": args.steps},
        "correctness": checks,
        "rows": rows,
    }
    args.out.write_text(json.dumps(result, indent=2))
    print(f"\nwrote {args.out} ({args.out.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
