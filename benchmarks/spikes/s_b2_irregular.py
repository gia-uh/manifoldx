"""S-b2: the same native capsule on an irregular cell.

A 25 m street grid with a ninth of its edges missing, BFS distances from the
centre, four candidate lots per block (kept by hash), building heights that read
the BFS distance, and lamp rows that carry it. The Python side is measured twice:
the loop shape and a vectorized numpy version with the graph and BFS still in
Python.

    uv run python benchmarks/spikes/s_b2_irregular.py --build
    uv run python benchmarks/spikes/s_b2_irregular.py
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
GAME = ROOT / "examples" / "native_builder" / "native"
GAME_SO = GAME / "target" / "release" / "libnative.so"
ENGINE = ROOT / "crates" / "spike-s-b"
ENGINE_SO = ENGINE / "target" / "release" / "lib_spike_sb.so"
OUT = Path(__file__).with_name("s_b2_results.json")
PY_DIR = ROOT / "examples" / "native_builder"

sys.path.insert(0, str(PY_DIR))


def cargo() -> str:
    return str(shutil.which("cargo") or Path.home() / ".cargo" / "bin" / "cargo")


def build() -> None:
    env = os.environ.copy()
    env["PYO3_PYTHON"] = sys.executable
    for manifest in (GAME / "Cargo.toml", ENGINE / "Cargo.toml"):
        subprocess.run([cargo(), "build", "--release", "--manifest-path", str(manifest)],
                       check=True, env=env)


def load(name: str, path: Path):
    if not path.exists():
        raise SystemExit(f"{path} is missing; run with --build")
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def modules():
    return load("native", GAME_SO), load("_spike_sb", ENGINE_SO)


def compare(builder, seed: int) -> dict:
    import numpy as np

    import irregular

    native = builder.build(0.0, 0.0, 1000.0, 0, seed)
    report = {"seed": seed, "rows": False, "meshes": []}
    for label, python in (
        ("loop", irregular.build_cell_irregular(seed=seed)),
        ("numpy", irregular.build_cell_irregular_numpy(seed=seed)),
    ):
        same_rows = native["rows"] == python["rows"]
        meshes = []
        for k, mesh in enumerate(native["meshes"]):
            item = {"material": mesh["material"]}
            for field, dtype in (("positions", "<f4"), ("normals", "<f4"), ("indices", "<u4")):
                got = np.frombuffer(mesh[field], dtype=dtype)
                want = python["meshes"][k][field].ravel()
                item[field] = bool(np.array_equal(got, want))
                item[f"{field}_count"] = int(got.size)
            meshes.append(item)
        report[label] = {
            "rows": same_rows,
            "meshes": meshes,
            "same": same_rows and all(
                m["positions"] and m["normals"] and m["indices"] for m in meshes
            ),
        }
    report["same"] = report["loop"]["same"] and report["numpy"]["same"]
    return report


def time_builders(builder, repeats: int = 20) -> dict:
    import irregular

    cases = {
        "python_loop": lambda: irregular.build_cell_irregular(seed=7),
        "python_numpy": lambda: irregular.build_cell_irregular_numpy(seed=7),
        "native_wall": lambda: builder.build(0.0, 0.0, 1000.0, 0, 7),
    }
    out = {}
    for name, fn in cases.items():
        for _ in range(3):
            fn()
        samples = []
        for _ in range(repeats):
            t0 = time.perf_counter()
            fn()
            samples.append((time.perf_counter() - t0) * 1000.0)
        out[name] = {"p50": statistics.median(samples), "min": min(samples),
                     "p95": sorted(samples)[min(len(samples) - 1, int(0.95 * len(samples)))]}
    inside = [builder.build(0.0, 0.0, 1000.0, 0, 7)["ms"] for _ in range(repeats)]
    out["native_inside"] = {"p50": statistics.median(inside), "min": min(inside)}
    wall = out["native_wall"]["p50"]
    out["speedup_vs_loop"] = out["python_loop"]["p50"] / wall
    out["speedup_vs_numpy"] = out["python_numpy"]["p50"] / wall
    return out


def peak_rss_kb(pid: int | None = None) -> int:
    path = "/proc/self/status" if pid is None else f"/proc/{pid}/status"
    with open(path) as f:
        for line in f:
            if line.startswith("VmHWM:"):
                return int(line.split()[1])
    return 0


def _pool_job(seed: int) -> int:
    import irregular

    return int(irregular.build_cell_irregular(seed=seed)["meshes"][1]["positions"].nbytes)


def memory_pool() -> None:
    from concurrent.futures import ProcessPoolExecutor
    import multiprocessing as mp

    pool = ProcessPoolExecutor(max_workers=2, mp_context=mp.get_context("spawn"))
    list(pool.map(_pool_job, range(8)))
    workers = {pid: peak_rss_kb(pid) for pid in list(pool._processes)}
    print(json.dumps({"kind": "pool", "parent_kb": peak_rss_kb(),
                      "workers_kb": list(workers.values()),
                      "total_kb": peak_rss_kb() + sum(workers.values())}))
    pool.shutdown(wait=True)


def memory_rust() -> None:
    native, sb = modules()
    builder = sb.NativeBuilder(native.build_cell_irregular)
    for seed in range(8):
        builder.build(0.0, 0.0, 1000.0, 0, seed)
    print(json.dumps({"kind": "rust", "total_kb": peak_rss_kb()}))


def memory() -> dict:
    def run(flag: str) -> dict:
        proc = subprocess.run([sys.executable, str(Path(__file__)), flag],
                              capture_output=True, text=True, cwd=ROOT)
        if proc.returncode != 0:
            raise RuntimeError(f"{flag} failed:\n{proc.stdout}\n{proc.stderr}")
        return json.loads(proc.stdout.strip().splitlines()[-1])

    pool = run("--memory-pool")
    rust = run("--memory-rust")
    return {"pool": pool, "rust": rust,
            "ratio": rust["total_kb"] / pool["total_kb"],
            "passes_25pct": rust["total_kb"] <= 0.25 * pool["total_kb"]}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--repeats", type=int, default=20)
    ap.add_argument("--memory-pool", action="store_true")
    ap.add_argument("--memory-rust", action="store_true")
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()

    if args.memory_pool:
        memory_pool()
        return 0
    if args.memory_rust:
        memory_rust()
        return 0

    if args.build:
        build()
    native, sb = modules()
    builder = sb.NativeBuilder(native.build_cell_irregular)

    comparisons = [compare(builder, seed) for seed in (0, 7, 12_345)]
    timing = time_builders(builder, args.repeats)
    memory_result = memory()

    result = {
        "config": {
            "python": sys.version.split()[0],
            "free_threaded": not sys._is_gil_enabled(),
            "cpus": os.cpu_count(), "loadavg": os.getloadavg(),
            "repeats": args.repeats,
        },
        "comparisons": comparisons,
        "timing": timing,
        "memory": memory_result,
    }
    result["passes"] = {
        "bytes_identical": all(c["same"] for c in comparisons),
        "speed_4x_vs_loop": timing["speedup_vs_loop"] >= 4.0,
        "speed_4x_vs_numpy": timing["speedup_vs_numpy"] >= 4.0,
        "memory_25pct": memory_result["passes_25pct"],
    }

    print("\n| check | result |")
    print("|---|---|")
    for name, value in result["passes"].items():
        print(f"| {name} | {value} |")
    print(f"\nbytes identical: {result['passes']['bytes_identical']}")
    print(f"timing: loop {timing['python_loop']['p50']:.2f} ms, "
          f"numpy {timing['python_numpy']['p50']:.2f} ms, "
          f"native {timing['native_wall']['p50']:.2f} ms "
          f"(inside {timing['native_inside']['p50']:.2f} ms)")
    print(f"speedup: {timing['speedup_vs_loop']:.1f}x vs loop, "
          f"{timing['speedup_vs_numpy']:.1f}x vs numpy")
    print(f"memory: pool {memory_result['pool']['total_kb'] / 1024:.1f} MB, "
          f"rust {memory_result['rust']['total_kb'] / 1024:.1f} MB, "
          f"ratio {memory_result['ratio']:.2f}")

    args.out.write_text(json.dumps(result, indent=2))
    print(f"\nwrote {args.out} ({args.out.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
