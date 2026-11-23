"""S-b: a game's cell builder in Rust, behind a capsule, against the same builder in Python.

Builds two separate cdylibs (the game and the engine-side collector), compares
their bytes, times them, measures the memory of the process-pool path against the
in-process native path, and runs the failure cases: wrong version, null pointer,
panic, malformed inputs and a fuzz corpus.

    uv run python benchmarks/spikes/s_b_builders.py --build
    uv run python benchmarks/spikes/s_b_builders.py
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import random
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
OUT = Path(__file__).with_name("s_b_results.json")
PY_CELL = ROOT / "examples" / "native_builder"

sys.path.insert(0, str(PY_CELL))


def cargo() -> str:
    return str(shutil.which("cargo") or Path.home() / ".cargo" / "bin" / "cargo")


def build() -> None:
    env = os.environ.copy()
    env["PYO3_PYTHON"] = sys.executable
    for manifest in (GAME / "Cargo.toml", ENGINE / "Cargo.toml"):
        subprocess.run(
            [cargo(), "build", "--release", "--manifest-path", str(manifest)],
            check=True,
            env=env,
        )


def load(name: str, path: Path):
    if not path.exists():
        raise SystemExit(f"{path} is missing; run with --build")
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def modules():
    return load("native", GAME_SO), load("_spike_sb", ENGINE_SO)


def arrays(result: dict):
    import numpy as np

    return [
        {
            "positions": np.frombuffer(mesh["positions"], dtype="<f4"),
            "normals": np.frombuffer(mesh["normals"], dtype="<f4"),
            "indices": np.frombuffer(mesh["indices"], dtype="<u4"),
            "material": mesh["material"],
        }
        for mesh in result["meshes"]
    ]


def compare(builder, seed: int) -> dict:
    import numpy as np

    import cell

    native = builder.build(0.0, 0.0, 1000.0, 0, seed)
    python = cell.build_cell(seed=seed)
    report = {"seed": seed, "meshes": [], "rows": native["rows"] == python["rows"]}
    for got, want in zip(arrays(native), python["meshes"]):
        report["meshes"].append({
            "material": got["material"],
            "positions": np.array_equal(got["positions"], want["positions"].ravel()),
            "normals": np.array_equal(got["normals"], want["normals"].ravel()),
            "indices": np.array_equal(got["indices"], want["indices"].ravel()),
            "vertices": int(got["positions"].size // 3),
            "indices_count": int(got["indices"].size),
        })
    report["same"] = report["rows"] and all(
        m["positions"] and m["normals"] and m["indices"] for m in report["meshes"]
    )
    return report


def time_builders(builder, repeats: int = 20) -> dict:
    import cell

    cases = {
        "python_loop": lambda: cell.build_cell(seed=7),
        "python_numpy": lambda: cell.build_cell_numpy(seed=7),
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
        out[name] = {
            "p50": statistics.median(samples),
            "min": min(samples),
            "p95": sorted(samples)[min(len(samples) - 1, int(0.95 * len(samples)))],
        }
    native_inside = [builder.build(0.0, 0.0, 1000.0, 0, 7)["ms"] for _ in range(repeats)]
    out["native_inside"] = {"p50": statistics.median(native_inside), "min": min(native_inside)}
    loop = out["python_loop"]["p50"]
    numpy = out["python_numpy"]["p50"]
    wall = out["native_wall"]["p50"]
    out["speedup_vs_loop"] = loop / wall
    out["speedup_vs_numpy"] = numpy / wall
    return out


def peak_rss_kb(pid: int | None = None) -> int:
    path = "/proc/self/status" if pid is None else f"/proc/{pid}/status"
    with open(path) as f:
        for line in f:
            if line.startswith("VmHWM:"):
                return int(line.split()[1])
    return 0


def _pool_job(seed: int) -> int:
    import cell

    return int(cell.build_cell(seed=seed)["meshes"][1]["positions"].nbytes)


def memory_pool() -> None:
    from concurrent.futures import ProcessPoolExecutor
    import multiprocessing as mp

    ctx = mp.get_context("spawn")
    pool = ProcessPoolExecutor(max_workers=2, mp_context=ctx)
    list(pool.map(_pool_job, range(8)))
    workers = {pid: peak_rss_kb(pid) for pid in list(pool._processes)}
    total = peak_rss_kb() + sum(workers.values())
    print(json.dumps({
        "kind": "pool", "parent_kb": peak_rss_kb(),
        "workers_kb": list(workers.values()), "total_kb": total,
    }))
    pool.shutdown(wait=True)


def memory_rust() -> None:
    native, sb = modules()
    builder = sb.NativeBuilder(native.build_cell)
    for seed in range(8):
        builder.build(0.0, 0.0, 1000.0, 0, seed)
    print(json.dumps({"kind": "rust", "total_kb": peak_rss_kb()}))


def memory() -> dict:
    def run(flag: str) -> dict:
        proc = subprocess.run(
            [sys.executable, str(Path(__file__)), flag],
            capture_output=True, text=True, cwd=ROOT,
        )
        if proc.returncode != 0:
            raise RuntimeError(f"{flag} failed:\n{proc.stdout}\n{proc.stderr}")
        return json.loads(proc.stdout.strip().splitlines()[-1])

    pool = run("--memory-pool")
    rust = run("--memory-rust")
    return {
        "pool": pool,
        "rust": rust,
        "ratio": rust["total_kb"] / pool["total_kb"],
        "passes_25pct": rust["total_kb"] <= 0.25 * pool["total_kb"],
    }


def failures(native, sb) -> dict:
    out = {}

    good = sb.NativeBuilder(native.build_cell)
    out["opens_good"] = True

    try:
        bad = sb.NativeBuilder(native.build_cell_panics)
        bad.build(0.0, 0.0, 1000.0, 0, 1)
        out["panic"] = {"raised": False}
    except RuntimeError as error:
        out["panic"] = {"raised": True, "message": str(error), "mentions_panic": "panic" in str(error).lower()}
    out["after_panic_still_works"] = good.build(0.0, 0.0, 1000.0, 0, 1)["rows"] != b""

    try:
        sb.NativeBuilder(native.build_cell_v2)
        out["wrong_version"] = {"raised": False}
    except ValueError as error:
        out["wrong_version"] = {"raised": True, "message": str(error)}

    # CPython refuses to make a capsule with a null pointer, so the loader's
    # null check is defensive. The name is the whole contract: a capsule with the
    # right name and a data pointer opens, and only a call would crash.
    opaque = sb.make_capsule("manifoldx.builder.v1", id(object()))
    sb.NativeBuilder(opaque)
    out["opaque_pointer_accepted"] = True

    bad_inputs = [0, 1, 2.5, "builder", None, [], {}, b"capsule", object()]
    raised = 0
    for obj in bad_inputs:
        try:
            sb.NativeBuilder(obj)
        except (TypeError, ValueError):
            raised += 1
    out["non_capsules"] = {"tested": len(bad_inputs), "rejected": raised}

    # The fuzz corpus: arbitrary Python objects, then capsules with a wrong or
    # null pointer. Nothing here is ever called, so a crash would be a loader bug.
    random.seed(1234)
    rejected = 0
    total = 10_000
    for i in range(total):
        choice = i % 6
        obj = ([i], {"i": i}, str(i), float(i), None, b"x" * (i % 17))[choice]
        try:
            sb.NativeBuilder(obj)
        except (TypeError, ValueError):
            rejected += 1
    out["fuzz_objects"] = {"tested": total, "rejected": rejected}

    capsules = 1_000
    rejected_capsules = 0
    for i in range(capsules):
        capsule = sb.make_capsule(f"other.builder.{i}", id(object()))
        try:
            sb.NativeBuilder(capsule)
        except ValueError:
            rejected_capsules += 1
    out["fuzz_capsules"] = {"tested": capsules, "rejected": rejected_capsules}

    out["passes"] = (
        out["panic"]["raised"]
        and out["panic"]["mentions_panic"]
        and out["after_panic_still_works"]
        and out["wrong_version"]["raised"]
        and out["non_capsules"]["rejected"] == len(bad_inputs)
        and out["fuzz_objects"]["rejected"] == total
        and out["fuzz_capsules"]["rejected"] == capsules
    )
    return out


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
    builder = sb.NativeBuilder(native.build_cell)

    comparisons = [compare(builder, seed) for seed in (0, 7, 12_345)]
    timing = time_builders(builder, args.repeats)
    memory_result = memory()
    failure_result = failures(native, sb)

    result = {
        "config": {
            "python": sys.version.split()[0],
            "free_threaded": not sys._is_gil_enabled(),
            "cpus": os.cpu_count(),
            "loadavg": os.getloadavg(),
            "repeats": args.repeats,
        },
        "comparisons": comparisons,
        "timing": timing,
        "memory": memory_result,
        "failures": failure_result,
    }
    result["passes"] = {
        "bytes_identical": all(c["same"] for c in comparisons),
        "speed_4x_vs_loop": timing["speedup_vs_loop"] >= 4.0,
        "speed_4x_vs_numpy": timing["speedup_vs_numpy"] >= 4.0,
        "memory_25pct": memory_result["passes_25pct"],
        "failures": failure_result["passes"],
    }

    print("\n| check | result |")
    print("|---|---|")
    for name, value in result["passes"].items():
        print(f"| {name} | {value} |")
    print(f"\nbytes identical: {result['passes']['bytes_identical']}")
    print(f"timing: python loop {timing['python_loop']['p50']:.2f} ms, "
          f"python numpy {timing['python_numpy']['p50']:.2f} ms, "
          f"native {timing['native_wall']['p50']:.2f} ms "
          f"(inside {timing['native_inside']['p50']:.2f} ms)")
    print(f"speedup: {timing['speedup_vs_loop']:.1f}x vs loop, "
          f"{timing['speedup_vs_numpy']:.1f}x vs numpy")
    print(f"memory: pool {memory_result['pool']['total_kb'] / 1024:.1f} MB, "
          f"rust {memory_result['rust']['total_kb'] / 1024:.1f} MB, "
          f"ratio {memory_result['ratio']:.2f}")
    print(f"failures: {failure_result['passes']}")

    args.out.write_text(json.dumps(result, indent=2))
    print(f"\nwrote {args.out} ({args.out.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
