"""S-c: reload with a live store.

Three measurements:

- ``prototype``: a Rust store (``crates/spike-s-c``) that keeps 10k rows and 200
  mesh handles while a Python app module is reimported and its systems swapped.
  Save to first frame, ten times, with the counts checked after each.
- ``current``: the same shape of app (117 meshes, 10k entities) on the current
  ``HotReloader``, which reruns the whole script and rebuilds the engine. This is
  the 3.1–3.5 s of #164, measured headless.
- ``native``: an incremental ``cargo build`` of the game crate plus loading the
  new cdylib, the other half of the dev loop.

    uv run python benchmarks/spikes/s_c_reload.py
"""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EXT = ROOT / "crates" / "spike-s-c"
EXT_SO = EXT / "target" / "release" / "lib_spike_sc.so"
GAME = ROOT / "examples" / "native_builder" / "native"
GAME_SRC = GAME / "src" / "lib.rs"
OUT = Path(__file__).with_name("s_c_results.json")

PROTO_DIR = Path("/tmp/opencode/s_c_proto")
CURRENT_DIR = Path("/tmp/opencode/s_c_current")

PROTO_APP = """\
TAG = {tag}
CALLS = 0
GEOMETRIES = [(i, 1.0 + i * 0.001) for i in range(117)]


def move(dt):
    global CALLS
    CALLS += 1


def steer(dt):
    pass


SYSTEMS = [move, steer]


def setup(world):
    if world.entity_count() == 0:
        world.spawn(10_000)
        for i in range(200):
            world.mesh(i)
    return list(SYSTEMS)
"""

CURRENT_APP = """\
import manifoldx as mx
from manifoldx.components import Mesh, Transform
from manifoldx.resources import cube

engine = mx.Engine("reload", width=64, height=64, max_entities=20000)
VERSION = {version}


@engine.on("startup")
def build(_):
    for i in range(117):
        engine.spawn(Mesh(cube(1.0 + i * 0.001, 1.0, 1.0)), Transform(pos=(i * 0.1, 0.0, 0.0)))
    engine.spawn(Transform(pos=(0.0, 0.0, 0.0)), n=10_000)
    engine.state.version = VERSION


@engine.on("frame")
def frame(_):
    engine.state.frames = engine.state.get("frames", 0) + 1


engine.run()
"""


def cargo() -> str:
    import shutil

    return str(shutil.which("cargo") or Path.home() / ".cargo" / "bin" / "cargo")


def build() -> None:
    env = os.environ.copy()
    env["PYO3_PYTHON"] = sys.executable
    subprocess.run([cargo(), "build", "--release", "--manifest-path", str(EXT / "Cargo.toml")],
                   check=True, env=env)


def load(name: str, path: Path):
    importlib.invalidate_caches()
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write(path: Path, source: str, clock: list[float]) -> None:
    path.write_text(source)
    clock[0] += 2.0
    os.utime(path, (clock[0], clock[0]))


def rss_kb() -> int:
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith("VmRSS:"):
                return int(line.split()[1])
    return 0


def prototype(repeats: int) -> dict:
    module = load("_spike_sc", EXT_SO)
    PROTO_DIR.mkdir(parents=True, exist_ok=True)
    path = PROTO_DIR / "app.py"
    clock = [time.time()]
    write(path, PROTO_APP.format(tag=0), clock)

    world = module.World()
    app = load("s_c_proto_app", path)
    world.set_systems(app.setup(world))
    for _ in range(5):
        world.step(1.0 / 60.0)

    counts = (world.entity_count(), world.mesh_count())
    frames = world.frame()
    rss = [rss_kb()]
    samples = []
    checks = []
    for tag in range(1, repeats + 1):
        t0 = time.perf_counter()
        write(path, PROTO_APP.format(tag=tag), clock)
        app = load("s_c_proto_app", path)
        world.set_systems(app.setup(world))
        world.step(1.0 / 60.0)
        samples.append((time.perf_counter() - t0) * 1000.0)
        rss.append(rss_kb())
        checks.append({
            "tag": app.TAG,
            "calls": app.CALLS,
            "entities": world.entity_count(),
            "meshes": world.mesh_count(),
            "frame": world.frame(),
        })
        assert app.TAG == tag
        assert app.CALLS == 1, "the new module's system did not run"
        assert (world.entity_count(), world.mesh_count()) == counts
        assert world.frame() == frames + tag

    return {
        "samples_ms": samples,
        "p50": statistics.median(samples),
        "p95": sorted(samples)[min(len(samples) - 1, int(0.95 * len(samples)))],
        "entities": counts[0],
        "meshes": counts[1],
        "rss_kb": rss,
        "rss_growth_pct": (rss[-1] - rss[0]) / rss[0] * 100.0,
        "checks": checks,
    }


def current(repeats: int) -> dict:
    import manifoldx as mx  # noqa: F401
    from manifoldx.backends import get_offscreen_canvas
    from manifoldx.hot import HotReloader

    CURRENT_DIR.mkdir(parents=True, exist_ok=True)
    path = CURRENT_DIR / "app.py"
    clock = [time.time()]
    write(path, CURRENT_APP.format(version=1), clock)

    def offscreen(engine):
        return get_offscreen_canvas(engine.w, engine.h)

    hot = HotReloader(script=str(path), canvas_factory=offscreen, poll=0, channel=False)
    hot.start()
    hot.step()  # startup: 117 meshes and 10k entities
    hot.step()
    samples = []
    for version in range(2, repeats + 2):
        t0 = time.perf_counter()
        write(path, CURRENT_APP.format(version=version), clock)
        hot.step()
        samples.append((time.perf_counter() - t0) * 1000.0)
        assert hot.state.version == version
    count = int(hot.engine.store._alive.sum())
    hot.engine.close()
    return {
        "samples_ms": samples,
        "p50": statistics.median(samples),
        "p95": sorted(samples)[min(len(samples) - 1, int(0.95 * len(samples)))],
        "entities": count,
    }


def native(repeats: int) -> dict:
    env = os.environ.copy()
    env["PYO3_PYTHON"] = sys.executable
    samples = []
    for _ in range(repeats):
        os.utime(GAME_SRC, None)
        t0 = time.perf_counter()
        subprocess.run([cargo(), "build", "--release", "--manifest-path", str(GAME / "Cargo.toml")],
                       check=True, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        samples.append((time.perf_counter() - t0) * 1000.0)
    t0 = time.perf_counter()
    game = load("native", GAME / "target" / "release" / "libnative.so")
    engine = load("_spike_sb", ROOT / "crates" / "spike-s-b" / "target" / "release" / "lib_spike_sb.so")
    engine.NativeBuilder(game.build_cell)
    load_ms = (time.perf_counter() - t0) * 1000.0
    return {
        "samples_ms": samples,
        "p50": statistics.median(samples),
        "p95": sorted(samples)[min(len(samples) - 1, int(0.95 * len(samples)))],
        "load_ms": load_ms,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--repeats", type=int, default=10)
    ap.add_argument("--skip-current", action="store_true")
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()

    if args.build:
        build()
    if not EXT_SO.exists():
        raise SystemExit(f"{EXT_SO} is missing; run with --build")

    proto = prototype(args.repeats)
    print(f"prototype: p50 {proto['p50']:.2f} ms, p95 {proto['p95']:.2f} ms, "
          f"{proto['entities']} entities, {proto['meshes']} meshes, RSS {proto['rss_growth_pct']:+.2f}%")

    cur = None if args.skip_current else current(5)
    if cur is not None:
        print(f"current:   p50 {cur['p50']:.1f} ms, p95 {cur['p95']:.1f} ms, {cur['entities']} entities")

    nat = native(5)
    print(f"native:    build p50 {nat['p50']:.2f} ms, p95 {nat['p95']:.2f} ms, load {nat['load_ms']:.2f} ms")

    result = {
        "config": {
            "python": sys.version.split()[0],
            "free_threaded": not sys._is_gil_enabled(),
            "cpus": os.cpu_count(), "loadavg": os.getloadavg(),
        },
        "prototype": proto,
        "current": cur,
        "native": nat,
        "passes": {
            "python_p50_under_1s": proto["p50"] < 1000.0,
            "python_p95_under_1_5s": proto["p95"] < 1500.0,
            "counts_survive": all(
                c["entities"] == proto["entities"] and c["meshes"] == proto["meshes"]
                for c in proto["checks"]
            ),
            "rss_under_5pct": abs(proto["rss_growth_pct"]) < 5.0,
            "native_p50_under_5s": nat["p50"] < 5000.0,
        },
    }
    print("\n| check | result |")
    print("|---|---|")
    for name, value in result["passes"].items():
        print(f"| {name} | {value} |")

    args.out.write_text(json.dumps(result, indent=2))
    print(f"\nwrote {args.out} ({args.out.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
