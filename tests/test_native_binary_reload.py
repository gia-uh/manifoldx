"""A rebuilt native builder swaps at the frame boundary and the old one stays live."""

from __future__ import annotations

import os
import shutil
import struct
import subprocess
import sys
import textwrap
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from manifoldx.cli import main

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = Path(__file__).parent / "core" / "fixtures" / "native_reload"
SUFFIX = ".pyd" if os.name == "nt" else ".so"


def _cargo():
    found = shutil.which("cargo")
    if found:
        return found
    default = Path.home() / ".cargo" / "bin" / "cargo"
    return str(default) if default.exists() else None


def _library(target: Path) -> Path:
    for name in (
        "native_reload.dll",
        "libnative_reload.dylib",
        "libnative_reload.so",
        "native_reload.so",
    ):
        path = target / "release" / name
        if path.exists():
            return path
    raise AssertionError(f"cargo left no native_reload library in {target / 'release'}")


@pytest.fixture(scope="session")
def builds(tmp_path_factory):
    native = pytest.importorskip("manifoldx._core_native")
    if not hasattr(native, "_StreamBuilder"):
        pytest.skip("native stream builder is unavailable")
    cargo = _cargo()
    if cargo is None:
        pytest.skip("cargo is unavailable for the bounded native fixture")
    out = tmp_path_factory.mktemp("native-reload-builds")
    target = ROOT / "target" / "native-reload-fixture"
    built = {}
    for variant in ("1", "2", "abi"):
        env = {
            **os.environ,
            "PYO3_PYTHON": sys.executable,
            "CARGO_TARGET_DIR": str(target),
            "MANIFOLDX_NATIVE_VARIANT": variant,
        }
        result = subprocess.run(
            [cargo, "build", "--release", "--manifest-path", str(FIXTURE / "Cargo.toml")],
            capture_output=True,
            text=True,
            env=env,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        folder = out / variant
        folder.mkdir()
        path = folder / f"native_reload{SUFFIX}"
        shutil.copy2(_library(target), path)
        built[variant] = path
    built["bytes"] = {key: value.read_bytes() for key, value in built.items() if key != "bytes"}
    return built


def material(result):
    return result[0][0][0]


def row(result):
    return struct.unpack("<I", result[1][0][2][:4])[0]


def test_two_valid_builds_and_an_abi_mismatch(builds):
    from manifoldx.native_binary import NativeBuilder

    builder = NativeBuilder(builds["1"])
    first = builder.build(0.0, 0.0, 1000.0, 0, 7)
    assert material(first) == 1 and row(first) == 1
    first_version = builder.version

    builder.reload(builds["2"])
    second = builder.build(0.0, 0.0, 1000.0, 0, 7)
    assert material(second) == 2 and row(second) == 2
    assert builder.version != first_version

    with pytest.raises(ValueError, match="expected builder capsule manifoldx.builder.v1"):
        builder.reload(builds["abi"])
    assert builder.version != first_version
    assert material(builder.build(0.0, 0.0, 1000.0, 0, 7)) == 2
    # Both libraries stay mapped: a rejection never unloads the last valid one.
    assert len(builder._libraries) == 3
    assert all(path.suffix == SUFFIX for path in builder._cache.iterdir())


def test_inflight_build_finishes_on_the_old_library(builds):
    from manifoldx.native_binary import NativeBuilder

    builder = NativeBuilder(builds["1"])
    with ThreadPoolExecutor(max_workers=1) as pool:
        running = pool.submit(builder.build, -1.0, 0.0, 1000.0, 0, 7)
        time.sleep(0.03)
        builder.reload(builds["2"])
        assert material(builder.build(0.0, 0.0, 1000.0, 0, 7)) == 2
        assert material(running.result(timeout=2)) == 1


def test_native_cli_reloads_a_build_at_the_frame_boundary(builds, tmp_path, monkeypatch):
    from manifoldx._desktop import _NativeEngine

    library = tmp_path / f"native_reload{SUFFIX}"
    library.write_bytes(builds["bytes"]["1"])
    app = tmp_path / "app.py"
    app.write_text(
        textwrap.dedent(f"""\
        import os
        from manifoldx import Engine, Material, Mesh, Transform
        from manifoldx.native_binary import NativeBuilder
        from manifoldx.resources import BasicMaterial, cube
        from manifoldx.streaming import Streamer

        engine = Engine('E8 native builder reload', width=24, height=24, max_entities=64)
        builder = NativeBuilder({str(library)!r})
        streamer = Streamer(
            engine, cell=1000.0, build=builder, radius=1, ready_radius=1,
            convert=lambda request, rows: ((), ()),
            materials={{1: BasicMaterial(color='#e83728'), 2: BasicMaterial(color='#28dc44')}},
        )
        engine.e8 = {{'builder': builder, 'streamer': streamer, 'x': 0.0}}

        @engine.on('startup')
        def startup(_):
            engine.entity = engine.spawn(
                Transform(), Mesh(cube(1, 1, 1)), Material(BasicMaterial(color='#fff'))
            )

        @engine.system
        def follow(query, dt):
            builder.poll()
            streamer.focus(engine.e8['x'], 0.0)

        engine.run()
        """)
    )

    stamp = time.time_ns()

    def replace(data):
        nonlocal stamp
        library.write_bytes(data)
        stamp += 1_000_000
        os.utime(library, ns=(stamp, stamp))

    def run_offscreen(engine):
        assert isinstance(engine, _NativeEngine)
        builder, streamer = engine.e8["builder"], engine.e8["streamer"]
        world = engine.world
        try:

            def settle(limit=800):
                for _ in range(limit):
                    engine.render_frame()
                    if streamer.settled:
                        return
                    time.sleep(0.002)
                raise AssertionError(f"streamer did not settle: {streamer.stats()}")

            engine.render_frame()
            settle()
            entity = engine.entity
            assert streamer._ready[(0, 0)].parts[0][1] == 1
            assert material(builder.build(0.0, 0.0, 1000.0, 0, 7)) == 1
            first_version = builder.version

            replace(builds["bytes"]["2"])
            engine.e8["x"] = 2000.0
            settle()
            assert builder.version != first_version
            assert streamer._ready[(1, 0)].parts[0][1] == 1
            assert streamer._ready[(2, 0)].parts[0][1] == 2
            assert material(builder.build(0.0, 0.0, 1000.0, 0, 7)) == 2
            meshes = dict(engine._registries["Mesh"])
            alive = world._native.alive_count
            second_version = builder.version

            replace(builds["bytes"]["abi"])
            engine.render_frame()
            assert "expected builder capsule manifoldx.builder.v1" in str(builder.last_error)
            assert builder.version == second_version
            assert material(builder.build(0.0, 0.0, 1000.0, 0, 7)) == 2
            engine.render_frame()
            assert engine.world is world
            assert engine.entity is entity
            assert entity.transform.pos.tolist() == [0.0, 0.0, 0.0]
            assert world._native.alive_count == alive
            assert dict(engine._registries["Mesh"]) == meshes
            assert streamer._ready[(1, 0)].parts[0][1] == 1
            assert streamer._ready[(2, 0)].parts[0][1] == 2
            assert {(1, 0), (2, 0)} <= streamer.drawn()
        finally:
            engine.close()

    monkeypatch.setenv("MANIFOLDX_CORE", "rust")
    monkeypatch.setattr(_NativeEngine, "run", run_offscreen)
    assert main(["run", str(app)]) == 0
