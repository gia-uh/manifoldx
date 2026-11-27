"""Native CLI and its frame boundary, using the Rust scheduler and offscreen renderer."""

import os
import sys
import textwrap

import numpy as np
import pytest

from manifoldx.cli import main


@pytest.fixture
def native_cli(monkeypatch):
    native = pytest.importorskip("manifoldx._core_native")
    if not hasattr(native, "_NativeGraphics"):
        pytest.skip("native renderer is unavailable")
    from manifoldx._desktop import _NativeEngine

    class RejectGlfw:
        def find_spec(self, fullname, path=None, target=None):
            if fullname in {"glfw", "rendercanvas.glfw"}:
                pytest.fail(f"native CLI tried to import {fullname}")

    monkeypatch.setenv("MANIFOLDX_CORE", "rust")
    monkeypatch.delitem(sys.modules, "rendercanvas.glfw", raising=False)
    monkeypatch.delitem(sys.modules, "glfw", raising=False)
    monkeypatch.setattr(sys, "meta_path", [RejectGlfw(), *sys.meta_path])
    clock = iter(range(1000))
    # Pass each poll deadline without a sleep. The native scheduler uses its own clock.
    monkeypatch.setattr("manifoldx._native_reload.monotonic", clock.__next__)
    return _NativeEngine


@pytest.mark.parametrize("as_module", [False, True])
def test_native_cli_runs_normal_entry_and_preserves_argv(
    native_cli, tmp_path, monkeypatch, as_module
):
    app_dir = tmp_path / "e8_game" if as_module else tmp_path
    app_dir.mkdir(exist_ok=True)
    (app_dir / "__init__.py").touch()
    (app_dir / "settings.py").write_text("LABEL = 'e8 title'\n")
    path = app_dir / "app.py"
    settings_import = "from .settings" if as_module else "from settings"
    path.write_text(
        textwrap.dedent(f"""\
        import sys
        from pathlib import Path
        from manifoldx import Engine
        {settings_import} import LABEL

        if __name__ == "__main__":
            assert sys.argv[1:] == ["--fps", "30"]
            with Path({str(tmp_path / "entry.log")!r}).open('a') as log:
                log.write('entry\\n')
            Engine(LABEL, width=16, height=16).cli()
        """)
    )
    monkeypatch.chdir(tmp_path)

    def run_offscreen(engine):
        try:
            assert engine.title == "e8 title"
            assert engine.render_frame().shape == (16, 16, 3)
        finally:
            engine.close()

    monkeypatch.setattr(native_cli, "run", run_offscreen)
    target = ["-m", "e8_game.app"] if as_module else [str(path)]
    assert main(["run", *target, "--", "--fps", "30"]) == 0
    assert (tmp_path / "entry.log").read_text() == "entry\n"
    for name in ("settings", "e8_game.settings", "e8_game"):
        sys.modules.pop(name, None)


def _movement(version):
    return textwrap.dedent(f"""\
        from __future__ import annotations
        from manifoldx import Query, Transform

        def register(system):
            @system
            def move(rows: Query[Transform], dt):
                rows[Transform].pos = ({version} / 100, 0, 0)
        """)


def test_native_cli_swaps_complete_registration_and_keeps_scene(
    native_cli, tmp_path, monkeypatch, capsys
):
    app = tmp_path / "app.py"
    app.write_text(
        textwrap.dedent("""\
        from manifoldx import Engine, Material, Mesh, Transform
        from manifoldx.resources import BasicMaterial, cube

        engine = Engine('E8', width=24, height=24, max_entities=64)
        engine.reload_systems('e8_move', 'e8_extra')

        @engine.on('startup')
        def startup(_):
            engine.startups = getattr(engine, 'startups', 0) + 1
            engine.entity = engine.spawn(Transform(), Mesh(cube(1, 1, 1)), Material(BasicMaterial(color="#fff")))
            engine._graphics.stream_commit((1, 0, 3, 4), 1, [], (0, 0, 0), [(1, [])], [], True)

        engine.run()
        """)
    )
    movement = tmp_path / "e8_move.py"
    extra = tmp_path / "e8_extra.py"
    good_extra = "def register(system):\n    pass\n"
    extra.write_text(good_extra)
    timestamp = 1_700_000_000_000_000_000

    def write(path, source):
        nonlocal timestamp
        path.write_text(source)
        timestamp += 1000
        os.utime(path, ns=(timestamp, timestamp))

    write(movement, _movement(0))

    def run_offscreen(engine):
        world, graphics = engine.world, engine._graphics
        try:
            engine.render_frame()
            entity = engine.entity
            meshes = engine._registries["Mesh"].copy()
            assert len(meshes) == 1
            assert engine.last_frame["instances"] == 1
            assert engine.last_frame["streamed_cells"] == 1
            assert world._native.alive_count == 2
            assert graphics.stream_result(1) is True

            def frame(expected):
                engine.render_frame()
                np.testing.assert_allclose(entity.transform.pos, (expected / 100, 0, 0))
                assert engine.startups == 1
                assert engine.world is world and engine._graphics is graphics
                assert engine._registries["Mesh"] == meshes
                assert engine.last_frame["instances"] == 1
                assert engine.last_frame["streamed_cells"] == 1
                assert world._native.alive_count == 2

            for version in range(1, 11):
                write(movement, _movement(version))
                frame(version)

            for invalid, message in (
                ("def register(\n", "SyntaxError"),
                (
                    "def register(system):\n    system(lambda query: None)\n",
                    "too many positional arguments",
                ),
                (
                    "def register(system):\n    def f(query, dt): pass\n    system(f)\n    system(f)\n",
                    "already registered",
                ),
                ("from manifoldx import Engine\nEngine('candidate')\n", "cannot create an Engine"),
                (
                    "def register(system):\n    for _ in range(32):\n        system(lambda query, dt: None)\n",
                    "too many desktop systems",
                ),
                (
                    textwrap.dedent("""\
                    from manifoldx import Query
                    class Missing: pass
                    def register(system):
                        @system
                        def unknown(rows: Query[Missing], dt): pass
                    """),
                    "unregistered component",
                ),
            ):
                running = sys.modules["e8_move"]
                write(movement, _movement(99))
                write(extra, invalid)
                frame(10)
                assert sys.modules["e8_move"] is running
                assert message in capsys.readouterr().err

            # Saving only the broken file retries the complete generation.
            write(extra, good_extra)
            frame(99)
            write(movement, good_extra)
            frame(99)  # An empty registration removes the old callback.
            entity.transform.pos = (0.12, 0, 0)
            frame(12)
        finally:
            engine.close()

    monkeypatch.setattr(native_cli, "run", run_offscreen)
    try:
        assert main(["run", str(app)]) == 0
    finally:
        sys.modules.pop("e8_move", None)
        sys.modules.pop("e8_extra", None)
