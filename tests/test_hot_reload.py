import os
import sys
import textwrap
import time

import pytest

import manifoldx as mx
from manifoldx.backends import get_offscreen_canvas
from manifoldx.hot import HotReloader, State

APP = """
import manifoldx as mx

engine = mx.Engine("hot", width=64, height=64)
engine.state.setdefault("loads", 0)
engine.state.loads += 1
VERSION = {version}

@engine.on("startup")
def startup(payload):
    engine.state.setdefault("frames", 0)
    engine.state.version = VERSION

@engine.on("frame")
def frame(payload):
    engine.state.frames += 1

engine.run()
"""


def offscreen(engine):
    try:
        return get_offscreen_canvas(engine.w, engine.h)
    except Exception as e:  # no wgpu backend on this machine
        pytest.skip(f"offscreen canvas unavailable: {e}")


_clock = [time.time()]


def write(path, source):
    path.write_text(textwrap.dedent(source))
    # Each write lands whole seconds later, so neither the watcher nor a cached
    # .pyc (which keys on mtime seconds and size) mistakes it for the old file.
    _clock[0] += 2
    os.utime(path, (_clock[0], _clock[0]))


@pytest.fixture
def app(tmp_path):
    path = tmp_path / "app.py"
    write(path, APP.format(version=1))
    return path


def test_state_is_a_dict_with_attribute_access():
    s = State()
    s.camera = {"tilt": 0.3}
    assert s["camera"] == {"tilt": 0.3}
    assert s.setdefault("step", 4) == 4 and s.step == 4
    with pytest.raises(AttributeError):
        s.missing


def test_plain_engine_has_its_own_state():
    a, b = mx.Engine("a"), mx.Engine("b")
    a.state.x = 1
    assert "x" not in b.state


def test_reload_keeps_state_and_window(app):
    hot = HotReloader(script=str(app), canvas_factory=offscreen, poll=0)
    hot.start()
    first = hot.engine
    for _ in range(3):
        hot.step()
    assert hot.state.frames == 3 and hot.state.version == 1

    write(app, APP.format(version=2))
    hot.step()
    assert hot.engine is not first
    assert hot.engine._render_canvas is first._render_canvas
    assert hot.engine._device is first._device
    assert not first._running
    hot.step()
    assert hot.state.loads == 2 and hot.state.version == 2
    assert hot.state.frames >= 4  # kept counting across the reload


def test_broken_edit_keeps_old_engine(app, capsys):
    hot = HotReloader(script=str(app), canvas_factory=offscreen, poll=0)
    hot.start()
    first = hot.engine
    hot.step()

    write(app, APP.format(version="1 +"))  # syntax error
    hot.step()  # reload fails, the old engine draws on
    assert hot.engine is first and first._running
    assert "SyntaxError" in capsys.readouterr().err
    assert hot.state.frames == 2

    write(app, APP.format(version=3))
    hot.step()
    assert hot.engine is not first


def test_error_in_frame_pauses_until_next_edit(app, capsys):
    hot = HotReloader(script=str(app), canvas_factory=offscreen, poll=0)
    hot.start()
    hot.step()

    write(app, APP.format(version=4).replace("engine.state.frames += 1", "1 / 0"))
    hot.step()  # reload; the new frame handler raises
    assert "ZeroDivisionError" in capsys.readouterr().err
    frames = hot.state.frames
    hot.step()
    assert capsys.readouterr().err == ""  # paused, not raising every frame
    assert hot.state.frames == frames

    write(app, APP.format(version=5))
    hot.step()
    assert hot.state.frames == frames + 1


def test_project_modules_are_reimported(tmp_path):
    write(tmp_path / "scene.py", "SIZE = 1\n")
    write(tmp_path / "app.py", APP.format(version="__import__('scene').SIZE"))
    hot = HotReloader(script=str(tmp_path / "app.py"), canvas_factory=offscreen, poll=0)
    hot.start()
    hot.step()
    assert hot.state.version == 1

    write(tmp_path / "scene.py", "SIZE = 7\n")
    hot.step()
    hot.step()
    assert hot.state.version == 7
    assert mx.Engine is __import__("manifoldx").Engine  # manifoldx itself never reloads


def test_editing_the_module_run_with_m_reloads_it(tmp_path, monkeypatch):
    (tmp_path / "watchpkg").mkdir()
    write(tmp_path / "watchpkg" / "__init__.py", "")
    write(tmp_path / "watchpkg" / "entry.py", APP.format(version=1))
    monkeypatch.chdir(tmp_path)
    hot = HotReloader(module="watchpkg.entry", canvas_factory=offscreen, poll=0, channel=False)
    try:
        hot.start()
        hot.step()
        assert hot.state.version == 1

        write(tmp_path / "watchpkg" / "entry.py", APP.format(version=2))
        hot.step()
        hot.step()
        assert hot.state.version == 2
    finally:
        sys.modules.pop("watchpkg", None)


def test_from_import_of_a_namespace_package_module_gets_the_new_code(tmp_path, monkeypatch):
    (tmp_path / "nspkg").mkdir()  # no __init__.py: a namespace package, which never reloads itself
    write(tmp_path / "nspkg" / "sub.py", "VALUE = 1\n")
    write(tmp_path / "app.py", APP.format(version="__import__('importlib').import_module('appglue').VALUE"))
    write(tmp_path / "appglue.py", "from nspkg import sub\nVALUE = sub.VALUE\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    hot = HotReloader(script=str(tmp_path / "app.py"), canvas_factory=offscreen, poll=0, channel=False)
    try:
        hot.start()
        hot.step()
        assert hot.state.version == 1

        write(tmp_path / "nspkg" / "sub.py", "VALUE = 2\n")
        hot.step()
        hot.step()
        assert hot.state.version == 2
    finally:
        for name in ("nspkg", "nspkg.sub", "appglue"):
            sys.modules.pop(name, None)


def test_reload_cancels_old_async_handlers(tmp_path):
    source = APP.replace(
        '@engine.on("frame")',
        '@engine.on("startup")\nasync def ticker(payload):\n    while True:\n        await engine.tick()\n        engine.state.ticks = engine.state.get("ticks", 0) + VERSION\n\n@engine.on("frame")',
    )
    path = tmp_path / "app.py"
    write(path, source.format(version=1))
    hot = HotReloader(script=str(path), canvas_factory=offscreen, poll=0)
    hot.start()
    for _ in range(3):
        hot.step()
    old_tasks = set(hot.engine._tasks)
    assert old_tasks

    write(path, source.format(version=1000))
    for _ in range(3):
        hot.step()
    assert all(t.done() for t in old_tasks)
    assert hot.state.ticks % 1000 < 10  # only the new ticker still counts


def test_a_launcher_in_the_projects_venv_stays_main_across_reloads(tmp_path, monkeypatch):
    # `uv run manifoldx run -m pkg.mod`: __main__ is .venv/bin/manifoldx, inside the project folder.
    import types

    (tmp_path / "venvpkg").mkdir()
    write(tmp_path / "venvpkg" / "__init__.py", "")
    write(tmp_path / "venvpkg" / "entry.py", APP.format(version=1))
    (tmp_path / ".venv" / "bin").mkdir(parents=True)
    write(tmp_path / ".venv" / "bin" / "manifoldx", "")
    launcher = types.ModuleType("__main__")
    launcher.__file__ = str(tmp_path / ".venv" / "bin" / "manifoldx")
    monkeypatch.setitem(sys.modules, "__main__", launcher)
    monkeypatch.setattr(sys, "prefix", str(tmp_path / ".venv"))
    monkeypatch.chdir(tmp_path)
    hot = HotReloader(module="venvpkg.entry", canvas_factory=offscreen, poll=0, channel=False)
    try:
        hot.start()
        hot.step()
        write(tmp_path / "venvpkg" / "entry.py", APP.format(version=2))
        hot.step()
        hot.step()
        assert hot.state.version == 2
        assert sys.modules["__main__"] is launcher
    finally:
        sys.modules.pop("venvpkg", None)
