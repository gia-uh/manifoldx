"""manifoldx export --web: run the app's setup in Python, then write a folder the browser runs.

out/
  app/<module>.js(.map)   the transpiled modules; the script (or -m module) is __main__
  pyrt.js, manifoldx.js   the runtime
  replay.mjs              loads the folder in Node
  state.json              the snapshot, engine.state, the engine's size and camera
  scene.glb               the scene (and scene.export-report.json, the glTF export's report)
  report.json             the roots, what was transpiled, what stayed setup, what was left out
"""

from __future__ import annotations

import json
import os
import runpy
import shutil
import sys
import types
from contextlib import contextmanager
from pathlib import Path

JS = Path(__file__).parent / "js"
RUNTIME = ("pyrt.js", "manifoldx.js", "replay.mjs")
_MANIFOLDX = str(Path(__file__).parents[1]) + os.sep


def capture_engine(module: str | None, script: str | None, app_args: list[str]):
    """Run the app (a script, or a module as python -m runs it) until it calls engine.run(),
    which returns at once: (the last engine that called it, the app's __main__ module), or
    (None, None) when it never did."""
    from manifoldx.engine import Engine

    captured = []

    def run(engine):
        captured.append((engine, sys.modules.get("__main__")))

    original, saved_argv, saved_path = Engine.run, sys.argv, list(sys.path)
    Engine.run = run
    sys.argv = [module or script, *app_args]
    # What python itself puts first on sys.path: the script's folder, or the working directory for -m.
    sys.path.insert(0, os.getcwd() if module else os.path.dirname(os.path.abspath(script)))
    try:
        if module:
            runpy.run_module(module, run_name="__main__", alter_sys=True)
        else:
            runpy.run_path(script, run_name="__main__")
    finally:
        Engine.run, sys.argv = original, saved_argv
        sys.path[:] = saved_path
    return captured[-1] if captured else (None, None)


def export_web(module: str | None, script: str | None, out: Path, app_args: list[str]) -> int:
    """Write the folder for the app; 0 when written, 1 when the app's code cannot cross
    (nothing written), 2 when the app never called engine.run()."""
    from .reach import find_roots
    from .snapshot import SnapshotError, snapshot
    from .transpile import reach_app, transpile_app
    from .types import LiveTypes, TranspileError, key_label, module_path

    engine, main = capture_engine(module, script, app_args)
    if engine is None:
        _say(f"{module or script} finished without calling engine.run(); nothing written")
        return 2
    root = os.getcwd() if module else os.path.dirname(os.path.abspath(script))
    sources, paths = _sources(main, root)
    with _as_main(main):
        if engine._device is None and engine._event_bus._handlers.get("startup"):
            engine._ensure_offscreen(
                1
            )  # fires startup and waits for it; load_texture needs a device
        roots = find_roots(engine)
        left_out: list[SnapshotError] = []
        try:
            reach = reach_app(sources, roots, LiveTypes(set(sources)))
            state = snapshot(roots, reach, engine, left_out)
            mods = transpile_app(sources, reach=reach, runtime="../pyrt.js")
        except TranspileError as e:
            real = {module_path(name): path for name, path in paths.items()}
            _say(str(TranspileError(real.get(e.path, e.path), e.line, e.what, e.root)))
            return 1
        except SnapshotError as e:
            _say(str(e))
            return 1

    names = sorted(sources, key=lambda n: (n != "__main__", n))
    state["modules"] = names
    env = reach.env
    report = {
        "app": f"-m {module}" if module else paths["__main__"],
        "roots": [{"kind": r.kind, "label": r.label, "event": r.event} for r in roots],
        "transpiled": sorted(key_label(k) for k in reach.functions),
        "setup": sorted(key_label(k) for k in set(env.functions) - reach.functions),
        "classes": sorted(reach.classes),
        "globals": sorted(f"{m}.{n}" for m, n in reach.globals),
        "modules": {name: f"app/{name}.js" for name in names},
        "state_left_out": [{"path": e.path_from_root, "why": e.what} for e in left_out],
    }

    out = Path(out)
    shutil.rmtree(out / "app", ignore_errors=True)
    (out / "app").mkdir(parents=True)
    for name, mod in mods.items():
        smap = json.loads(mod.source_map)
        smap["sources"] = [paths[name]]
        (out / "app" / f"{name}.js").write_text(mod.code)
        (out / "app" / f"{name}.js.map").write_text(json.dumps(smap))
    for f in RUNTIME:
        shutil.copy(JS / f, out / f)
    (out / "state.json").write_text(json.dumps(state, allow_nan=False))
    engine.export_gltf(out / "scene.glb")
    (out / "report.json").write_text(json.dumps(report, indent=1) + "\n")
    for e in left_out:
        _say(f"left out of the state: {e}")
    return 0


def _say(line: str) -> None:
    print(f"manifoldx export --web: {line}", file=sys.stderr)


def _sources(main: types.ModuleType, root: str) -> tuple[dict[str, str], dict[str, str]]:
    """The app's modules, {name: source}, and their paths from root, {name: path}: __main__
    (the script or the -m module), and every module loaded from root's folder outside
    site-packages and manifoldx, the files `manifoldx run` watches."""
    root = os.path.abspath(root) + os.sep
    entry = os.path.abspath(main.__file__)
    found = {"__main__": entry}
    for name, m in list(sys.modules.items()):
        f = getattr(m, "__file__", None)
        if name == "__main__" or not f or not f.endswith(".py"):
            continue
        f = os.path.abspath(f)
        owned = f.startswith(root) and "site-packages" not in f and not f.startswith(_MANIFOLDX)
        if owned and f != entry:
            found[name] = f
    sources = {name: Path(f).read_text() for name, f in found.items()}
    return sources, {name: os.path.relpath(f, root) for name, f in found.items()}


@contextmanager
def _as_main(main: types.ModuleType):
    """The app's module as __main__ again while the export reads its globals."""
    saved = sys.modules.get("__main__")
    sys.modules["__main__"] = main
    try:
        yield
    finally:
        sys.modules["__main__"] = saved
