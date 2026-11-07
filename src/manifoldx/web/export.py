"""manifoldx export --web: run the app's setup in Python, then write a folder the browser runs.

out/
  app/<module>.js(.map)   the transpiled modules; the script (or -m module) is __main__
  pyrt.js, manifoldx.js   the runtime
  replay.mjs              loads the folder in Node
  state.json              the snapshot, engine.state, the engine's size and camera, the GUI
  scene.glb               the scene (and scene.export-report.json, the glTF export's report)
  report.json             the roots, what was transpiled, what stayed setup, what was left out
  index.html              the page: serve the folder (python -m http.server) and open it
  browser.js, input.js, scene.js, gui.js, sourcemap.js, progress.js   the page's modules
  vendor/three/           three.js, as its SOURCE file says
  fonts/                  the GUI's font, DejaVu Sans Mono, as the desktop draws it
"""

from __future__ import annotations

import gzip
import hashlib
import html
import json
import os
import runpy
import shutil
import sys
import tempfile
import time
import types
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

JS = Path(__file__).parent / "js"
RUNTIME = ("pyrt.js", "manifoldx.js", "replay.mjs")
PAGE = ("browser.js", "input.js", "scene.js", "gui.js", "sourcemap.js", "progress.js")
FONT = Path(__file__).parents[1] / "viz" / "assets" / "DejaVuSansMono.ttf"
# three.js for the browser page, vendored as its SOURCE file says.
THREE_VERSION = "0.186.1"
VENDOR = Path(__file__).parent / "vendor" / "three"
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


@dataclass
class Built:
    """A version of the app built in memory: what write_version writes."""

    modules: dict[str, tuple[str, str]]  # name: (code, source map JSON with its sources set)
    state: dict
    report: dict  # report.json, and the glTF export's report as "scene_report"
    glb: bytes
    scene_sha: str  # sha256 of glb, hex, first 16
    title: str
    crossed: frozenset[str]  # the engine.state keys the snapshot wrote
    left_out: list  # SnapshotError, a key of engine.state that could not cross
    timings: dict[str, float]  # seconds: startup reach snapshot transpile gui scene


def build_export(
    engine, main, root, *, app_label, runtime="../pyrt.js", gpu=None, scene_camera=True
) -> Built:
    """Build the app's version from the engine its setup left; raises TranspileError (naming
    the file's real path), SnapshotError or GuiUnsupported. gpu is (adapter, device, format)
    of an engine to reuse the device of, for the startup on a headless frame. scene_camera=False
    leaves the camera out of the glb, so a camera that moved keeps the scene's hash."""
    from manifoldx.gltf.engine_export import export_engine

    from .gui import gui_spec
    from .reach import find_roots
    from .snapshot import snapshot
    from .transpile import reach_app, transpile_app
    from .types import LiveTypes, TranspileError, key_label, module_path

    timings: dict[str, float] = {}
    clock = time.perf_counter()

    def lap(name):
        nonlocal clock
        now = time.perf_counter()
        timings[name], clock = now - clock, now

    sources, paths = _sources(main, root)
    with _as_main(main):
        if engine._device is None and engine._event_bus._handlers.get("startup"):
            # fires startup and waits for it; load_texture needs a device
            engine._ensure_offscreen(1, gpu=gpu)
        lap("startup")
        roots = find_roots(engine)
        left_out: list = []
        try:
            reach = reach_app(sources, roots, LiveTypes(set(sources)))
            lap("reach")
            state = snapshot(roots, reach, engine, left_out)
            lap("snapshot")
            mods = transpile_app(sources, reach=reach, runtime=runtime)
            lap("transpile")
        except TranspileError as e:
            real = {module_path(name): path for name, path in paths.items()}
            raise TranspileError(real.get(e.path, e.path), e.line, e.what, e.root) from None
        gui = gui_spec(engine)
        lap("gui")

    names = sorted(sources, key=lambda n: (n != "__main__", n))
    state["modules"] = names
    state["gui"] = gui
    env = reach.env
    report = {
        "app": app_label,
        "roots": [{"kind": r.kind, "label": r.label, "event": r.event} for r in roots],
        "transpiled": sorted(key_label(k) for k in reach.functions),
        "setup": sorted(key_label(k) for k in set(env.functions) - reach.functions),
        "classes": sorted(reach.classes),
        "globals": sorted(f"{m}.{n}" for m, n in reach.globals),
        "modules": {name: f"app/{name}.js" for name in names},
        "state_left_out": [{"path": e.path_from_root, "why": e.what} for e in left_out],
    }
    modules = {}
    for name, mod in mods.items():
        smap = json.loads(mod.source_map)
        smap["sources"] = [paths[name]]
        modules[name] = (mod.code, json.dumps(smap))
    with tempfile.TemporaryDirectory() as tmp:
        # Not engine.export_gltf: its report would call the transpiled roots dropped.
        scene = export_engine(
            engine, Path(tmp) / "scene.glb", transpiled=[r.func for r in roots], camera=scene_camera
        )
        glb = (Path(tmp) / "scene.glb").read_bytes()
    report["scene_report"] = scene.to_dict()
    lap("scene")
    crossed = frozenset(k for k, _ in state["table"][state["state"]["__ref"]]["__dict"])
    return Built(
        modules=modules,
        state=state,
        report=report,
        glb=glb,
        scene_sha=hashlib.sha256(glb).hexdigest()[:16],
        title=str(engine.title),
        crossed=crossed,
        left_out=left_out,
        timings=timings,
    )


def write_static(out, dev=False) -> None:
    """The files every version shares: the page's modules, the runtime, three.js, the font;
    with dev also dev.js, the dev loop's side of the page."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    for f in RUNTIME + PAGE + (("dev.js",) if dev else ()):
        shutil.copy(JS / f, out / f)
    shutil.rmtree(out / "vendor" / "three", ignore_errors=True)
    shutil.copytree(VENDOR, out / "vendor" / "three")
    (out / "fonts").mkdir(exist_ok=True)
    shutil.copy(FONT, out / "fonts" / FONT.name)


def write_version(built: Built, out, scene_path="scene.glb") -> None:
    """app/, state.json and report.json into out, and the glb with its export report at
    scene_path (relative to out) unless a file is there already."""
    out = Path(out)
    shutil.rmtree(out / "app", ignore_errors=True)
    (out / "app").mkdir(parents=True)
    for name, (code, smap) in built.modules.items():
        (out / "app" / f"{name}.js").write_text(code)
        (out / "app" / f"{name}.js.map").write_text(smap)
    (out / "state.json").write_text(json.dumps(built.state, allow_nan=False))
    report = {k: v for k, v in built.report.items() if k != "scene_report"}
    (out / "report.json").write_text(json.dumps(report, indent=1) + "\n")
    glb = out / scene_path
    if not glb.exists():
        glb.parent.mkdir(parents=True, exist_ok=True)
        glb.write_bytes(built.glb)
        scene_report = json.dumps(built.report["scene_report"], indent=2) + "\n"
        glb.with_suffix(".export-report.json").write_text(scene_report)


def page_html(title, scene_bytes, scene_gzip_bytes, dev=False) -> str:
    """index.html for the app: its title, the scene's sizes for the progress line, and with
    dev the mark that makes the page follow manifoldx run --web."""
    page = (JS / "index.html").read_text()
    for key, value in {
        "title": html.escape(str(title)),
        "scene_bytes": str(scene_bytes),
        "scene_gzip_bytes": str(scene_gzip_bytes),
        "dev": '<meta name="mx-dev">' if dev else "",
    }.items():
        page = page.replace("{{" + key + "}}", value)
    return page


def export_web(module: str | None, script: str | None, out: Path, app_args: list[str]) -> int:
    """Write the folder for the app; 0 when written, 1 when the app's code cannot cross
    (nothing written), 2 when the app never called engine.run()."""
    from .gui import GuiUnsupported
    from .snapshot import SnapshotError
    from .types import TranspileError

    engine, main = capture_engine(module, script, app_args)
    if engine is None:
        _say(f"{module or script} finished without calling engine.run(); nothing written")
        return 2
    root = os.getcwd() if module else os.path.dirname(os.path.abspath(script))
    label = f"-m {module}" if module else os.path.relpath(os.path.abspath(main.__file__), root)
    try:
        built = build_export(engine, main, root, app_label=label)
    except (TranspileError, SnapshotError, GuiUnsupported) as e:
        _say(str(e))
        return 1
    out = Path(out)
    write_static(out)
    write_version(built, out)
    page = page_html(built.title, len(built.glb), len(gzip.compress(built.glb, 9)))
    (out / "index.html").write_text(page)
    for e in built.left_out:
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
