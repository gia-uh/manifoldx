"""`manifoldx asset FILE COMMAND`: run one asset file's stages and write what the command asks for.

The command parses arguments, calls :class:`manifoldx.asset.Asset` and the layer-2 functions :func:`save` and
:func:`inspect`, and has no logic of its own: whatever it does, a Python caller does with the same names.
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

from manifoldx.asset.base import Asset
from manifoldx.asset import blender
from manifoldx.asset.inspect import inspect as inspect_scene
from manifoldx.asset.write import save

COMMANDS = ("draw", "build", "lint", "inspect", "paint", "render")

# What each command runs. Lint comes after build, and paint after lint: paint's A-CODE and A-OUTLINE findings
# then join the same report, as Asset.run appends them when the paint stage runs.
_STAGES = {
    "draw": ("draw",),
    "build": ("build", "lint"),
    "lint": ("build", "lint"),
    "inspect": ("build", "lint"),
    "paint": ("build", "lint", "paint"),
    "render": ("build", "lint", "paint"),
}
_SAVES = ("build", "paint", "render")
_REPORTS = ("build", "lint")


def add_parser(sub):
    parser = sub.add_parser("asset", help="run an asset file's stages: draw, build, lint, inspect, paint, render")
    parser.add_argument("file", metavar="FILE", help="the Python file that defines one Asset subclass")
    parser.add_argument("action", metavar="COMMAND", choices=COMMANDS)
    parser.add_argument("--asset", metavar="NAME", help="which Asset subclass, when the file defines several")
    parser.add_argument("--json", action="store_true", help="print the report as JSON (build and lint)")
    parser.add_argument("--budget", type=float, metavar="DOLLARS", help="cap the spend of the paid calls")
    parser.add_argument("--model", metavar="ID", help="the image model to reach")
    parser.add_argument("--again", action="append", default=[], metavar="NAME",
                        help="forget cached images: an image, a view, or a part's views (repeatable)")
    parser.add_argument("--max-tris", type=int, metavar="N", help="warn above this many triangles in a part")
    parser.add_argument("--lod", choices=("low",), help="Parts: a coarse mesh with few rings and segments")
    parser.add_argument("--out", metavar="DIR", help="where to write (default NAME.asset/, or NAME.asset/low/)")
    return parser


def load(path, name: str | None = None):
    """The one :class:`Asset` subclass a Python file defines, or the one ``name`` selects."""
    file = Path(path)
    if not file.exists():
        raise ValueError(f"no such file: {path}")
    spec = importlib.util.spec_from_file_location(f"_manifoldx_asset_{file.stem}", file)
    if spec is None or spec.loader is None:
        raise ValueError(f"{path} is not a Python file")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    sys.path.insert(0, str(file.resolve().parent))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.pop(0)
    found = [value for value in vars(module).values()
             if isinstance(value, type) and value is not Asset and issubclass(value, Asset)
             and value.__module__ == module.__name__]
    if name is not None:
        found = [value for value in found if value.__name__ == name or value.name == name]
        if not found:
            raise ValueError(f"{path} defines no asset called {name!r}")
    if not found:
        raise ValueError(f"{path} defines no Asset subclass")
    if len(found) > 1:
        names = ", ".join(value.__name__ for value in found)
        raise ValueError(f"{path} defines several Asset subclasses ({names}); choose one with --asset NAME")
    return found[0]()


def run(args) -> int:
    """Run one command of one asset file; returns 0 on success, 1 on a finding, 2 on a usage error."""
    try:
        asset = load(args.file, args.asset)
        if args.action == "render" and not blender.available():
            raise blender.BlenderMissing("no blender executable is on PATH; install Blender to render stills")
        if args.lod is not None:
            asset.lod = args.lod
        out = Path(args.out) if args.out is not None else (
            asset.cache_dir() / "low" if args.lod == "low" else asset.cache_dir())
        result = asset.run(stages=_STAGES[args.action], model=args.model, budget=args.budget,
                           max_tris=args.max_tris, again=tuple(args.again))
        if args.action in _SAVES:
            save(result, out)
        if args.action == "inspect":
            for path in inspect_scene(asset, result.context, result.scene, out, report=result.report):
                print(path)
        if args.action == "render":
            style = "photo" if getattr(asset, "photoreal", False) else getattr(asset, "render_style", "toon")
            model, albedo = out / "model.glb", out / "albedo.png"
            for path in blender.render(model, out, style=style, albedo=albedo):
                print(path)
        if args.json and args.action in _REPORTS:
            print(result.report.to_json())
        elif args.action in _REPORTS:
            print(result.report.text() or "no findings")
        return result.report.exit_code
    except (FileNotFoundError, ValueError, TypeError, RuntimeError, subprocess.CalledProcessError) as exc:
        print(f"manifoldx asset: {exc}", file=sys.stderr)
        return 2
