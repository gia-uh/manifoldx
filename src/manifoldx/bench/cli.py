"""`manifoldx bench`: list, run and show an app's scenarios, and time micro cases.

    manifoldx bench list [--json]
    manifoldx bench run NAME|FILE.json [--size WxH] [--loop serial|pipelined|check] [--step 1/30]
                        [--param k=v]... [--smoke] [--label L] [--json]
    manifoldx bench show RUN [--json]
    manifoldx bench micro [MODULE...] [--case C] [--sizes N...] [--profile] [--smoke] [--against DIR] [--json]

The app's pyproject.toml says where its scenarios are:

    [tool.manifoldx.bench]
    scenarios = "myapp.scenarios"     # a module of @bench.scenario and @bench.step functions
    runs = "bench/runs"               # one folder per run
    history = "bench/history.jsonl"   # one line per run
    micro = ["myapp.bench_cases"]     # modules of @bench.case functions
    folders = ["cache"]              # folders under the project sized at each run's start and end

Exit codes: 0 the run passed, 1 it failed, 2 bad usage or configuration. Progress goes to stderr,
at most once a second; with --json, stdout holds only the JSON.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
import time
import tomllib
from dataclasses import dataclass, field
from pathlib import Path


class ConfigError(Exception):
    """The project's configuration is missing or wrong; the message says what to add."""


@dataclass
class Config:
    """[tool.manifoldx.bench] of the nearest pyproject.toml, with its defaults."""

    root: Path
    scenarios: str | None
    runs: Path
    history: Path
    micro: list = field(default_factory=list)
    app_name: str | None = None
    has_table: bool = True
    folders: dict = field(default_factory=dict)  # name -> folder under root, sized by each run


def load_config(start: Path) -> Config:
    """Read the nearest pyproject.toml at or above `start`."""
    start = Path(start).resolve()
    for d in (start, *start.parents):
        if (d / "pyproject.toml").exists():
            data = tomllib.loads((d / "pyproject.toml").read_text())
            t = data.get("tool", {}).get("manifoldx", {}).get("bench")
            table = t or {}
            return Config(d, table.get("scenarios"), d / table.get("runs", "bench/runs"),
                          d / table.get("history", "bench/history.jsonl"), list(table.get("micro", [])),
                          data.get("project", {}).get("name"), t is not None, _folders(d, table.get("folders", [])))
    raise ConfigError(f"no pyproject.toml in {start} or above; manifoldx bench reads [tool.manifoldx.bench] from it")


def _folders(root: Path, names: list) -> dict:
    out = {}
    for name in names:
        path = (root / name).resolve()
        if not path.is_relative_to(root):
            raise ConfigError(f"folders: {name!r} is outside the project ({root}); name folders under it")
        out[str(name)] = path
    return out


def _import_scenarios(config: Config) -> None:
    if config.scenarios:
        if str(config.root) not in sys.path:
            sys.path.insert(0, str(config.root))
        try:
            importlib.import_module(config.scenarios)
        except ImportError as e:
            raise ConfigError(f"cannot import the scenarios module {config.scenarios!r} named in "
                              f"{config.root / 'pyproject.toml'}: {e}") from None


def _size(text: str):
    try:
        w, h = (int(x) for x in text.lower().split("x"))
        return w, h
    except ValueError:
        raise argparse.ArgumentTypeError(f"size must be WxH, such as 640x360, not {text!r}") from None


def _step(text: str) -> float:
    try:
        a, _, b = text.partition("/")
        dt = float(a) / float(b or 1)
    except (ValueError, ZeroDivisionError):
        dt = 0.0
    if dt <= 0:
        raise argparse.ArgumentTypeError(f"step must be a positive number of seconds, such as 1/30, not {text!r}")
    return dt


def _parser():
    p = argparse.ArgumentParser(prog="manifoldx bench", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    ls = sub.add_parser("list", help="scenarios and steps")
    ls.add_argument("--json", action="store_true")
    r = sub.add_parser("run", help="run a scenario and write its run folder")
    r.add_argument("scenario", help="a scenario's name, or a JSON scenario file")
    r.add_argument("--size", type=_size, default=(640, 360), help="WxH of the offscreen frame (default 640x360)")
    r.add_argument("--loop", choices=["serial", "pipelined", "check"], default="serial")
    r.add_argument("--step", type=_step, help="fixed dt in seconds, such as 1/30 (default: the wall clock)")
    r.add_argument("--param", action="append", default=[], metavar="K=V")
    r.add_argument("--smoke", action="store_true", help="cap each waiting step at the scenario's smoke size")
    r.add_argument("--label", help="a free label kept with the run (default: $MANIFOLDX_BENCH_LABEL)")
    r.add_argument("--json", action="store_true", help="print only the summary, as JSON")
    s = sub.add_parser("show", help="print a run's summary")
    s.add_argument("run", help="a run folder, or its name under the runs folder")
    s.add_argument("--json", action="store_true")
    m = sub.add_parser("micro", help="time micro cases")
    m.add_argument("modules", nargs="*", help="modules of @bench.case functions (default: the config's micro)")
    m.add_argument("--case", action="append")
    m.add_argument("--sizes", nargs="+", type=int)
    m.add_argument("--profile", action="store_true")
    m.add_argument("--smoke", action="store_true")
    m.add_argument("--against", type=Path, metavar="DIR", help="another manifoldx project to compare with")
    m.add_argument("--json", action="store_true")
    return p


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        return {"list": _list, "run": _run, "show": _show, "micro": _micro}[args.cmd](args)
    except ConfigError as e:
        print(f"manifoldx bench: {e}", file=sys.stderr)
        return 2


def _list(args) -> int:
    from manifoldx.bench.scenario import SCENARIOS, STEPS

    try:
        _import_scenarios(load_config(Path.cwd()))
    except ConfigError:
        pass  # the engine's steps list anyway
    scenarios = [{"name": s.name, "params": s.params, "sizes": s.sizes, "doc": s.doc} for s in SCENARIOS.values()]
    steps = {"engine": sorted(n for n, d in STEPS.items() if d.engine),
             "app": sorted(n for n, d in STEPS.items() if not d.engine)}
    if args.json:
        print(json.dumps({"scenarios": scenarios, "steps": steps}, indent=1))
        return 0
    for s in scenarios:
        print(f"{s['name']:<24} {s['doc']}  params={s['params']}")
    print("engine steps: " + ", ".join(steps["engine"]))
    if steps["app"]:
        print("app steps:    " + ", ".join(steps["app"]))
    return 0


def _run(args) -> int:
    from manifoldx.bench.folder import write_run
    from manifoldx.bench.runner import run_scenario
    from manifoldx.bench.scenario import SCENARIOS, load_json

    config = load_config(Path.cwd())
    if not config.has_table:
        raise ConfigError(f"{config.root / 'pyproject.toml'} has no [tool.manifoldx.bench]; add one with "
                          "scenarios = \"<module of your @bench.scenario functions>\"")
    _import_scenarios(config)  # a JSON scenario may call the app steps it registers
    if args.scenario.endswith(".json"):
        try:
            sc = load_json(Path(args.scenario))
        except (OSError, ValueError, KeyError) as e:
            raise ConfigError(f"cannot read the scenario file {args.scenario}: {e}") from None
    else:
        if args.scenario not in SCENARIOS:
            raise ConfigError(f"no scenario {args.scenario!r}; known: {', '.join(sorted(SCENARIOS)) or 'none'}")
        sc = SCENARIOS[args.scenario]
    overrides = {}
    for kv in args.param:
        k, sep, v = kv.partition("=")
        if not sep:
            raise ConfigError(f"--param takes k=v, not {kv!r}")
        overrides[k] = v
    last = {"t": 0.0}

    def progress(n, phase, step, steps):
        now = time.monotonic()
        if now - last["t"] >= 1.0 or n == 1:
            last["t"] = now
            print(f"frame {n} phase {phase} step {step}/{steps}", file=sys.stderr, flush=True)

    import contextlib

    quiet = contextlib.redirect_stdout(sys.stderr) if args.json else contextlib.nullcontext()
    try:
        with quiet:  # with --json, whatever the app prints goes to stderr
            run = run_scenario(sc, size=args.size, loop=args.loop, step_dt=args.step, overrides=overrides,
                               smoke=args.smoke, label=args.label or os.environ.get("MANIFOLDX_BENCH_LABEL"),
                               app_name=config.app_name, progress=progress, folders=config.folders)
    except ValueError as e:  # bad params or loop
        raise ConfigError(str(e)) from None
    folder = write_run(run, config.runs, config.history)
    summary = {**run.summary, "folder": folder.name}
    if args.json:
        print(json.dumps(summary, indent=1))
    else:
        _table(summary, folder)
    if run.error:
        print(f"manifoldx bench: the run failed: {run.error}", file=sys.stderr)
        return 1
    return 0


def _table(summary: dict, folder: Path) -> None:
    host = summary["host"]
    print(f"{summary['scenario']}  loop={summary['loop']}  size={summary['size'][0]}x{summary['size'][1]}  "
          f"{host['adapter']}{' (software)' if host['software'] else ''}  load {host['load_before']}–{host['load_max']}")
    print(f"{'phase':<14}{'frames':>7}{'p50 ms':>9}{'p95 ms':>9}{'max ms':>9}{'draw p50':>9}{'cpu p50':>9}"
          "  slowest systems")
    for name, ph in summary["phases"].items():
        top = ", ".join(f"{k.rsplit('.', 1)[-1]} {v}" for k, v in list(ph["systems_ms"].items())[:3])
        print(f"{name:<14}{ph['frames']:>7}{ph['p50_ms']:>9.1f}{ph['p95_ms']:>9.1f}{ph['max_ms']:>9.1f}"
              f"{ph['draw_p50_ms']:>9.1f}{ph['cpu_p50_ms']:>9.1f}  {top}")
    print(f"run folder: {folder}")


def _show(args) -> int:
    from manifoldx.bench.folder import read_run

    folder = Path(args.run)
    if not folder.exists():
        folder = load_config(Path.cwd()).runs / args.run
    if not (folder / "run.json").exists():
        raise ConfigError(f"no run at {folder}")
    summary = read_run(folder)
    if args.json:
        print(json.dumps(summary, indent=1))
    else:
        _table(summary, folder)
    return 0


def _micro(args) -> int:
    from manifoldx.bench import micro

    config = None
    modules = args.modules
    if not modules:
        config = load_config(Path.cwd())
        modules = config.micro
        if not modules:
            raise ConfigError("no micro modules: name some, or list them in [tool.manifoldx.bench] micro")
    cwd = (config or load_config(Path.cwd())).root
    if str(cwd) not in sys.path:
        sys.path.insert(0, str(cwd))
    if args.against:
        rest = [x for c in args.case or [] for x in ("--case", c)]
        rest += (["--sizes", *map(str, args.sizes)] if args.sizes else []) + (["--smoke"] if args.smoke else [])
        try:
            base, head = micro.against_results(modules, args.against, rest, cwd=cwd)
        except micro.AgainstError as e:
            raise ConfigError(str(e)) from None
        table = micro.compare(base, head, Path(args.against).name)
        print(json.dumps({"base": base, "head": head, "table": table}, indent=1) if args.json else table)
        return 0
    cases = [c for c in micro.load(modules) if not args.case or c.name in args.case]
    if args.json:
        import contextlib

        with contextlib.redirect_stdout(sys.stderr):
            results = micro.run(cases, args.sizes, args.profile, args.smoke)
    else:
        results = micro.run(cases, args.sizes, args.profile, args.smoke)
    if args.json:
        from dataclasses import asdict

        print(json.dumps([asdict(r) for r in results], indent=1))
    return 1 if any(r.error for r in results) else 0
