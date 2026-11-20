"""`manifoldx bench`: list, run and show an app's scenarios, and time micro cases.

    manifoldx bench list [--json]
    manifoldx bench run NAME|FILE.json|--all [--size WxH] [--loop serial|pipelined|check] [--step 1/30]
                        [--param k=v[,v2...]]... [--repeat N] [--against DIR | --manifoldx SPEC]
                        [--smoke] [--label L] [--json]
    manifoldx bench check [NAME...] [--full] [--size WxH] [--json]
    manifoldx bench show RUN [--probe P [--phase NAME]] [--json]
    manifoldx bench report RUN [--open]
    manifoldx bench compare PATH [PATH...] [--out DIR] [--json]
    manifoldx bench clean [--json]
    manifoldx bench micro [MODULE...] [--case C] [--sizes N...] [--profile] [--smoke] [--against DIR [--repeat N]]
                          [--json]

The app's pyproject.toml says where its scenarios are:

    [tool.manifoldx.bench]
    scenarios = "myapp.scenarios"     # a module of @bench.scenario and @bench.step functions
    runs = "bench/runs"               # one folder per run
    history = "bench/history.jsonl"   # one line per run
    micro = ["myapp.bench_cases"]     # modules of @bench.case functions
    folders = ["cache"]              # folders under the project sized at each run's start and end
    budgets = "bench/budgets.toml"    # more checks: one [scenario] table of budgets each

Exit codes: 0 the run passed, 1 it failed or a check failed, 2 bad usage or configuration. Progress goes to stderr,
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
    budgets: Path | None = None  # the budgets file: more checks, by scenario


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
                          data.get("project", {}).get("name"), t is not None, _folders(d, table.get("folders", [])),
                          d / table["budgets"] if table.get("budgets") else None)
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


def _budgets(config: Config) -> dict:
    """The budgets file's checks by scenario, once the scenarios are imported; {} without a file."""
    from manifoldx.bench.checks import read_budgets
    from manifoldx.bench.scenario import SCENARIOS

    if config.budgets is None:
        return {}
    try:
        out = read_budgets(config.budgets)
    except (OSError, ValueError) as e:  # tomllib's error is a ValueError
        raise ConfigError(f"cannot read the budgets file {config.budgets}: {e}") from None
    unknown = [n for n in out if n not in SCENARIOS]
    if unknown:
        raise ConfigError(f"{config.budgets} names {', '.join(unknown)}, which no scenario is; known: "
                          f"{', '.join(sorted(SCENARIOS)) or 'none'}")
    return out


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
    r.add_argument("scenario", nargs="?", help="a scenario's name, or a JSON scenario file")
    r.add_argument("--all", action="store_true", help="run every scenario, one after another")
    r.add_argument("--size", type=_size, default=(640, 360), help="WxH of the offscreen frame (default 640x360)")
    r.add_argument("--loop", choices=["serial", "pipelined", "check"], default="serial")
    r.add_argument("--step", type=_step, help="fixed dt in seconds, such as 1/30 (default: the wall clock)")
    r.add_argument("--param", action="append", default=[], metavar="K=V",
                   help="a param's value; K=A,B sweeps it, each value a side run in fresh processes")
    r.add_argument("--repeat", type=int, default=1, metavar="N",
                   help="runs of each side, interleaved (with a sweep, --against or --manifoldx)")
    r.add_argument("--against", type=Path, metavar="DIR",
                   help="another copy of the app to run the same scenario on, through uv (the baseline)")
    r.add_argument("--manifoldx", metavar="SPEC",
                   help="run this copy on another manifoldx as well: anything uv run --with takes, or a directory")
    r.add_argument("--smoke", action="store_true", help="cap each waiting step at the scenario's smoke size")
    r.add_argument("--label", help="a free label kept with the run (default: $MANIFOLDX_BENCH_LABEL)")
    r.add_argument("--json", action="store_true", help="print only the summary, as JSON")
    k = sub.add_parser("check", help="run scenarios at smoke size and fail on any check that fails")
    k.add_argument("names", nargs="*", help="scenarios (default: every one with checks)")
    k.add_argument("--full", action="store_true", help="run each step in full, not capped at the smoke size")
    k.add_argument("--size", type=_size, default=(160, 90), help="WxH of the offscreen frame (default 160x90)")
    k.add_argument("--json", action="store_true", help="print only the results, as JSON")
    s = sub.add_parser("show", help="print a run's summary")
    s.add_argument("run", help="a run folder, or its name under the runs folder")
    s.add_argument("--probe", choices=sorted([*PHASE_PROBES, *RUN_PROBES]), help="print only this probe")
    s.add_argument("--phase", help="only this phase (with --probe)")
    s.add_argument("--json", action="store_true")
    rp = sub.add_parser("report", help="write a run's report.html again and print its path")
    rp.add_argument("run", help="a run folder, or its name under the runs folder")
    rp.add_argument("--open", action="store_true", help="open it in the browser")
    cmp = sub.add_parser("compare", help="compare runs, sides or a set: a Markdown table and contact sheets")
    cmp.add_argument("paths", nargs="+", type=Path, help="run folders, side folders or a set folder; the first "
                     "side is the baseline")
    cmp.add_argument("--out", type=Path, help="where compare.md and sheets/ go (default: the set folder, or a "
                     "new folder beside the first path)")
    cmp.add_argument("--json", action="store_true")
    c = sub.add_parser("clean", help="remove the environments --against and --manifoldx keep")
    c.add_argument("--json", action="store_true")
    m = sub.add_parser("micro", help="time micro cases")
    m.add_argument("modules", nargs="*", help="modules of @bench.case functions (default: the config's micro)")
    m.add_argument("--case", action="append")
    m.add_argument("--sizes", nargs="+", type=int)
    m.add_argument("--profile", action="store_true")
    m.add_argument("--smoke", action="store_true")
    m.add_argument("--against", type=Path, metavar="DIR", help="another manifoldx project to compare with")
    m.add_argument("--repeat", type=int, default=3, metavar="N",
                   help="rounds of each side with --against, interleaved (default 3)")
    m.add_argument("--json", action="store_true")
    return p


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        return {"list": _list, "run": _run, "check": _check, "show": _show, "report": _report, "micro": _micro, "clean": _clean,
                "compare": _compare}[args.cmd](args)
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


def _project(args) -> Config:
    """The project's config, its scenarios imported."""
    config = load_config(Path.cwd())
    if not config.has_table:
        raise ConfigError(f"{config.root / 'pyproject.toml'} has no [tool.manifoldx.bench]; add one with "
                          "scenarios = \"<module of your @bench.scenario functions>\"")
    import contextlib

    with contextlib.redirect_stdout(sys.stderr) if args.json else contextlib.nullcontext():
        _import_scenarios(config)  # a JSON scenario may call the app steps it registers; with --json, what the
        # app prints as it loads goes to stderr, so stdout holds only the summary
    return config


def _run(args) -> int:
    from manifoldx.bench.scenario import SCENARIOS

    if bool(args.scenario) == args.all:
        raise ConfigError("run takes a scenario's name, a JSON scenario file, or --all")
    config = _project(args)
    budgets = _budgets(config)
    if not args.all:
        code, out = _run_one(args, config, args.scenario, budgets)
        if args.json:
            print(json.dumps(out, indent=1))
        return code
    codes, outs = [], []
    for name in list(SCENARIOS):
        if not args.json:
            print(f"== {name} ==")
        code, out = _run_one(args, config, name, budgets)
        codes.append(code)
        outs.append(out)
        if out.get("error") == "interrupted" or out.get("interrupted"):  # Ctrl-C ends the whole run
            break
    if args.json:
        print(json.dumps(outs, indent=1))
    return max(codes, default=0)


def _check(args) -> int:
    """Run each scenario at smoke size with its checks and the budgets file's; 1 when any fails."""
    import contextlib

    from manifoldx.bench.folder import write_run
    from manifoldx.bench.runner import run_scenario
    from manifoldx.bench.scenario import SCENARIOS

    config = _project(args)
    budgets = _budgets(config)
    names = args.names or [n for n, sc in SCENARIOS.items() if sc.checks or budgets.get(n)]
    unknown = [n for n in names if n not in SCENARIOS]
    if unknown:
        raise ConfigError(f"no scenario {', '.join(unknown)}; known: {', '.join(sorted(SCENARIOS)) or 'none'}")
    if not names:
        raise ConfigError("no scenario has a check: add one with name.check(...) or in a budgets file")
    outs, failed = [], []
    for name in names:
        with contextlib.redirect_stdout(sys.stderr) if args.json else contextlib.nullcontext():
            run = run_scenario(SCENARIOS[name], size=args.size, smoke=not args.full, app_name=config.app_name,
                               folders=config.folders, budgets=budgets.get(name),
                               label=os.environ.get("MANIFOLDX_BENCH_LABEL"))
        folder = write_run(run, config.runs, config.history)
        results = run.summary["checks"]
        outs.append({"scenario": name, "folder": folder.name, "error": run.error, "checks": results})
        if run.error:
            failed.append(f"{name}: the run failed: {run.error}")
        failed += [f"{name}: {c['message']}" for c in results if not c["ok"]]
        if not args.json:
            if run.error:
                print(f"FAIL {name}: the run failed: {run.error}")
            for c in results:
                print(f"{'SKIP' if c['skipped'] else 'PASS' if c['ok'] else 'FAIL'} {name}: {c['message']}")
            if not results and not run.error:
                print(f"PASS {name}: ran, no checks")
        if run.error == "interrupted":  # Ctrl-C ends the whole check
            break
    if args.json:
        print(json.dumps(outs, indent=1))
    for line in failed:
        print(f"manifoldx bench: {line}", file=sys.stderr)
    return 1 if failed else 0


def _run_one(args, config: Config, scenario: str, budgets: dict) -> tuple[int, dict]:
    """Run one scenario, or the set its sweep or other copy makes; print its table unless --json and
    return the exit code and what --json prints."""
    from manifoldx.bench.folder import write_run
    from manifoldx.bench.runner import run_scenario
    from manifoldx.bench.scenario import SCENARIOS, load_json

    if scenario.endswith(".json"):
        try:
            sc = load_json(Path(scenario))
        except (OSError, ValueError, KeyError) as e:
            raise ConfigError(f"cannot read the scenario file {scenario}: {e}") from None
    else:
        if scenario not in SCENARIOS:
            raise ConfigError(f"no scenario {scenario!r}; known: {', '.join(sorted(SCENARIOS)) or 'none'}")
        sc = SCENARIOS[scenario]
    overrides = {}
    for kv in args.param:
        k, sep, v = kv.partition("=")
        if not sep:
            raise ConfigError(f"--param takes k=v, not {kv!r}")
        overrides[k] = v
    from manifoldx.bench.scenario import sweep

    try:
        combos = sweep(sc.params, overrides)
    except ValueError as e:
        raise ConfigError(str(e)) from None
    if args.repeat < 1:
        raise ConfigError(f"--repeat must be 1 or more, not {args.repeat}")
    if len(combos) > 1 or args.against or args.manifoldx:
        return _run_set(args, config, combos, scenario)
    if args.repeat > 1:
        raise ConfigError("--repeat needs more than one side: a swept --param, --against or --manifoldx")
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
                               app_name=config.app_name, progress=progress, folders=config.folders,
                               budgets=budgets.get(sc.name))
    except ValueError as e:  # bad params or loop
        raise ConfigError(str(e)) from None
    folder = write_run(run, config.runs, config.history)
    summary = {**run.summary, "folder": folder.name}
    if not args.json:
        _table(summary, folder)
    failed = [c["message"] for c in run.summary["checks"] if not c["ok"]]
    for message in failed:
        print(f"manifoldx bench: {message}", file=sys.stderr)
    if run.error:
        print(f"manifoldx bench: the run failed: {run.error}", file=sys.stderr)
    return (1 if run.error or failed else 0), summary


def _run_set(args, config: Config, combos: list[dict], scenario: str) -> tuple[int, dict]:
    """Run every side of a sweep in fresh processes, interleaved, into one set folder."""
    from manifoldx.bench import envs
    from manifoldx.bench.sides import Side, run_set

    if args.against and args.manifoldx:
        raise ConfigError("--against and --manifoldx go one at a time: with both, which copy runs which "
                          "manifoldx has two readings")
    copies = [("this copy", config.root, None, False)]
    if args.against or args.manifoldx:
        try:
            envs.uv()
        except envs.UvMissing as e:
            raise ConfigError(str(e)) from None
    if args.against:
        other = args.against.resolve()
        if not (other / "pyproject.toml").exists():
            raise ConfigError(f"--against needs another copy of the app, a project with a pyproject.toml; {other} "
                              "has none. manifoldx makes no copies: make one with your version control or unpack "
                              "a release")
        copies.insert(0, (other.name, other, None, True))
    if args.manifoldx:
        spec = args.manifoldx  # a path is resolved here: the child runs from the app's root
        spec = str(Path(spec).resolve()) if Path(spec).exists() else spec
        copies.append((args.manifoldx, config.root, spec, False))
    if scenario.endswith(".json"):  # each copy reads its own file, at the same place under its root
        path = Path(scenario).resolve()
        scenario = str(path.relative_to(config.root)) if path.is_relative_to(config.root) else str(path)
    sides = []
    for name, root, spec, other in copies:
        for combo in combos:
            params = " ".join(f"{k}={v}" for k, v in combo.items())
            label = params if len(copies) == 1 and params else " ".join(x for x in (name, params) if x)
            sides.append(Side(label, root, combo, spec, other))
    run_args = ["--size", f"{args.size[0]}x{args.size[1]}", "--loop", args.loop]
    run_args += (["--step", repr(args.step)] if args.step else []) + (["--smoke"] if args.smoke else [])
    label = args.label or os.environ.get("MANIFOLDX_BENCH_LABEL")
    run_args += ["--label", label] if label else []
    try:
        result = run_set(scenario, sides, args.repeat, run_args, config.runs)
    except envs.UvMissing as e:
        raise ConfigError(str(e)) from None
    out = {"set": result.folder.name, **{k: v for k, v in result.data.items() if k != "scenario"}}
    from manifoldx.bench import compare

    sides = compare.load_sides([result.folder])
    done = sum(1 for s in sides if s.runs) >= 2
    cmp = compare.write(sides, result.folder) if done else None
    if cmp:
        out.update(table=cmp["table"], rows=cmp["rows"], sheets=cmp["sheets"])
    if not args.json:
        for e in result.data["runs"]:
            if e["error"] or e["exit"]:
                side = result.data["sides"][e["side"]]["label"]
                print(f"{side}, round {e['round'] + 1}: {e['error'] or 'exit ' + str(e['exit'])}")
        if cmp:
            print(cmp["table"])
            for name, path in cmp["sheets"].items():
                print(f"sheet {name}: {path}")
        print(f"set folder: {result.folder}")
    return (1 if result.failed else 0), out


FRAME_KEYS = ("frames", "p50_ms", "p95_ms", "p99_ms", "max_ms", "over_1_5x", "over_3x", "draw_p50_ms", "cpu_p50_ms")
PHASE_PROBES = {"frames": None, "split": "split_ms", "systems": "systems_ms", "counters": "counters",
                "spans": "spans_ms", "streamer": "streamer", "workers": "workers", "app": "app",
                "memory": None, "origin": "origin_max_m"}  # memory: per phase, rss_max_mb and gc_ms
RUN_PROBES = ("host",)
TABLE_COUNTERS = ("draws", "bytes_uploaded", "material_reads")


def _probe(summary: dict, probe: str, phase: str | None):
    """One probe of a run: of one phase, of every phase by name, or of the run (host, and memory without a phase)."""
    if probe in RUN_PROBES or (probe == "memory" and phase is None):
        return summary[probe]
    phases = summary["phases"]
    if phase is not None and phase not in phases:
        raise ConfigError(f"no phase {phase!r} in this run; its phases: {', '.join(phases) or 'none'}")

    def pick(ph):
        if probe == "frames":
            return {k: ph[k] for k in FRAME_KEYS if k in ph}
        if probe == "memory":
            return {k: ph.get(k) for k in ("rss_max_mb", "gc_ms")}
        return ph.get(PHASE_PROBES[probe])

    return pick(phases[phase]) if phase is not None else {name: pick(ph) for name, ph in phases.items()}


def _ms(v) -> str:
    return f"{v:>9.1f}" if v is not None else f"{'-':>9}"


def _table(summary: dict, folder: Path) -> None:
    host = summary["host"]
    print(f"{summary['scenario']}  loop={summary['loop']}  size={summary['size'][0]}x{summary['size'][1]}  "
          f"{host['adapter']}{' (software)' if host['software'] else ''}  load {host['load_before']}–{host['load_max']}")
    print(f"{'phase':<14}{'frames':>7}{'p50 ms':>9}{'p95 ms':>9}{'max ms':>9}{'draw p50':>9}{'cpu p50':>9}"
          "  slowest systems")
    for name, ph in summary["phases"].items():
        top = ", ".join(f"{k.rsplit('.', 1)[-1]} {v}" for k, v in list(ph["systems_ms"].items())[:3])
        draw = ph.get("draw_p50_ms")  # a schema 1 run (no "schema" key) has none, and cpu meant draw
        cpu = ph["cpu_p50_ms"] if summary.get("schema", 1) >= 2 else None
        if draw is None and cpu is None:
            draw = ph["cpu_p50_ms"]
        print(f"{name:<14}{ph['frames']:>7}{ph['p50_ms']:>9.1f}{ph['p95_ms']:>9.1f}{ph['max_ms']:>9.1f}"
              f"{_ms(draw)}{_ms(cpu)}  {top}")
        c = ph.get("counters", {})
        more = [f"{k} {c[k]['mean']:g}" for k in TABLE_COUNTERS if k in c]
        if (ph.get("streamer") or {}).get("settled_s") is not None:
            more.append(f"settled {ph['streamer']['settled_s']:.2f} s")
        if more:
            print(f"{'':<14}per frame: {', '.join(more)}")
    for c in summary.get("checks", []):
        if c["skipped"] or not c["ok"]:
            print(f"{'skipped' if c['skipped'] else 'FAILED'} {c['message']}")
    print(f"run folder: {folder}")
    if (Path(folder) / "report.html").exists():
        print(f"report: {Path(folder) / 'report.html'}")


def _folder(name: str) -> Path:
    """A run folder, by its path or its name under the runs folder."""
    folder = Path(name)
    if not folder.exists():
        folder = load_config(Path.cwd()).runs / name
    if not (folder / "run.json").exists():
        raise ConfigError(f"no run at {folder}")
    return folder


def _report(args) -> int:
    from manifoldx.bench.report import write_report

    path = write_report(_folder(args.run))
    print(path)
    if args.open:
        import webbrowser

        webbrowser.open(path.resolve().as_uri())
    return 0


def _show(args) -> int:
    from manifoldx.bench.folder import read_run

    folder = _folder(args.run)
    summary = read_run(folder)
    if args.phase and not args.probe:
        raise ConfigError("--phase goes with --probe")
    if args.probe:
        print(json.dumps(_probe(summary, args.probe, args.phase), indent=1))
    elif args.json:
        print(json.dumps(summary, indent=1))
    else:
        _table(summary, folder)
    return 0


def _compare(args) -> int:
    import secrets
    from datetime import UTC, datetime

    from manifoldx.bench import compare

    try:
        sides = compare.load_sides(args.paths)
    except (OSError, ValueError) as e:
        raise ConfigError(str(e)) from None
    if sum(1 for s in sides if s.runs) < 2:
        raise ConfigError("compare needs two sides or more: two run folders, or a set folder with two sides")
    out = args.out
    if out is None:
        if len(args.paths) == 1:
            out = args.paths[0]
        else:
            stamp = datetime.now(UTC).strftime("%Y-%m-%dT%H-%M-%S")
            out = args.paths[0].resolve().parent / f"{stamp}-compare-{secrets.token_hex(2)}"
    cmp = compare.write(sides, out)
    if args.json:
        print(json.dumps(cmp, indent=1))
    else:
        print(cmp["table"])
        for name, path in cmp["sheets"].items():
            print(f"sheet {name}: {path}", file=sys.stderr)
        print(f"written to {cmp['folder']}", file=sys.stderr)
    return 0


def _clean(args) -> int:
    from manifoldx.bench import envs

    where = envs.cache_dir()
    mb = envs.clean()
    print(json.dumps({"removed": str(where), "mb": round(mb, 1)}) if args.json else f"removed {where}: {mb:.1f} MB")
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
            base, head = micro.against_results(modules, args.against, rest, cwd=cwd, repeat=max(1, args.repeat),
                                               profile=args.profile)
        except micro.AgainstError as e:
            raise ConfigError(str(e)) from None
        table = micro.compare(base, head, Path(args.against).name)
        print(json.dumps({"base": base, "head": head, "table": table}, indent=1) if args.json else table)
        return 1 if any(r["error"] for side in (base, head) for r in side["results"]) else 0
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
