"""Run benchmark suites; optionally against another git ref's src/ for a before/after table.

    uv run python -m benchmarks --list
    uv run python -m benchmarks frame ecs --sizes 1000 10000
    uv run python -m benchmarks frame --case sprites --against main

With --against REF, the same suites (this checkout's benchmarks/) run twice in
fresh processes: once on `git archive REF src/` and once on this checkout's src/,
and a Markdown table of both goes to stdout, ready for a pull request.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import platform
import subprocess
import sys
import tempfile
import traceback
from dataclasses import asdict
from pathlib import Path

from benchmarks import harness

SUITES = ["frame", "ecs", "compute", "skinning", "modeling", "loaders", "web"]
ROOT = Path(__file__).resolve().parent.parent


def load(suites):
    for s in suites:
        importlib.import_module(f"benchmarks.suites.{s}")
    return [c for s in suites for c in harness.CASES.get(s, [])]


def run(cases, sizes=None, profile=False, smoke=False):
    results = []
    for c in cases:
        for n in [c.smoke] if smoke else (sizes or c.sizes):
            try:
                out = c.fn(n)
                ms, extra = (out.pop("ms"), out) if isinstance(out, dict) else (out, {})
                r = harness.Result(c.suite, c.name, n, float(ms), extra)
            except Exception as e:  # a case the code under test cannot run (an API it lacks)
                r = harness.Result(c.suite, c.name, n, None, error=f"{type(e).__name__}: {e}")
                if not smoke:
                    traceback.print_exc(limit=3, file=sys.stderr)
            results.append(r)
            print(_row(r), flush=True)
        if profile:
            n = (sizes or c.sizes)[-1]
            print(f"--- profile {c.suite}.{c.name} n={n} (setup included)")
            print(harness.profile(lambda: c.fn(n)))
    return results


def _row(r):
    if r.error:
        return f"{r.suite:<9} {r.case:<22} n={r.n:<9} ERROR {r.error}"
    extra = "  ".join(f"{k}={v}" for k, v in r.extra.items())
    return f"{r.suite:<9} {r.case:<22} n={r.n:<9} {r.ms:12.3f} ms  {extra}"


def _fmt(ms):
    return "error" if ms is None else (f"{ms:.3f}" if ms < 1 else f"{ms:.1f}")


def compare(base, head, ref):
    """Markdown table of base vs head results, matched on (suite, case, n)."""
    by_key = {(r["suite"], r["case"], r["n"]): r for r in base["results"]}
    lines = [
        f"| case | n | {ref} ms | this branch ms | speed-up | |",
        "|---|---|---|---|---|---|",
    ]
    for r in head["results"]:
        b = by_key.get((r["suite"], r["case"], r["n"]))
        b_ms = b["ms"] if b else None
        if b_ms and r["ms"]:
            ratio = f"{b_ms / r['ms']:.1f}×"
        else:
            ratio = "–"
        extra = ", ".join(f"{k} {v}" for k, v in r["extra"].items())
        if b and b["extra"] != r["extra"] and b["extra"]:
            extra += " (was " + ", ".join(f"{k} {v}" for k, v in b["extra"].items()) + ")"
        lines.append(
            f"| {r['suite']}.{r['case']} | {r['n']} | {_fmt(b_ms) if b else 'n/a'} | {_fmt(r['ms'])} | {ratio} | {extra} |"
        )
    return "\n".join(lines)


def _child(args, src: Path, out: Path):
    """Run this CLI in a fresh process with `src` first on the import path."""
    cmd = [sys.executable, "-m", "benchmarks", *args.suites, "--json", str(out)]
    for c in args.case or []:
        cmd += ["--case", c]
    if args.sizes:
        cmd += ["--sizes", *map(str, args.sizes)]
    env = {**os.environ, "PYTHONPATH": os.pathsep.join([str(src), str(ROOT)])}
    subprocess.run(cmd, cwd=ROOT, env=env, check=True)
    data = json.loads(out.read_text())
    imported = Path(data["manifoldx"]).resolve()
    if not imported.is_relative_to(src.resolve()):  # the run must have measured the code asked for
        sys.exit(f"benchmarks: expected manifoldx from {src}, the run imported {imported}")
    return data


def main(argv=None):
    p = argparse.ArgumentParser(prog="python -m benchmarks", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("suites", nargs="*", help=f"suites to run (default: all of {', '.join(SUITES)})")
    p.add_argument("--case", action="append", help="only this case (repeatable)")
    p.add_argument("--sizes", nargs="+", type=int, help="sizes for every case instead of its own")
    p.add_argument("--json", type=Path, help="write the results here")
    p.add_argument("--profile", action="store_true", help="cProfile each case once at its largest size")
    p.add_argument("--against", metavar="REF", help="also run on REF's src/ and print a comparison")
    p.add_argument("--list", action="store_true", help="list suites and cases")
    args = p.parse_args(argv)
    args.suites = args.suites or SUITES
    unknown = set(args.suites) - set(SUITES)
    if unknown:
        p.error(f"unknown suite(s): {', '.join(sorted(unknown))}")

    if args.list:
        for c in load(args.suites):
            print(f"{c.suite:<9} {c.name:<22} sizes={c.sizes}  {c.doc}")
        return

    if args.against:
        with tempfile.TemporaryDirectory(prefix="mx-bench-") as tmp:
            tmp = Path(tmp)
            archive = subprocess.run(["git", "archive", args.against, "src"], cwd=ROOT,
                                     capture_output=True, check=True).stdout
            subprocess.run(["tar", "-x", "-C", str(tmp)], input=archive, check=True)
            print(f"== {args.against} ==", flush=True)
            base = _child(args, tmp / "src", tmp / "base.json")
            print("== this checkout ==", flush=True)
            head = _child(args, ROOT / "src", tmp / "head.json")
        table = compare(base, head, args.against)
        print("\n" + table)
        if args.json:
            args.json.write_text(json.dumps({"base": base, "head": head, "table": table}, indent=1))
        return

    import manifoldx

    cases = [c for c in load(args.suites) if not args.case or c.name in args.case]
    results = run(cases, args.sizes, args.profile)
    if args.json:
        args.json.write_text(json.dumps({
            "manifoldx": manifoldx.__file__,
            "machine": f"{platform.machine()} · {os.cpu_count()} cpus · Python {platform.python_version()}",
            "results": [asdict(r) for r in results],
        }, indent=1))


if __name__ == "__main__":
    main()
