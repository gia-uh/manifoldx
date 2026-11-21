"""Run benchmark suites; optionally against another git ref's src/ for a before/after table.

    uv run python -m benchmarks --list
    uv run python -m benchmarks frame ecs --sizes 1000 10000
    uv run python -m benchmarks frame --case sprites --against ../manifoldx-main

With --against DIR, the same suites (this checkout's benchmarks/) run twice in
fresh processes: once on DIR's src/ (another manifoldx project, such as a copy of
main you made) and once on this checkout's src/, and a Markdown table of both goes
to stdout, ready for a pull request.
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

from manifoldx.bench import micro

SUITES = ["frame", "ecs", "compute", "skinning", "modeling", "loaders", "web", "trace", "startup", "streaming", "audio"]
ROOT = Path(__file__).resolve().parent.parent


def modules(suites):
    """The module of each suite name."""
    return [f"benchmarks.suites.{s}" for s in suites]


def main(argv=None):
    p = argparse.ArgumentParser(prog="python -m benchmarks", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("suites", nargs="*", help=f"suites to run (default: all of {', '.join(SUITES)})")
    p.add_argument("--case", action="append", help="only this case (repeatable)")
    p.add_argument("--sizes", nargs="+", type=int, help="sizes for every case instead of its own")
    p.add_argument("--json", type=Path, help="write the results here")
    p.add_argument("--profile", action="store_true", help="cProfile each case once at its largest size")
    p.add_argument("--against", metavar="DIR", type=Path,
                   help="also run on another manifoldx project's src/ and print a comparison")
    p.add_argument("--list", action="store_true", help="list suites and cases")
    args = p.parse_args(argv)
    args.suites = args.suites or SUITES
    unknown = set(args.suites) - set(SUITES)
    if unknown:
        p.error(f"unknown suite(s): {', '.join(sorted(unknown))}")

    if args.list:
        for c in micro.load(modules(args.suites)):
            print(f"{c.suite:<9} {c.name:<22} sizes={c.sizes}  {c.doc}")
        return

    if args.against:
        rest = [x for c in args.case or [] for x in ("--case", c)] + (["--sizes", *map(str, args.sizes)] if args.sizes else [])
        table = micro.against(modules(args.suites), args.against, rest, cwd=ROOT)
        print("\n" + table)
        if args.json:
            args.json.write_text(json.dumps({"table": table}, indent=1))
        return

    import manifoldx

    cases = [c for c in micro.load(modules(args.suites)) if not args.case or c.name in args.case]
    results = micro.run(cases, args.sizes, args.profile)
    if args.json:
        args.json.write_text(json.dumps({
            "manifoldx": manifoldx.__file__,
            "machine": f"{platform.machine()} · {os.cpu_count()} cpus · Python {platform.python_version()}",
            "results": [asdict(r) for r in results],
        }, indent=1))


if __name__ == "__main__":
    main()
