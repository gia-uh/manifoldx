"""A set: one scenario run on several sides, each in fresh processes, interleaved, into one folder.

A side is one copy of the app with one combination of params, on its own manifoldx or another. Each
run of a side is a child `manifoldx bench run NAME --json` in a new process; round r runs the sides in
order, round r+1 in reverse (A B, B A, A B), so a load that comes and goes lands on every side.

    <runs>/<UTC time>-<scenario>-set-<4 hex>/
      set.json            the sides, the runs in the order they ran, each with its pid, exit and error;
                          "interrupted": true when Ctrl-C stopped it, with the runs that finished
      <k>-<label>/<run>/  each side's run folders, moved here from the runs folder of the copy that ran them
"""

from __future__ import annotations

import json
import re
import secrets
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path


@dataclass
class Side:
    """One copy of the app (`root`) with one combination of params, on its manifoldx or on `manifoldx`."""

    label: str
    root: Path
    params: dict = field(default_factory=dict)
    manifoldx: str | None = None
    other: bool = False


@dataclass
class SetResult:
    folder: Path
    failed: bool
    data: dict


def order(n_sides: int, repeat: int) -> list[tuple[int, int]]:
    """(side, round) in the order they run: each round in order, every other round reversed."""
    out = []
    for r in range(repeat):
        sides = range(n_sides) if r % 2 == 0 else reversed(range(n_sides))
        out += [(k, r) for k in sides]
    return out


def _slug(label: str) -> str:
    return re.sub(r"[^A-Za-z0-9._=-]+", "_", label).strip("_") or "side"


def _summary(out: str) -> dict | None:
    """The child's summary: the last JSON object it printed, after anything an app or an older manifoldx
    printed before it; None when there is none."""
    lines = out.splitlines()
    for i in range(len(lines) - 1, -1, -1):
        if lines[i].startswith("{"):
            try:
                found = json.loads("\n".join(lines[i:]))
            except ValueError:
                continue
            if isinstance(found, dict):
                return found
    return None


def run_set(scenario: str, sides: list[Side], repeat: int, run_args: list[str], runs_dir: Path) -> SetResult:
    """Run `scenario` on every side `repeat` times, interleaved; write the set folder and return it."""
    from manifoldx.bench import envs
    from manifoldx.bench.cli import load_config

    stamp = datetime.now(UTC).strftime("%Y-%m-%dT%H-%M-%S")
    folder = Path(runs_dir) / f"{stamp}-{_slug(scenario)}-set-{secrets.token_hex(2)}"
    folder.mkdir(parents=True)
    data = {"scenario": scenario, "repeat": repeat,
            "sides": [{**asdict(s), "root": str(s.root)} for s in sides], "runs": []}
    plan = order(len(sides), repeat)
    uses_uv = any(s.other or s.manifoldx for s in sides)
    before = envs.size_mb() if uses_uv else 0.0
    child = None
    try:
        for n, (k, r) in enumerate(plan, 1):
            side = sides[k]
            print(f"== side {k + 1}/{len(sides)} {side.label}, round {r + 1}/{repeat} ({n}/{len(plan)}) ==",
                  file=sys.stderr, flush=True)
            argv, env, cwd = envs.command(side)
            params = [x for key, v in side.params.items() for x in ("--param", f"{key}={v}")]
            child = subprocess.Popen([*argv, "bench", "run", scenario, "--json", *run_args, *params],
                                     cwd=cwd, env=env, stdout=subprocess.PIPE, text=True)
            out, _ = child.communicate()
            entry = {"side": k, "round": r, "folder": None, "pid": child.pid, "exit": child.returncode, "error": None}
            summary = _summary(out)
            if summary is None or "folder" not in summary:  # it stopped before printing its summary
                entry["error"] = f"exit {child.returncode}: no summary"
            else:
                entry["error"] = summary.get("error")
                made = load_config(Path(side.root)).runs / summary["folder"]
                dest = folder / f"{k}-{_slug(side.label)}"
                try:
                    dest.mkdir(exist_ok=True)
                    shutil.move(str(made), dest / made.name)
                    entry["folder"] = f"{dest.name}/{made.name}"
                except OSError as e:
                    entry["error"] = f"{entry['error'] or 'ran'}; its folder {made} could not be moved: {e}"
            data["runs"].append(entry)
            (folder / "set.json").write_text(json.dumps(data, indent=1))
    except KeyboardInterrupt:  # Ctrl-C: stop the run in flight, keep and compare the runs that finished
        if child is not None and child.poll() is None:
            child.kill()
            child.wait()
        data["interrupted"] = True
        print(f"bench: interrupted after {len(data['runs'])} of {len(plan)} runs", file=sys.stderr, flush=True)
        (folder / "set.json").write_text(json.dumps(data, indent=1))
    if uses_uv:
        dropped = envs.trim(keep={envs.key(s.root, s.manifoldx) for s in sides if s.other or s.manifoldx})
        if dropped:
            print(f"bench: the environment cache passed its cap; dropped {len(dropped)} unused",
                  file=sys.stderr)
        data["cache_mb"] = round(envs.size_mb(), 1)
        data["cache_added_mb"] = round(max(0.0, data["cache_mb"] - before), 1)
        print(f"bench: environments under {envs.cache_dir()}: +{data['cache_added_mb']} MB, "
              f"{data['cache_mb']} MB in all (manifoldx bench clean empties it)", file=sys.stderr)
        (folder / "set.json").write_text(json.dumps(data, indent=1))
    failed = data.get("interrupted", False) or any(e["exit"] != 0 or e["error"] for e in data["runs"])
    return SetResult(folder, failed, data)
