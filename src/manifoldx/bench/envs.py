"""How a side of a set runs: the child's command, its environment, and where uv keeps it.

This copy runs on the interpreter already running; another copy (--against) or another manifoldx
(--manifoldx) runs through `uv run`. manifoldx never makes the other copy: the caller does, with its
own version control or by unpacking a release.

The project environment of each (project, spec) pair lives under one cache folder, named by a hash of
the two, so `uv run --project DIR` writes no `.venv` into the other copy and a second run reuses it. The
layer `--with SPEC` adds lives in uv's own cache, which uv shares across projects (`uv cache prune`).
"""

from __future__ import annotations

import hashlib
import os
import shutil
import sys
from pathlib import Path

CHILD = "import sys; from manifoldx.cli import main; sys.exit(main())"


class UvMissing(Exception):
    """--against or --manifoldx was asked for and uv is not on the PATH."""


def uv() -> str:
    """The uv on the PATH, or UvMissing saying what still works without it."""
    found = shutil.which("uv")
    if found is None:
        raise UvMissing("--against and --manifoldx run the other side with uv, which is not on the PATH. Install "
                        "uv, or make both runs yourself and compare them with manifoldx bench compare RUN_A RUN_B. "
                        "Plain runs, sweeps on this copy and micro cases need no uv.")
    return found


def cache_dir() -> Path:
    """$MANIFOLDX_BENCH_CACHE, else manifoldx/bench-envs under $XDG_CACHE_HOME or ~/.cache."""
    if os.environ.get("MANIFOLDX_BENCH_CACHE"):
        return Path(os.environ["MANIFOLDX_BENCH_CACHE"])
    base = os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache"
    return Path(base) / "manifoldx" / "bench-envs"


def key(root, spec: str | None) -> str:
    """The cache folder's name for a project and a manifoldx spec."""
    return hashlib.sha256(f"{Path(root).resolve()}\0{spec or ''}".encode()).hexdigest()[:16]


def command(side) -> tuple[list[str], dict, Path]:
    """(argv prefix, env, cwd) for one child `manifoldx bench run` of this side."""
    if not (side.other or side.manifoldx):
        return [sys.executable, "-c", CHILD], dict(os.environ), side.root
    home = cache_dir() / key(side.root, side.manifoldx)
    home.mkdir(parents=True, exist_ok=True)
    (home / "used").touch()  # the cap drops the least recently used first
    env = {k: v for k, v in os.environ.items() if k != "VIRTUAL_ENV"}  # this copy's, not the child's
    env["UV_PROJECT_ENVIRONMENT"] = str(home / "venv")
    argv = [uv(), "run", "--project", str(side.root)]
    if side.manifoldx:
        # uv keeps a directory's build until its pyproject.toml changes: editable sees every edit.
        argv += ["--with-editable" if Path(side.manifoldx).is_dir() else "--with", side.manifoldx]
    return [*argv, "manifoldx"], env, side.root


def cap_mb() -> float:
    """The cache's cap: $MANIFOLDX_BENCH_CACHE_GB, 2 GB by default."""
    return 1024.0 * float(os.environ.get("MANIFOLDX_BENCH_CACHE_GB", 2))


def size_mb(path=None) -> float:
    """What a folder holds, the cache by default, in MB."""
    from manifoldx.bench.probes import folder_mb

    path = cache_dir() if path is None else Path(path)
    return folder_mb(path) if path.exists() else 0.0


def trim(cap: float | None = None, keep: set | frozenset = frozenset()) -> list[str]:
    """Drop the least recently used environments until the cache is under its cap; never one in `keep`."""
    root = cache_dir()
    if not root.exists():
        return []
    cap = cap_mb() if cap is None else cap
    homes = sorted((d for d in root.iterdir() if d.is_dir()),
                   key=lambda d: (d / "used").stat().st_mtime if (d / "used").exists() else 0.0)
    sizes = {d.name: size_mb(d) for d in homes}
    total, dropped = sum(sizes.values()), []
    for d in homes:
        if total <= cap:
            break
        if d.name in keep:
            continue
        shutil.rmtree(d, ignore_errors=True)
        total -= sizes[d.name]
        dropped.append(d.name)
    return dropped


def clean() -> float:
    """Remove the whole cache; return the MB it held."""
    held = size_mb()
    shutil.rmtree(cache_dir(), ignore_errors=True)
    return held
