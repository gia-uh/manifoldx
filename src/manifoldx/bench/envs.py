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
