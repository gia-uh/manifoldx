"""How a side of a set runs: the child's command, its environment, and where uv keeps it.

This copy runs on the interpreter already running; another copy (--against) or another manifoldx
(--manifoldx) runs through `uv run`. manifoldx never makes the other copy: the caller does, with its
own version control or by unpacking a release.
"""

from __future__ import annotations

import os
import sys

CHILD = "import sys; from manifoldx.cli import main; sys.exit(main())"


def command(side, cache=None):
    """(argv prefix, env, cwd) for one child `manifoldx bench run` of this side."""
    return [sys.executable, "-c", CHILD], dict(os.environ), side.root
