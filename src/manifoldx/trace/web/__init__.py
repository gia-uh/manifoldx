"""The tracer for the browser: trace.js and its modules, and the WGSL kernels they fetch.

    from manifoldx.trace.web import copy_to
    copy_to("static/trace")  # then, in the page: import { loadGlb, createTracer } from "/static/trace/web/trace.js"
"""

from __future__ import annotations

import shutil
from importlib import resources
from pathlib import Path


def copy_to(dest) -> Path:
    """Write web/*.js and wgsl/*.wgsl under dest, as trace.js expects to find them; returns trace.js's path."""
    dest = Path(dest)
    root = resources.files("manifoldx.trace")
    for folder, suffix in (("web", ".js"), ("wgsl", ".wgsl")):
        out = dest / folder
        out.mkdir(parents=True, exist_ok=True)
        for item in root.joinpath(folder).iterdir():
            if item.name.endswith(suffix):
                with resources.as_file(item) as path:
                    shutil.copyfile(path, out / item.name)
    return dest / "web" / "trace.js"
