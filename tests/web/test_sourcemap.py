"""Source maps: a JS exception in a transpiled module names the Python file and line."""

import json
import re
import shutil
import subprocess
from pathlib import Path

from manifoldx.web import transpile_app

PYRT = Path(__file__).parents[2] / "src" / "manifoldx" / "web" / "js" / "pyrt.js"

SRC = """
class Other:
    def method(self):
        return 1


class Thing:
    def __init__(self):
        self.missing: Other = None
        self.n = 0

    def poke(self):
        self.n += 1
        return self.missing.method()


def main():
    t = Thing()
    return t.poke()
"""

# The line of `self.missing.method()` in SRC.
CALL_LINE = SRC.splitlines().index("        return self.missing.method()") + 1


def _write(folder: Path, sources: dict[str, str]) -> None:
    for name, mod in transpile_app(sources).items():
        (folder / f"{name}.js").write_text(mod.code)
        (folder / f"{name}.js.map").write_text(mod.source_map)
    shutil.copy(PYRT, folder / "pyrt.js")


def test_a_stack_trace_names_the_python_line(tmp_path):
    _write(tmp_path, {"mod": SRC})
    url = (tmp_path / "mod.js").as_uri()
    done = subprocess.run(
        [
            "node",
            "--enable-source-maps",
            "--input-type=module",
            "-e",
            f'import {{ main }} from "{url}";\nmain();',
        ],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert done.returncode != 0
    assert re.search(rf"mod\.py:{CALL_LINE}:\d+", done.stderr), done.stderr


def test_each_module_ends_with_its_map_url():
    mods = transpile_app({"mod": SRC, "pkg.sub": "def f(x: int) -> int:\n    return x + 1\n"})
    for name, mod in mods.items():
        assert mod.code.rstrip("\n").splitlines()[-1] == f"//# sourceMappingURL={name}.js.map"
        m = json.loads(mod.source_map)
        assert m["version"] == 3
        assert m["file"] == f"{name}.js"
    assert json.loads(mods["pkg.sub"].source_map)["sources"] == ["pkg/sub.py"]


def test_each_mapped_line_points_at_its_statement():
    """Decode the map: the generated line holding `x + 1` maps to the Python line 2."""
    mod = transpile_app({"mod": "def f(x: int) -> int:\n    return x + 1\n"})["mod"]
    lines = _decode(json.loads(mod.source_map)["mappings"])
    js = mod.code.splitlines()
    at = next(i for i, t in enumerate(js) if "x + 1" in t)
    assert lines[at] == 2
    assert lines[next(i for i, t in enumerate(js) if t.startswith("export function f"))] == 1
    assert lines[0] is None  # the header comment comes from no Python line


_B64 = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/"


def _decode(mappings: str) -> list[int | None]:
    """The 1-based source line of each generated line's first segment, None for none."""
    out, src_line = [], 0
    for group in mappings.split(";"):
        first = None
        for seg in filter(None, group.split(",")):
            values, shift, acc = [], 0, 0
            for ch in seg:
                d = _B64.index(ch)
                acc += (d & 31) << shift
                shift += 5
                if not d & 32:
                    values.append(-(acc >> 1) if acc & 1 else acc >> 1)
                    acc = shift = 0
            if len(values) >= 4:
                src_line += values[2]
                if first is None:
                    first = src_line + 1
        out.append(first)
    return out
