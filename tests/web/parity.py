"""Run a snippet in Python and, transpiled, in Node, and return both results as JSON values."""

import importlib
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

from manifoldx.web import transpile_app

PYRT = Path(__file__).parents[2] / "src" / "manifoldx" / "web" / "js" / "pyrt.js"

# Typed arrays and Sets become arrays; a Set's items are sorted by their JSON text on both
# sides, because Python and JS iterate sets in different orders.
_REPLACER = """(k, v) => {
  if (ArrayBuffer.isView(v)) return Array.from(v);
  if (v instanceof Set) return [...v].sort((a, b) => {
    const x = JSON.stringify(a), y = JSON.stringify(b);
    return x < y ? -1 : x > y ? 1 : 0;
  });
  return v;
}"""


def _py_default(v):
    if isinstance(v, (set, frozenset)):
        return sorted(v, key=lambda x: json.dumps(x, default=_py_default))
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, np.generic):
        return v.item()
    raise TypeError(f"{type(v).__name__} has no JSON value")


def run_python(sources: dict[str, str], folder: Path, main: str) -> object:
    for name, src in sources.items():
        (folder / f"{name}.py").write_text(src)
    sys.path.insert(0, str(folder))
    try:
        for name in sources:
            sys.modules.pop(name, None)
        importlib.invalidate_caches()
        module = importlib.import_module(main)
        return json.loads(json.dumps(module.main(), default=_py_default))
    finally:
        sys.path.remove(str(folder))
        for name in sources:
            sys.modules.pop(name, None)


def run_node(sources: dict[str, str], folder: Path, main: str) -> object:
    for name, mod in transpile_app(sources).items():
        (folder / f"{name}.js").write_text(mod.code)
    shutil.copy(PYRT, folder / "pyrt.js")
    url = (folder / f"{main}.js").as_uri()
    script = f'import {{ main }} from "{url}";\nconsole.log(JSON.stringify(main(), {_REPLACER}));'
    done = subprocess.run(
        ["node", "--input-type=module", "-e", script], capture_output=True, text=True, timeout=60
    )
    if done.returncode != 0:
        raise RuntimeError(f"node failed:\n{done.stderr}")
    return json.loads(done.stdout)


def run_modules(sources: dict[str, str], main: str = "mod") -> tuple[object, object]:
    """main() of the module `main` among sources ({module name: source}), run in Python and
    in Node: (python_value, js_value)."""
    with tempfile.TemporaryDirectory() as tmp:
        folder = Path(tmp)
        return run_python(sources, folder, main), run_node(sources, folder, main)


def run_both(src: str) -> tuple[object, object]:
    """main() of src run in Python and in Node: (python_value, js_value)."""
    return run_modules({"mod": src})
