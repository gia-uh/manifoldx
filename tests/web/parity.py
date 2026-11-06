"""Run a snippet in Python and, transpiled, in Node, and return both results as JSON values."""

import importlib.util
import json
import shutil
import subprocess
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


def run_python(src: str, folder: Path) -> object:
    path = folder / "mod.py"
    path.write_text(src)
    spec = importlib.util.spec_from_file_location("mod", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return json.loads(json.dumps(module.main(), default=_py_default))


def run_node(src: str, folder: Path) -> object:
    for name, mod in transpile_app({"mod": src}).items():
        (folder / f"{name}.js").write_text(mod.code)
    shutil.copy(PYRT, folder / "pyrt.js")
    url = (folder / "mod.js").as_uri()
    script = f'import {{ main }} from "{url}";\nconsole.log(JSON.stringify(main(), {_REPLACER}));'
    done = subprocess.run(
        ["node", "--input-type=module", "-e", script], capture_output=True, text=True, timeout=60
    )
    if done.returncode != 0:
        raise RuntimeError(f"node failed:\n{done.stderr}")
    return json.loads(done.stdout)


def run_both(src: str) -> tuple[object, object]:
    """main() of src run in Python and in Node: (python_value, js_value)."""
    with tempfile.TemporaryDirectory() as tmp:
        folder = Path(tmp)
        return run_python(src, folder), run_node(src, folder)
