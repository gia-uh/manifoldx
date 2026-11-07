"""random.Random in frame code: the generator made at setup crosses in the snapshot as its
MT19937 words, and pyrt's Random draws the same numbers as CPython's, bit for bit."""

import ast
import importlib
import json
import random
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

import manifoldx as mx
from manifoldx.web import TranspileError, snapshot, transpile_app
from manifoldx.web.reach import Reach, find_roots
from manifoldx.web.types import LiveTypes, TypeEnv

JS = Path(__file__).parents[2] / "src" / "manifoldx" / "web" / "js"

METHODS = {
    "random": "r.random()",
    "uniform": "r.uniform(-3.5, 12.25)",
    "randint": "r.randint(-5, 17)",
    "randrange": "r.randrange(3, 1000, 7)",
    "randrange_wide": "r.randrange(0, 2 ** 40)",  # two words of getrandbits
    "randrange_down": "r.randrange(50, -50, -3)",
    "choice": "r.choice(NAMES)",
}

APP = """
import random

NAMES = ["ash", "birch", "cedar", "elm", "fir", "oak", "pine"]
R = None


def main():
    r = R
    return [{expr} for _ in range(10_000)]


def step(query, dt):
    main()


def setup(engine):
    global R
    R = random.Random(20261107)
    for _ in range({skip}):
        R.random()
    engine.system(step)
"""


def _load(folder: Path, src: str):
    """The module `app` of src, imported from folder, with setup run on a fresh engine."""
    (folder / "app.py").write_text(src)
    sys.modules.pop("app", None)
    sys.path.insert(0, str(folder))
    try:
        importlib.invalidate_caches()
        mod = importlib.import_module("app")
    finally:
        sys.path.remove(str(folder))
    engine = mx.Engine("t", width=64, height=64)
    mod.setup(engine)
    return mod, engine


def _export(folder: Path, src: str):
    """Setup run in Python, its snapshot taken and what the roots reach transpiled into
    folder/js: (module, snapshot, roots)."""
    mod, engine = _load(folder, src)
    roots = find_roots(engine)
    sources = {"app": src}
    live = LiveTypes(set(sources))
    reach = Reach(roots, TypeEnv({"app": ast.parse(src)}, live))
    snap = json.loads(json.dumps(snapshot(roots, reach, engine), allow_nan=False))
    out = folder / "js"
    out.mkdir(exist_ok=True)
    for name, m in transpile_app(sources, roots=roots, live=LiveTypes(set(sources))).items():
        (out / f"{name}.js").write_text(m.code)
    for f in ("pyrt.js", "manifoldx.js"):
        shutil.copy(JS / f, out / f)
    (out / "state.json").write_text(json.dumps(snap))
    return mod, snap, out


def _node_main(out: Path):
    """main() of the exported app, run in Node on the loaded state."""
    script = """import * as app from "./app.js";
import { loadState } from "./manifoldx.js";
import { readFileSync } from "node:fs";
const state = JSON.parse(readFileSync(new URL("./state.json", import.meta.url)));
loadState(state, { app }, () => null);
console.log(JSON.stringify(app.main()));
"""
    (out / "main.mjs").write_text(script)
    done = subprocess.run(["node", str(out / "main.mjs")], capture_output=True, text=True)
    if done.returncode != 0:
        raise RuntimeError(f"node failed:\n{done.stderr}")
    return json.loads(done.stdout)


def _both(tmp_path: Path, src: str):
    mod, _, out = _export(tmp_path, src)
    sys.modules.pop("app", None)
    return mod.main(), _node_main(out)


@pytest.mark.parametrize("method", list(METHODS))
@pytest.mark.parametrize("start", ["seeded", "mid_stream"])
def test_ten_thousand_draws_match(tmp_path, method, start):
    src = APP.format(expr=METHODS[method], skip=1234 if start == "mid_stream" else 0)
    py, js = _both(tmp_path, src)
    assert len(py) == 10_000
    assert js == py  # rel=0: equal floats, every one
    assert [type(x) for x in js] == [type(x) for x in py]


MODULE_FUNCTIONS = """
import random


def main():
    out = []
    for _ in range(100):
        out += [random.random(), random.randint(1, 6), random.choice([1, 2, 3])]
    return out


def step(query, dt):
    main()


def setup(engine):
    random.seed(5)
    random.random()
    random.random()
    engine.system(step)
"""


def test_the_module_functions_use_the_module_state(tmp_path):
    py, js = _both(tmp_path, MODULE_FUNCTIONS)
    assert len(py) == 300
    assert js == py


def test_the_state_crosses_as_its_words(tmp_path):
    mod, snap, _ = _export(tmp_path, APP.format(expr="r.random()", skip=1234))
    version, words, gauss = mod.R.getstate()
    form = {"key": list(words[:-1]), "pos": words[-1], "gauss": gauss}
    assert version == 3 and len(form["key"]) == 624
    assert {"__random": form} in snap["table"]
    assert "random" not in snap  # no module function is reached
    sys.modules.pop("app", None)


def test_the_module_state_crosses_when_reached_code_calls_it(tmp_path):
    _, snap, _ = _export(tmp_path, MODULE_FUNCTIONS)
    sys.modules.pop("app", None)
    version, words, gauss = random._inst.getstate()
    assert snap["random"] == {"key": list(words[:-1]), "pos": words[-1], "gauss": gauss}


def test_an_empty_choice_raises_as_python(tmp_path):
    src = APP.format(expr="r.choice(NAMES[:0])", skip=0)
    _, _, out = _export(tmp_path, src)
    sys.modules.pop("app", None)
    with pytest.raises(RuntimeError, match="IndexError: Cannot choose from an empty sequence"):
        _node_main(out)


FRAME_CODE = """
import random

R = None


def step(query, dt):
    {line}


def setup(engine):
    global R
    R = random.Random(3)
    engine.system(step)
"""


@pytest.mark.parametrize(
    "line, what",
    [
        ("random.seed(1)", "`random.seed(1)`, a reseed in frame code (seed it at setup)"),
        ("random.shuffle([1, 2])", "random.shuffle()"),
        ("R.shuffle([1, 2])", "`R.shuffle` on a random.Random"),
        ("R.seed = 2", "`R.seed` on a random.Random"),
    ],
)
def test_frame_code_outside_the_generator_fails_the_export(tmp_path, line, what):
    src = FRAME_CODE.format(line=line)
    _, engine = _load(tmp_path, src)
    sys.modules.pop("app", None)
    with pytest.raises(TranspileError) as e:
        transpile_app({"app": src}, roots=find_roots(engine), live=LiveTypes({"app"}))
    assert (e.value.path, e.value.line) == ("app.py", 8)
    assert what in str(e.value)


def test_a_random_made_in_frame_code_fails_the_export(tmp_path):
    src = FRAME_CODE.format(line="random.Random(1)")
    _, engine = _load(tmp_path, src)
    sys.modules.pop("app", None)
    with pytest.raises(TranspileError) as e:
        transpile_app({"app": src}, roots=find_roots(engine), live=LiveTypes({"app"}))
    assert str(e.value).startswith(
        "app.py:8: `random.Random(1)`, a generator made in frame code (make it at setup)"
    )


def test_the_state_not_loaded_is_a_mismatch(tmp_path):
    src = APP.format(expr="r.random()", skip=0)
    mod, _, out = _export(tmp_path, src)
    sys.modules.pop("app", None)
    runtime = (out / "manifoldx.js").read_text()
    assert runtime.count("Random.fromState(e.__random)") == 1
    zero = random.Random(0).getstate()[1]
    seeded = json.dumps({"key": list(zero[:-1]), "pos": zero[-1], "gauss": None})
    (out / "manifoldx.js").write_text(
        runtime.replace("Random.fromState(e.__random)", f"Random.fromState({seeded})")
    )
    assert _node_main(out) != mod.main()
