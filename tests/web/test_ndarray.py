"""numpy arrays of any rank in frame code: each snippet's main() gives the same JSON value in
Python and in Node, bit for bit, under numpy 2's dtype rules."""

import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pytest

from manifoldx.web import transpile_app
from manifoldx.web.types import TranspileError

from .parity import PYRT, run_both, run_node, run_python

CASES = {
    # a Python float keeps float32 float32 (NEP 50); each step rounds
    "f32_scalar": "import numpy as np\ndef main():\n    a = np.full((4, 3), 0.1, np.float32)\n    return (a * 3.0 + 0.7 - a / 7.0).tolist()",
    # (c,) against (k, c), a float32 row against float64 rows
    "trailing_row": "import numpy as np\ndef main():\n    a = np.ones((3, 2), np.float32) * 0.3\n    r = np.array([0.1, 0.2])\n    return [(a + r).tolist(), (a * np.array([0.1, 0.2], np.float32)).tolist()]",
    "equal_shapes": "import numpy as np\ndef main():\n    a = np.arange(0.0, 6.0, 1.0, np.float32)\n    return (a * a - a / 3.0).tolist()",
    "int32": "import numpy as np\ndef main():\n    i = np.arange(0, 6, 1, np.int32)\n    return [(i * 2 + 1).tolist(), (i / 4).tolist(), (i + 0.5).tolist()]",
    "neg_abs": "import numpy as np\ndef main():\n    a = np.full((2, 2), -0.3, np.float32)\n    return [(-a).tolist(), abs(a).tolist()]",
    "compare_masks": "import numpy as np\ndef main():\n    a = np.arange(0.0, 8.0, 1.0)\n    m = (a > 2.0) & (a <= 6.0) | ~(a != 7.0)\n    return m.tolist()",
    # numpy's fast paths: square, sqrt, reciprocal
    "powers": "import numpy as np\ndef main():\n    a = np.full((3,), 1.7, np.float32)\n    return [(a ** 2).tolist(), (a ** 0.5).tolist(), (a ** -1).tolist()]",
    # a float32 array compares against a Python float rounded to float32: 0.1f > 0.1
    "f32_compare": "import numpy as np\ndef main():\n    a = np.full((2,), 0.1, np.float32)\n    return [(a > 0.1).tolist(), (a == 0.1).tolist()]",
    # arange in float32 fills start + i * delta, each step rounded
    "arange_f32": "import numpy as np\ndef main():\n    return np.arange(0.1, 2.0, 0.3, np.float32).tolist()",
    "zeros_bool_int": "import numpy as np\ndef main():\n    return [np.zeros((2, 2), np.bool_).tolist(), np.zeros(3, np.int32).tolist(), np.ones((2,), np.float64).tolist()]",
}


@pytest.mark.parametrize("name", CASES)
def test_ndarray(name):
    py, js = run_both(CASES[name])
    assert js == py


def test_unrounded_float32_is_caught(tmp_path):
    # with $rt.nd.sub writing Float64Array for float32 inputs (patched into the copied pyrt.js),
    # "f32_scalar" differs: the breaking test for "one array operation left unrounded". Each
    # operation casts its operands to its dtype, so an unrounded middle step rounds again at
    # the next one; the chain's last operation is the one that shows.
    src = {"mod": CASES["f32_scalar"]}
    py = run_python(src, tmp_path, "mod")
    broken = tmp_path / "broken.js"
    unrounded = (
        'const sub32 = nd.sub;\nnd.sub = (a, b, d) => sub32(a, b, d === "f32" ? "f64" : d);\n'
    )
    broken.write_text(PYRT.read_text() + unrounded)
    folder = tmp_path / "node"
    folder.mkdir()
    assert _run_node_with(src, folder, broken) != py


def _run_node_with(sources, folder: Path, pyrt: Path):
    """run_node, with pyrt as the module's runtime."""
    import json

    for name, mod in transpile_app(sources).items():
        (folder / f"{name}.js").write_text(mod.code)
    shutil.copy(pyrt, folder / "pyrt.js")
    script = f'import {{ main }} from "{(folder / "mod.js").as_uri()}";\nconsole.log(JSON.stringify(main()));'
    done = subprocess.run(
        ["node", "--input-type=module", "-e", script], capture_output=True, text=True, timeout=60
    )
    if done.returncode != 0:
        raise RuntimeError(f"node failed:\n{done.stderr}")
    return json.loads(done.stdout)


BAD_BROADCAST = "import numpy as np\ndef main():\n    a = np.ones((3, 2))\n    b = np.ones((3,))\n    return (a + b).tolist()"


def test_a_broadcast_numpy_refuses_fails_in_the_page_too():
    # (3, 2) + (3,): Python raises ValueError; Node throws "operands could not be broadcast
    # together with shapes (3,2) (3,)"
    with tempfile.TemporaryDirectory() as tmp:
        with pytest.raises(ValueError, match=r"operands could not be broadcast together with shapes \(3,2\) \(3,\)"):  # fmt: skip
            run_python({"mod": BAD_BROADCAST}, Path(tmp), "mod")
        with pytest.raises(RuntimeError) as err:
            run_node({"mod": BAD_BROADCAST}, Path(tmp), "mod")
    assert "ValueError: operands could not be broadcast together with shapes (3,2) (3,)" in str(
        err.value
    )


def test_power_by_another_exponent_fails_the_export():
    # a ** 3 on an array: TranspileError "... `a ** 3`, a power numpy does not compute
    # exactly in the page (2, 0.5 and -1 are in; use a * a * a) ..."
    src = "import numpy as np\ndef main():\n    a = np.ones((2, 2), np.float32)\n    return (a ** 3).tolist()"
    with pytest.raises(TranspileError) as err:
        transpile_app({"mod": src})
    assert err.value.line == 4
    assert (
        "`a ** 3`, a power numpy does not compute exactly in the page "
        "(2, 0.5 and -1 are in; use a * a * a)" in str(err.value)
    )


# ---------------------------------------------------------------- functions, reductions, indexing

# (k, 3) arrays of each dtype: k = 1, 7, 9 and 300 fall below, inside and past numpy's
# pairwise blocks of 8 and 128 (3k elements flat).
DTYPES = {"f64": "np.float64", "f32": "np.float32"}
SIZES = (1, 7, 9, 300)


def with_data(body: str, k: int, dtype: str) -> str:
    return (
        "import numpy as np\n"
        "def main():\n"
        f"    x = np.arange(0.0, {k}.0, 1.0, {DTYPES[dtype]}) * 0.731 - 2.3\n"
        "    a = np.stack([x, x * 0.7 - 1.3, x * x * 0.01 - 0.5], axis=1)\n"
        f"    return {body}\n"
    )


FUNCTIONS = {
    "sqrt_floor": "[np.sqrt(abs(a)).tolist(), np.floor(a).tolist()]",
    "clip": "np.clip(a, -1.0, 1.0).tolist()",
    "minimum_maximum": "[np.minimum(a, 0.5).tolist(), np.maximum(a, a * 0.5).tolist()]",
    "norm": "np.linalg.norm(a, axis=1).tolist()",
    "sum": "[float(np.sum(a)), np.sum(a, axis=0).tolist(), np.sum(a, axis=1).tolist()]",
    "mean": "[float(np.mean(a)), np.mean(a, axis=0).tolist(), np.mean(a, axis=1).tolist()]",
    "min": "[float(np.min(a)), np.min(a, axis=0).tolist(), np.min(a, axis=1).tolist()]",
    "max": "[float(np.max(a)), np.max(a, axis=0).tolist(), np.max(a, axis=1).tolist()]",
    "where_index": "[w.tolist() for w in np.where(a > 0.5)]",
    # np.inf is a Python float: a float32 a stays float32, which the division shows
    "where_choose": "(np.minimum(np.where(a > 0.5, a, np.inf), 9.0) / 3.0).tolist()",
    "stack_columns": "[np.stack([x, x * 2.0], axis=0).tolist(), np.column_stack([x, x * 2.0]).tolist()]",
    "argmin_argmax": "[int(np.argmin(a)), int(np.argmax(a)), int(np.argmin(x)), int(np.argmax(x))]",
}


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("k", SIZES)
@pytest.mark.parametrize("name", FUNCTIONS)
def test_numpy_functions(name, k, dtype):
    py, js = run_both(with_data(FUNCTIONS[name], k, dtype))
    assert js == py


# numpy's float32 sin, cos and exp are its own kernels; the page computes in float64 and
# rounds, which may differ in the last float32 bits (float64: V8's against numpy's).
TRANSCENDENTALS = (
    "[np.sin(a).tolist(), np.cos(a).tolist(), np.exp(np.clip(a, -20.0, 20.0)).tolist()]"
)
# The most float32 steps the page may stand from numpy: numpy's float32 exp is itself up to
# two steps from the correctly rounded value; float64 differs in one step at most.
ULPS = {"sin": 1, "cos": 1, "exp": 2}


def ulps(want, got) -> "np.ndarray":
    """How many representable values of want's dtype lie between want and got."""

    ints = {np.float32: np.int32, np.float64: np.int64}[want.dtype.type]
    got = np.asarray(got, dtype=want.dtype)
    return np.abs(want.view(ints).astype(np.int64) - got.view(ints).astype(np.int64))


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("k", SIZES)
def test_transcendentals_within_a_step(k, dtype):
    import numpy as np

    py, js = run_both(with_data(TRANSCENDENTALS, k, dtype))
    np_dtype = np.float32 if dtype == "f32" else np.float64
    for name, p, j in zip(ULPS, py, js, strict=True):
        steps = ulps(np.array(p, dtype=np_dtype), j)
        assert steps.max() <= (ULPS[name] if dtype == "f32" else 1), name


INDEXING = {
    "argmin_tie": "import numpy as np\ndef main():\n    return [int(np.argmin(np.array([3.0, 1.0, 1.0, 2.0]))), int(np.argmax(np.array([3.0, 1.0, 3.0])))]",
    "argmin_nan": "import numpy as np\ndef main():\n    a = np.array([3.0, 0.0, 1.0])\n    b = a / a\n    return [int(np.argmin(b)), int(np.argmax(b)), int(np.argmin(np.array([2.0, np.nan, 0.0])))]",
    "min_nan": "import numpy as np\ndef main():\n    a = np.stack([np.array([1.0, np.nan]), np.array([0.5, 2.0])], axis=1)\n    return [str(float(np.min(a))), [str(v) for v in np.max(a, axis=1).tolist()]]",
    "getitem": with_data(
        "[a[2].tolist(), a[:, 1].tolist(), a[a[:, 1] > 0.5].tolist(), a[np.array([2, 0])].tolist(), "
        "float(a[:, 1][0]), float(a[-1][2]), x[x > 1.0].tolist(), len(a), list(a.shape)]",
        9,
        "f32",
    ),
    "vec_mask": "import numpy as np\ndef main():\n    v = np.array([0.5, 2.0, 3.0])\n    return [v[v > 1.0].tolist(), v[np.array([2, 0])].tolist()]",
    "sum_mask": with_data("[int(np.sum(a > 0.5)), int(np.sum(np.where(a > 0.5)[0]))]", 9, "f64"),
}


@pytest.mark.parametrize("name", INDEXING)
def test_indexing(name):
    py, js = run_both(INDEXING[name])
    assert js == py


def test_an_index_past_the_end_throws_numpys_index_error():
    src = "import numpy as np\ndef main():\n    a = np.ones((3, 2))\n    return a[3].tolist()"
    with tempfile.TemporaryDirectory() as tmp:
        with pytest.raises(IndexError, match="index 3 is out of bounds for axis 0 with size 3"):
            run_python({"mod": src}, Path(tmp), "mod")
        with pytest.raises(RuntimeError, match="IndexError: index 3 is out of bounds for axis 0 with size 3"):  # fmt: skip
            run_node({"mod": src}, Path(tmp), "mod")


def test_a_write_through_an_index_fails_the_export():
    # a[mask] = 0.0 in a function reached code calls: TranspileError naming the line and
    # "`a[mask] = 0.0`, a write through an index (build a new array with np.where)"
    src = (
        "import numpy as np\n"
        "def zero(a, mask):\n"
        "    a[mask] = 0.0\n"
        "    return a\n"
        "def main():\n"
        "    a = np.ones((3, 2))\n"
        "    return zero(a, a > 0.5).tolist()\n"
    )
    with pytest.raises(TranspileError) as err:
        transpile_app({"mod": src})
    assert err.value.line == 3
    assert "`a[mask] = 0.0`, a write through an index (build a new array with np.where)" in str(err.value)  # fmt: skip


@pytest.mark.parametrize("method", ["sum", "min", "max", "mean"])
def test_a_reduction_method_fails_naming_the_function(method):
    src = f"import numpy as np\ndef main():\n    a = np.ones((3, 2))\n    return a.{method}()"
    with pytest.raises(TranspileError) as err:
        transpile_app({"mod": src})
    assert f"`a.{method}()` (write np.{method}(a))" in str(err.value)


def test_float32_transcendentals_are_measured(tmp_path, capsys):
    # np.sin, np.cos, np.exp over 10,000 float32 values from np.linspace(-50, 50):
    # count how many differ from the page's (Math.sin in float64, then rounded to float32),
    # and assert every difference is at most one float32 step (rel 2**-23). The count goes
    # in this plan's "where this plan departs" section. numpy's float32 exp is up to two
    # steps from the correctly rounded value, so exp is held to two (ULPS).
    import json

    import numpy as np

    values = np.linspace(-50, 50, 10_000).astype(np.float32)
    script = (
        f'import {{ nd, NdArray }} from "{PYRT.as_uri()}";\n'
        f'const v = new NdArray(Float32Array.from({json.dumps(values.tolist())}), [{len(values)}], "f32");\n'
        'const e = nd.exp(nd.clip(v, -50, 50, "f32"), "f32");\n'
        'console.log(JSON.stringify([nd.sin(v, "f32"), nd.cos(v, "f32"), e]));'
    )
    (tmp_path / "measure.mjs").write_text(script)
    done = subprocess.run(
        ["node", str(tmp_path / "measure.mjs")], capture_output=True, text=True, timeout=60
    )
    assert done.returncode == 0, done.stderr
    page = json.loads(done.stdout)
    counts = {}
    for name, got in zip(ULPS, page, strict=True):
        want = getattr(np, name)(values)
        steps = ulps(want, got)
        assert steps.max() <= ULPS[name], name
        counts[name] = [int((steps == 1).sum()), int((steps == 2).sum())]
    with capsys.disabled():
        print(f"\nfloat32 values one and two steps from numpy's, of 10,000: {counts}")
