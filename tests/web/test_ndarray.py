"""numpy arrays of any rank in frame code: each snippet's main() gives the same JSON value in
Python and in Node, bit for bit, under numpy 2's dtype rules."""

import shutil
import subprocess
import tempfile
from pathlib import Path

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
