"""@ on matrix stacks, np.eye, nested np.array and np.stack of any rank in frame code: each
snippet's main() gives the same JSON value in Python and in Node. A float32 @ is numpy's
forward chain, each exact product added in float64 and rounded to float32 (plan 5c,
departure 1), so it compares bit for bit."""

import tempfile
from pathlib import Path

import numpy as np
import pytest

from manifoldx.web import transpile_app
from manifoldx.web.replay import REL
from manifoldx.web.types import TranspileError

from .parity import PYRT, run_both, run_node, run_python

# mat(i, n): an (n, n) matrix of dtype D from nested rows of sines, a different one per i.
HEAD = """import math
import numpy as np


def mat(i: int, n: int):
    return np.array([[math.sin((i * n * n + r * n + c) * 0.37) * 3.0 for c in range(n)] for r in range(n)], dtype=D)


def stack(i: int, k: int, n: int):
    return np.stack([mat(i * k + j, n) for j in range(k)])

"""


def snippet(body: str, dtype: str = "np.float32") -> str:
    return HEAD.replace("dtype=D", f"dtype={dtype}") + "\ndef main():\n" + body


STACKS = snippet(
    "    a = np.stack([stack(i, 5, 4) for i in range(40)])\n"
    "    b = np.stack([stack(i + 40, 5, 4) for i in range(40)])\n"
    "    return (a @ b).tolist()\n"
)


def _fma() -> bool:
    try:
        return " fma " in Path("/proc/cpuinfo").read_text().replace("\n", " ")
    except OSError:
        return False


def test_float32_stacks_bit_for_bit():
    py, js = run_both(STACKS)
    assert np.array(py).shape == (40, 5, 4, 4)
    differ = int(np.sum(np.array(py, np.float32) != np.array(js, np.float32)))
    print(f"float32 @ on (40, 5, 4, 4): {differ} of 3200 values differ; cpu fma: {_fma()}")
    assert js == py


def test_one_matrix_broadcasts_over_a_stack():
    py, js = run_both(
        snippet(
            "    m = mat(99, 4)\n"
            "    s = stack(0, 30, 4)\n"
            "    return [(m @ s).tolist(), (s @ m).tolist()]\n"
        )
    )
    assert np.array(py[0]).shape == (30, 4, 4)
    assert js == py


def test_three_by_three():
    py, js = run_both(snippet("    return (stack(0, 1000, 3) @ stack(1, 1000, 3)).tolist()\n"))
    assert np.array(py).shape == (1000, 3, 3)
    assert js == py


def test_float64_within_rel():
    py, js = run_both(
        snippet(
            "    a = np.stack([stack(i, 5, 4) for i in range(40)])\n"
            "    b = np.stack([stack(i + 40, 5, 4) for i in range(40)])\n"
            "    return (a @ b).tolist()\n",
            "np.float64",
        )
    )
    p, j = np.array(py), np.array(js)
    assert p.shape == j.shape == (40, 5, 4, 4)
    assert np.all(np.abs(p - j) <= REL * np.abs(p))


def test_mixed_dtypes_promote():
    # float32 @ float64 is float64 (numpy 2): a float32 result would differ by far more than REL
    py, js = run_both(
        snippet(
            "    a = stack(0, 6, 4)\n"
            "    b = np.stack([np.array([[math.cos(r + c * 0.5) for c in range(4)] for r in range(4)], dtype=np.float64)] * 6)\n"
            "    return (a @ b).tolist()\n"
        )
    )
    p, j = np.array(py), np.array(js)
    assert np.any(p != p.astype(np.float32))
    assert np.all(np.abs(p - j) <= REL * np.abs(p))


def test_eye_nested_array_and_stack():
    py, js = run_both(
        "import math\nimport numpy as np\n\n\ndef main():\n"
        "    c, s = math.cos(0.3), math.sin(0.3)\n"
        "    e = np.eye(4, dtype=np.float32)\n"
        "    r = np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], dtype=np.float32)\n"
        "    rows = np.array([np.array([1.0, 2.0]), np.array([3.0, 4.0])])\n"
        "    three = np.stack([e, e * 2.0, e * 3.0])\n"
        "    two = np.stack([three, three * 0.5])\n"
        "    return [e.tolist(), np.eye(3).tolist(), r.tolist(), rows.tolist(), list(three.shape), "
        "list(two.shape), two.tolist()]\n"
    )
    assert py[4] == [3, 4, 4] and py[5] == [2, 3, 4, 4]
    assert js == py


def _both_raise(src: str):
    with tempfile.TemporaryDirectory() as tmp:
        with pytest.raises(ValueError) as py:
            run_python({"mod": src}, Path(tmp), "mod")
        with pytest.raises(RuntimeError) as js:
            run_node({"mod": src}, Path(tmp), "mod")
    return str(py.value), str(js.value)


def test_a_core_dimension_mismatch_raises_numpys_message():
    py, js = _both_raise(
        "import numpy as np\n\n\ndef main():\n"
        "    return (np.ones((4, 3), np.float32) @ np.ones((4, 4), np.float32)).tolist()\n"
    )
    want = (
        "matmul: Input operand 1 has a mismatch in its core dimension 0, with gufunc signature "
        "(n?,k),(k,m?)->(n?,m?) (size 4 is different from 3)"
    )
    assert py == want and f"ValueError: {want}" in js


def test_leading_dimensions_that_do_not_broadcast_raise_numpys_message():
    py, js = _both_raise(
        "import numpy as np\n\n\ndef main():\n"
        "    return (np.ones((2, 4, 4), np.float32) @ np.ones((3, 4, 4), np.float32)).tolist()\n"
    )
    assert f"ValueError: {py}" in js


def test_matrix_times_vector_fails_the_export():
    src = (
        "import numpy as np\n\n\ndef main():\n"
        "    m = np.eye(3)\n"
        "    v = np.array([1.0, 2.0, 3.0])\n"
        "    return (m @ v).tolist()\n"
    )
    with pytest.raises(TranspileError) as err:
        transpile_app({"mod": src})
    assert err.value.line == 7
    assert (
        "`m @ v`, a matrix times a vector, which numpy rounds in an order the page cannot match "
        "(write the vector as a column: m @ np.array([[x], [y], [z]]))" in str(err.value)
    )


def test_eye_takes_n_and_a_dtype_only():
    src = "import numpy as np\n\n\ndef main():\n    return np.eye(3, 4).tolist()\n"
    with pytest.raises(TranspileError) as err:
        transpile_app({"mod": src})
    assert "this call to np.eye()" in str(err.value)


def test_stack_along_axis_1_takes_vectors_only():
    src = "import numpy as np\n\n\ndef main():\n    e = np.eye(2)\n    return np.stack([e, e], axis=1).tolist()\n"
    with pytest.raises(TranspileError):
        transpile_app({"mod": src})


def test_a_ragged_nested_array_raises_numpys_message():
    py, js = _both_raise("import numpy as np\n\n\ndef main():\n    return np.array([[1.0, 2.0], [3.0]]).tolist()\n")
    want = (
        "setting an array element with a sequence. The requested array has an inhomogeneous "
        "shape after 1 dimensions. The detected shape was (2,) + inhomogeneous part."
    )
    assert py == want and f"ValueError: {want}" in js


# The float32 step of nd.matmul in pyrt.js, and the same step with each product rounded first.
CHAIN = "acc = Math.fround(acc + A[ia + p] * B[ib + p * m]);"
ROUNDED = "acc = Math.fround(acc + Math.fround(A[ia + p] * B[ib + p * m]));"


def test_rounding_each_product_first_is_a_mismatch(tmp_path):
    src = PYRT.read_text()
    assert src.count(CHAIN) == 1
    broken = tmp_path / "pyrt.js"
    broken.write_text(src.replace(CHAIN, ROUNDED))
    py = run_python({"mod": STACKS}, tmp_path, "mod")
    folder = tmp_path / "node"
    folder.mkdir()
    for name, mod in transpile_app({"mod": STACKS}).items():
        (folder / f"{name}.js").write_text(mod.code)
    (folder / "pyrt.js").write_text(broken.read_text())
    import json
    import subprocess

    script = f'import {{ main }} from "{(folder / "mod.js").as_uri()}";\nconsole.log(JSON.stringify(main()));'
    done = subprocess.run(["node", "--input-type=module", "-e", script], capture_output=True, text=True, timeout=60)
    assert done.returncode == 0, done.stderr
    assert json.loads(done.stdout) != py
