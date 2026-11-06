"""Small numpy vectors, float32 and math: each snippet's main() gives the same JSON value in
Python and in Node."""

import pytest

from .parity import run_both

CASES = {
    "broadcast": "import numpy as np\ndef main(): a = np.array([1.0, 2.0, 3.0]); return list(a * 2 + a / 4 - 1)",
    "inplace_alias": "import numpy as np\ndef main():\n    a = np.zeros(3); b = a\n    a += np.array([1.0, 2.0, 3.0])\n    return list(b)",
    "slice_norm": "import numpy as np\ndef main(): v = np.array([3.0, 4.0, 12.0]); return [float(np.linalg.norm(v)), list(v[:2])]",
    "arange_floats": "import numpy as np\ndef main(): return [float(t) for t in np.arange(0.1, 2.0, 0.3)]",
    "float32_ops": "import numpy as np\ndef main():\n    p = np.array([0.1, 0.2, 0.3], np.float32); t = np.array([1.7, -2.9, 5.3], np.float32)\n    d = t - p\n    return [float(x) for x in d / np.linalg.norm(d)]",
    "radians": "import math\ndef main(): return [math.radians(x) for x in (90.0, 33.3, -12.7)] + [math.degrees(1.234)]",
    # 359.0 and 0.1 are where x * pi / 180 and x * 180 / pi round away from CPython's.
    "radians_rounding": "import math\ndef main(): return [math.radians(359.0), math.degrees(0.1)]",
    # A float32 scalar stays float32 against a Python number (numpy 2); each value below
    # differs from the same arithmetic in float64.
    "float32_scalars": "import numpy as np\ndef main():\n    x = np.array([0.1, 0.7, 0.3], np.float32)\n    return [float(x[0] * 3.0 + x[1]), float(np.linalg.norm(x) / 7.0)]",
    # += into a float32 array rounds the Python scalar to float32 first.
    "float32_inplace": "import numpy as np\ndef main():\n    a = np.array([8.72, 5.07], np.float32)\n    a += 8.701\n    return [float(x) for x in a]",
    "mixed_precision": "import numpy as np\ndef main(): return list(np.array([0.1, 0.2], np.float32) + np.array([0.1, 0.2]))",
}


@pytest.mark.parametrize("name", CASES)
def test_numpy(name):
    py, js = run_both(CASES[name])
    assert js == py
