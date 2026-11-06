"""Text: Python counts code points where JS strings count UTF-16 units, rounds format ties to
even on the exact binary value, and writes 3.0 as "3.0". Each snippet's main() gives the
same JSON value in Python and in Node."""

import pytest

from manifoldx.web import TranspileError, transpile_app

from .parity import run_both

NAME = "Biblioteca Central de Ciencias Matemáticas y Física Teóricas"  # 60 code points
HYPHENS = "Ruta Bogotá-Lima y estilo hispano-árabe en la región andina 😀😀"
LONG_WORD = "Código ab-123456789012 del año--siguiente, señor"

CASES = {
    "len": 'def main(): return [len("Matemática"), len("a😀b"), len("Matema\\u0301tica"), len("")]',
    # 0.05 is just above the tie, 2.25 and 12.75 are exact ties (to even), 3.0 has no digits
    # to round; -0.04 keeps its sign; 1e22 is past where toFixed switches to exponents.
    "fixed_ties": (
        "def main():\n"
        '    out = [f"{x:.1f}" for x in (0.05, 2.25, 12.75, 3.0, -0.04, 0.25, 1e22)]\n'
        '    return out + [f"{x:.0f}" for x in (0.5, 1.5, 2.5, -2.5)] + [f"{2.675:.2f}", f"{3:.2f}"]'
    ),
    "str_float_int": (
        "def main():\n"
        "    a = 3.0\n"
        "    b = 3\n"
        "    c = None\n"
        '    return [f"{a}", f"{b}", str(a), str(b), f"{a}|{b}", f"{True}", f"{c}", f"{-0.0}",\n'
        '            f"{0.1 + 0.2}", f"{1e16}", f"{1e-5}", f"{1.5e-7}", f"{123456789.0}", f"{2.5e15}"]'
    ),
    # numpy formats a float32 in an f-string as the Python float it holds.
    "fstring_float32": (
        "import numpy as np\n"
        "def main():\n"
        "    v = np.array([0.1, 3.0], np.float32)\n"
        '    return [f"{v[0]}", f"{v[1]}", f"{v[0]:.3f}"]'
    ),
    "fstring_text": (
        "def main():\n"
        '    name = "Matemática"\n'
        "    x = 2.25\n"
        '    return f"{name} 😀: {x:.1f} ({len(name)})"'
    ),
    "wrap": (
        "import textwrap\n"
        "def main():\n"
        "    out = []\n"
        f"    for text in ({NAME!r}, {HYPHENS!r}, {LONG_WORD!r}):\n"
        "        for width in (10, 28):\n"
        "            out.append(textwrap.wrap(text, width))\n"
        f"    return out + [textwrap.wrap({NAME!r}, width=28), textwrap.wrap('  ', 5)]"
    ),
    "str_methods": (
        "def main():\n"
        '    s = "  Óptica Física  "\n'
        '    return [s.strip(), s.lower(), s.upper(), s.strip().startswith("Ópt"),\n'
        '            s.endswith("x"), ", ".join(["a", "b"])]'
    ),
}


@pytest.mark.parametrize("name", CASES)
def test_text(name):
    py, js = run_both(CASES[name])
    assert js == py


def test_the_name_is_sixty_code_points():
    assert len(NAME) == 60


def test_an_int_or_float_in_an_fstring_with_no_spec_fails():
    src = "def f(flag: bool):\n    x = 1 if flag else 2.5\n    return f'{x}'\n"
    with pytest.raises(TranspileError, match="annotate"):
        transpile_app({"mod": src})


def test_an_unknown_value_in_an_fstring_with_no_spec_fails():
    with pytest.raises(TranspileError, match="annotate `x`"):
        transpile_app({"mod": "def f(x):\n    return f'{x}'\n"})
