#!/usr/bin/env bash
# Smoke the installed manifold-gfx wheel, from outside the checkout.
#
# Usage: native-wheel-smoke.sh <venv-python-or-bindir> [--expect-missing]
#
# The argument is the interpreter of the venv the wheel was installed in, or
# its bin directory. Every Python invocation runs with PYTHONPATH removed,
# so nothing from a checkout can satisfy the imports. With --expect-missing
# the same assertions run and the script exits 0 only when the extension
# assertion fails: the negative control for the wheel gate. Never point the
# script at the checkout's own environment.
#
# Asserted, in order:
#   1. manifoldx resolves inside this venv's site-packages
#   2. manifoldx._core_native resolves inside this venv and its bootstrap
#      reports the runtime as not implemented, from the Rust side
#   3. the public API imports (manifoldx.Engine)
#   4. the console script runs: manifoldx --help
#   5. the console entry point is registered
#   6. the wheel's package data still ships (assets/ibl)

set -u

fail() { echo "FAIL: $*" >&2; exit 1; }

PY_ARG="${1:?usage: native-wheel-smoke.sh <venv-python-or-bindir> [--expect-missing]}"
case "${2:-}" in
    "") EXPECT_MISSING=0 ;;
    --expect-missing) EXPECT_MISSING=1 ;;
    *) fail "unknown argument: $2" ;;
esac

if [ -d "$PY_ARG" ]; then
    PY="$PY_ARG/python"
    BIN="$PY_ARG"
else
    PY="$PY_ARG"
    BIN="$(dirname "$PY")"
fi
[ -x "$PY" ] || fail "no python at $PY"

VENV="$("$PY" -c 'import sys; print(sys.prefix)')" || fail "cannot read sys.prefix"
[ -n "$VENV" ] || fail "empty sys.prefix"

pyrun() { env -u PYTHONPATH "$PY" -c "$1"; }

# 1. the package resolves inside this venv
SP="$(pyrun 'import manifoldx, pathlib; print(pathlib.Path(manifoldx.__file__).resolve())')" \
    || fail "import manifoldx failed"
case "$SP" in
    "$VENV"*) : ;;
    *) fail "manifoldx resolved outside the venv: $SP (venv: $VENV)" ;;
esac
echo "ok 1  package resolves in the venv: $SP"

# 2. the extension and its bootstrap
if pyrun '
import manifoldx._core_native as m
b = dict(m.bootstrap())
assert b["runtime"] == "not implemented", b
assert b["core"] == "manifoldx-core", b
assert b["version"], b
print("bootstrap:", b)
'; then
    if [ "$EXPECT_MISSING" = 1 ]; then
        fail "the extension worked under --expect-missing: the negative control did not happen"
    fi
    echo "ok 2  extension imported and bootstrap ran"
else
    if [ "$EXPECT_MISSING" = 1 ]; then
        echo "ok 2  extension missing, as the negative control expects"
    else
        fail "manifoldx._core_native did not import, or its bootstrap failed"
    fi
fi

# 3. the public API imports
pyrun 'from manifoldx import Engine; print("Engine:", Engine.__name__)' \
    || fail "the public API did not import"
echo "ok 3  public API imports"

# 4. the console script runs
env -u PYTHONPATH "$BIN/manifoldx" --help >/dev/null || fail "manifoldx --help failed"
echo "ok 4  CLI --help runs"

# 5. the console entry point is registered
pyrun 'from importlib.metadata import entry_points
eps = [e for e in entry_points(group="console_scripts") if e.name == "manifoldx"]
assert eps, "no manifoldx console script registered"
print(eps[0].value)' || fail "the console entry point is missing"
echo "ok 5  console entry point registered"

# 6. the wheel's package data still ships
pyrun 'import manifoldx, pathlib
p = pathlib.Path(manifoldx.__file__).parent / "assets"
assert (p / "ibl").is_dir() and any((p / "ibl").iterdir()), p
print(p / "ibl")' || fail "package data missing"
echo "ok 6  package data ships"

if [ "$EXPECT_MISSING" = 1 ]; then
    echo "negative control held: the wheel smoke fails exactly when the extension is gone"
fi
