#!/usr/bin/env bash
# The installed-wheel verification for the phase-0 native gate: a fresh venv,
# the wheel installed into it, the smoke gate from outside the checkout, and
# the negative control (the extension renamed inside the disposable venv; the
# smoke must fail, then pass again when it is restored).
#
# Usage: native-verify.sh <python> <dist-dir>
#
# Builds nothing: `make native-build` runs first (native-verify depends on
# it). The wheel is chosen for the interpreter's ABI tag, so a dist with both
# the GIL and the free-threaded wheel works. The venv is disposable and
# removed at the end; needs uv on PATH.

set -euo pipefail

PY_ARG="${1:?usage: native-verify.sh <python> <dist-dir>}"
DIST="${2:?usage: native-verify.sh <python> <dist-dir>}"
HERE="$(cd "$(dirname "$0")" && pwd)"

command -v uv >/dev/null || { echo "FAIL: uv is not on PATH" >&2; exit 1; }
[ -d "$DIST" ] || { echo "FAIL: no dist dir at $DIST (run make native-build first)" >&2; exit 1; }

# The tag computation needs a runnable interpreter path; uv can also be
# handed a name such as python3.14t, so resolve it first.
if [ ! -x "$PY_ARG" ] && ! command -v "$PY_ARG" >/dev/null 2>&1; then
    PY_ARG="$(uv python find "$PY_ARG" 2>/dev/null)" \
        || { echo "FAIL: cannot resolve interpreter: $1" >&2; exit 1; }
fi

TAG="$("$PY_ARG" -c 'import sys, sysconfig
t = "t" if sysconfig.get_config_var("Py_GIL_DISABLED") else ""
print(f"cp{sys.version_info.major}{sys.version_info.minor}{t}")')"
WHEEL="$(ls "$DIST"/manifold_gfx-*-"$TAG"-*.whl 2>/dev/null | head -n 1)"
[ -n "$WHEEL" ] || { echo "FAIL: no $TAG wheel in $DIST" >&2; exit 1; }
echo "wheel:  $WHEEL"

VENV="$(dirname "$DIST")/native-smoke-venv-$TAG"
rm -rf "$VENV"
trap 'rm -rf "$VENV"' EXIT
uv venv --python "$PY_ARG" "$VENV" >/dev/null
uv pip install --python "$VENV/bin/python" "$WHEEL" >/dev/null
echo "venv:   $VENV"

SO="$("$VENV/bin/python" -c 'import manifoldx._core_native, pathlib
print(pathlib.Path(manifoldx._core_native.__file__))')"
echo "ext:    $SO"

echo "--- smoke (installed, from outside the checkout) ---"
bash "$HERE/native-wheel-smoke.sh" "$VENV/bin/python"

echo "--- negative control: extension renamed inside the disposable venv ---"
mv "$SO" "$SO.gate-missing"
if ! bash "$HERE/native-wheel-smoke.sh" "$VENV/bin/python" --expect-missing; then
    mv "$SO.gate-missing" "$SO"
    echo "FAIL: the smoke passed with the extension removed" >&2
    exit 1
fi

echo "--- restored: the smoke must pass again ---"
mv "$SO.gate-missing" "$SO"
bash "$HERE/native-wheel-smoke.sh" "$VENV/bin/python" | tail -n 1

echo "native-verify: passed for $TAG"
