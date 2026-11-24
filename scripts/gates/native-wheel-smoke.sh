#!/usr/bin/env bash
# Smoke a real installed wheel from a disposable cwd outside the checkout.
# Usage: native-wheel-smoke.sh <venv-python-or-bindir>
set -euo pipefail

PY="${1:?usage: native-wheel-smoke.sh <venv-python-or-bindir>}"
[ "$#" -eq 1 ] || { echo "FAIL: unexpected argument: $2" >&2; exit 1; }
[ ! -d "$PY" ] || PY="$PY/python"
BIN="$(cd "$(dirname "$PY")" && pwd -P)"
PY="$BIN/$(basename "$PY")"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
RUNDIR="$(mktemp -d "${TMPDIR:-/var/tmp}/manifoldx-wheel-smoke.XXXXXX")"
trap 'rm -rf "$RUNDIR"' EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
cd "$RUNDIR"
unset PYTHONPATH PYTHONHOME

"$PY" - "$ROOT" <<'PYTHON'
import importlib.machinery
import importlib.util
import pathlib
import sys
from importlib.metadata import entry_points

root = pathlib.Path(sys.argv[1]).resolve()
venv = pathlib.Path(sys.prefix).resolve()
cwd = pathlib.Path.cwd().resolve()
assert venv != pathlib.Path(sys.base_prefix).resolve(), "not a virtual environment"
assert not cwd.is_relative_to(root), f"cwd is inside the checkout: {cwd}"
for entry in sys.path:
    path = pathlib.Path(entry or cwd).resolve()
    assert not path.is_relative_to(root), f"sys.path points into the checkout: {path}"
print(f"interpreter: {sys.executable}; prefix: {venv}; cwd: {cwd}", flush=True)

import manifoldx

package = pathlib.Path(manifoldx.__file__).resolve()
assert package.is_relative_to(venv), f"manifoldx resolved outside the venv: {package}"
print(f"package: {package}", flush=True)
spec = importlib.util.find_spec("manifoldx._core_native")
if spec is None or spec.origin is None:
    sys.exit("FAIL: manifoldx._core_native is not installed in the venv")
extension = pathlib.Path(spec.origin).resolve()
assert extension.is_relative_to(venv), (
    f"manifoldx._core_native resolved outside the venv: {extension}"
)
assert isinstance(spec.loader, importlib.machinery.ExtensionFileLoader), (
    f"manifoldx._core_native is not a compiled extension: {extension}"
)
print(f"extension: {extension}", flush=True)

import manifoldx._core_native as native
from manifoldx import Engine

bootstrap = dict(native.bootstrap())
assert bootstrap["core"] == "manifoldx-core", bootstrap
assert bootstrap["runtime"] == "not implemented", bootstrap
assert bootstrap["version"], bootstrap
print(f"bootstrap: {bootstrap}; public API: {Engine.__name__}")
assert any(e.name == "manifoldx" for e in entry_points(group="console_scripts"))
ibl = package.parent / "assets" / "ibl"
assert ibl.is_dir() and any(ibl.iterdir()), f"package data missing: {ibl}"
print(f"entry point registered; package data: {ibl}")
PYTHON

# Inherit the same clean environment and disposable cwd for the console script.
"$BIN/manifoldx" --help
echo "wheel smoke passed from $PWD"
