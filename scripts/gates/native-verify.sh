#!/usr/bin/env bash
# Install an existing wheel, run the ordinary smoke, remove the extension,
# require the SAME smoke to fail for that reason, restore it and smoke again.
# Usage: native-verify.sh <python> <dist-dir>; builds nothing, requires uv.
# NATIVE_VERIFY_LOGS retains the smoke logs after the disposable venv is removed.
set -euo pipefail
unset PYTHONPATH PYTHONHOME
fail() { echo "FAIL: $*" >&2; exit 1; }
PY_ARG="${1:?usage: native-verify.sh <python> <dist-dir>}"
DIST="${2:?usage: native-verify.sh <python> <dist-dir>}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
DIST="$(cd "$DIST" && pwd -P)"
[ -z "${NATIVE_VERIFY_LOGS:-}" ] || {
    mkdir -p "$NATIVE_VERIFY_LOGS"
    NATIVE_VERIFY_LOGS="$(cd "$NATIVE_VERIFY_LOGS" && pwd -P)"
}

# Keep uv's named interpreter lookup: python3.14t need not be on PATH.
if [ -x "$PY_ARG" ]; then
    PY="$PY_ARG"
elif command -v "$PY_ARG" >/dev/null 2>&1; then
    PY="$(command -v "$PY_ARG")"
else
    PY="$(uv python find "$PY_ARG")"
fi
PY="$(cd "$(dirname "$PY")" && pwd -P)/$(basename "$PY")"
TAGS="$("$PY" -c 'import sys, sysconfig; p = f"cp{sys.version_info.major}{sys.version_info.minor}"; print(p, p + ("t" if sysconfig.get_config_var("Py_GIL_DISABLED") else ""))')"
read -r PYVER ABI <<< "$TAGS"
WHEEL="$("$PY" - "$DIST" "$PYVER" "$ABI" <<'PYTHON'
import pathlib
import sys

dist, pyver, abi = sys.argv[1:]
candidates = []
for path in sorted(pathlib.Path(dist).glob("manifold_gfx-*.whl")):
    _, py_tags, abi_tags, _ = path.stem.rsplit("-", 3)
    if pyver in py_tags.split(".") and abi in abi_tags.split("."):
        candidates.append(path)
if len(candidates) != 1:
    sys.exit(f"FAIL: expected one wheel for {pyver}/{abi}, found {len(candidates)}: {candidates}")
print(candidates[0])
PYTHON
)"

echo "wheel: $WHEEL; interpreter: $PY; ABI: $PYVER/$ABI; invocation cwd: $PWD"
SCRATCH="$(mktemp -d "${TMPDIR:-/var/tmp}/manifoldx-native-verify.XXXXXX")"
cleanup() {
    local status=$?
    if [ -n "${NATIVE_VERIFY_LOGS:-}" ]; then
        cp "$SCRATCH"/*.log "$NATIVE_VERIFY_LOGS/" || echo "FAIL: could not retain logs" >&2
    fi
    rm -rf "$SCRATCH"
    exit "$status"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
cd "$SCRATCH"
VENV="$SCRATCH/venv"
uv venv --python "$PY" "$VENV"
uv pip install --python "$VENV/bin/python" "$WHEEL"
VENVPY="$VENV/bin/python"

# Capture exit codes directly, retain diagnostics even on an unexpected failure.
run_smoke() {
    local label="$1" status
    if bash "$HERE/native-wheel-smoke.sh" "$VENVPY" >"$SCRATCH/$label.log" 2>&1; then
        status=0
    else
        status=$?
    fi
    cat "$SCRATCH/$label.log"
    echo "$label exit: $status" | tee -a "$SCRATCH/status.log"
    return "$status"
}
run_entity_slots() {
    local label="$1" status
    if "$VENVPY" -I "$HERE/native-entity-slots.py" >"$SCRATCH/$label.log" 2>&1; then
        status=0
    else
        status=$?
    fi
    cat "$SCRATCH/$label.log"
    echo "$label exit: $status" | tee -a "$SCRATCH/status.log"
    return "$status"
}
run_smoke smoke-before
run_entity_slots entity-slots-before
"$VENVPY" -I "$HERE/native-selection-smoke.py" installed
SO="$("$VENVPY" -c 'import manifoldx._core_native as m; print(m.__file__)')"
mv "$SO" "$SO.gate-removed"
[ ! -e "$SO" ] && [ -f "$SO.gate-removed" ] || fail "ineffective removal: $SO"
if run_smoke smoke-removed; then
    fail "ordinary smoke passed with _core_native removed"
fi
grep -qx 'FAIL: manifoldx._core_native is not installed in the venv' "$SCRATCH/smoke-removed.log" \
    || fail "ordinary smoke failed for a reason other than missing _core_native"
if run_entity_slots entity-slots-missing; then
    fail "entity slot gate passed with _core_native removed"
fi
grep -q 'the installed extension has a module spec' "$SCRATCH/entity-slots-missing.log" \
    || fail "entity slot gate failed for a reason other than the missing extension"
"$VENVPY" -I "$HERE/native-selection-smoke.py" missing
mv "$SO.gate-removed" "$SO"
run_smoke smoke-restored
run_entity_slots entity-slots-restored
"$VENVPY" -I "$HERE/native-selection-smoke.py" installed
echo "native-verify: passed for $PYVER/$ABI (installed -> missing -> restored)"
