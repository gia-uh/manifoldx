"""manifoldx runs on free-threaded CPython: no dependency may turn the GIL back on."""

import os
import subprocess
import sys

import pytest

IMPORTS = "import manifoldx, numpy, wgpu, sys; print(sys._is_gil_enabled())"


def _gil_after_imports(env: dict) -> str:
    out = subprocess.run(
        [sys.executable, "-c", IMPORTS], capture_output=True, text=True, check=True, env=env
    )
    return out.stdout.strip()


def test_gil_is_off_after_importing_manifoldx():
    env = {k: v for k, v in os.environ.items() if k != "PYTHON_GIL"}
    assert _gil_after_imports(env) == "False"


@pytest.mark.skipif(os.environ.get("PYTHON_GIL") != "1", reason="the gil CI job sets PYTHON_GIL=1")
def test_python_gil_1_turns_it_on():
    assert _gil_after_imports(dict(os.environ)) == "True"
