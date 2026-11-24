"""Cheap rejection checks. Native CI tests success with its real built wheel.

The blocking native-verify.sh step runs the installed -> missing -> restored
ordinary smoke; Python fixtures here must never stand in for a working extension.
"""

import os
import pathlib
import subprocess
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
SMOKE = ROOT / "scripts/gates/native-wheel-smoke.sh"
VERIFY = ROOT / "scripts/gates/native-verify.sh"
pytestmark = pytest.mark.skipif(os.name == "nt", reason="the gate scripts are bash")


@pytest.fixture
def installation(tmp_path):
    venv = tmp_path / "venv"
    subprocess.run([sys.executable, "-m", "venv", "--without-pip", str(venv)], check=True)
    site = next(venv.glob("lib/python*/site-packages"))
    package = site / "manifoldx"
    package.mkdir()
    (package / "__init__.py").write_text("")
    return venv, site, package


def smoke(venv, tmp_path):
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    result = subprocess.run(
        ["bash", str(SMOKE), str(venv / "bin/python")],
        cwd=ROOT,
        env={
            **os.environ,
            "TMPDIR": str(scratch),
            "PYTHONPATH": str(ROOT / "src"),
            "PYTHONHOME": "/nonexistent-python-home",
        },
        capture_output=True,
        text=True,
    )
    assert not list(scratch.iterdir()), "smoke failed to clean up its disposable cwd"
    assert result.returncode != 0, result.stdout + result.stderr
    return result.stdout + result.stderr


def test_smoke_rejects_a_python_module_as_the_native_extension(installation, tmp_path):
    venv, _, package = installation
    (package / "_core_native.py").write_text("def bootstrap(): return {}\n")
    assert "not a compiled extension" in smoke(venv, tmp_path)


def test_smoke_reports_the_missing_extension(installation, tmp_path):
    venv, _, _ = installation
    assert "FAIL: manifoldx._core_native is not installed in the venv" in smoke(venv, tmp_path)


def test_smoke_rejects_a_package_behind_a_venv_prefix(installation, tmp_path):
    venv, _, package = installation
    outside = tmp_path / "venv-evil" / "manifoldx"
    outside.parent.mkdir()
    package.rename(outside)
    package.symlink_to(outside, target_is_directory=True)
    assert "manifoldx resolved outside the venv" in smoke(venv, tmp_path)


def test_smoke_rejects_an_extension_symlink_outside_the_venv(installation, tmp_path):
    import sysconfig

    venv, _, package = installation
    outside = tmp_path / "venv-evil" / "native.so"
    outside.parent.mkdir()
    outside.touch()
    (package / ("_core_native" + sysconfig.get_config_var("EXT_SUFFIX"))).symlink_to(outside)
    assert "manifoldx._core_native resolved outside the venv" in smoke(venv, tmp_path)


def test_verify_rejects_ambiguous_same_abi_wheels(tmp_path):
    import sysconfig

    pyver = f"cp{sys.version_info.major}{sys.version_info.minor}"
    abi = pyver + ("t" if sysconfig.get_config_var("Py_GIL_DISABLED") else "")
    for version in ("0.1.0", "0.2.0"):
        (tmp_path / f"manifold_gfx-{version}-{pyver}-{abi}-linux_x86_64.whl").touch()
    result = subprocess.run(
        ["bash", str(VERIFY), sys.executable, str(tmp_path)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode != 0
    assert f"expected one wheel for {pyver}/{abi}, found 2" in result.stderr
