"""Static packaging contracts for the phase-0 native wheel.

These run in the repo environment with no wheel built: they check that the
packaging files declare the phase-0 contracts. The checks that need the
artifact — a fresh install, the extension, the CLI, the negative control —
live in scripts/gates/ and run through `make native-verify`.
"""

import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _load(name: str) -> dict:
    with open(ROOT / name, "rb") as f:
        return tomllib.load(f)


def test_build_backend_is_maturin():
    build = _load("pyproject.toml")["build-system"]
    assert build["build-backend"] == "maturin"
    assert build["requires"] == ["maturin>=1.11,<2"]


def test_maturin_targets_the_production_extension():
    maturin = _load("pyproject.toml")["tool"]["maturin"]
    assert maturin["python-source"] == "src"
    assert maturin["module-name"] == "manifoldx._core_native"
    assert maturin["manifest-path"] == "crates/manifoldx-python/Cargo.toml"


def test_workspace_has_the_production_members_and_excludes_the_spikes():
    workspace = _load("Cargo.toml")["workspace"]
    assert set(workspace["members"]) == {
        "crates/manifoldx-core",
        "crates/manifoldx-python",
        "crates/manifoldx-render",
        "crates/manifoldx-sdk",
    }
    assert len(workspace["exclude"]) == 5
    assert all("spike" in excluded for excluded in workspace["exclude"])


def test_core_has_no_python_dependency():
    core = _load("crates/manifoldx-core/Cargo.toml")
    assert "dependencies" not in core


def test_extension_name_does_not_collide_with_the_spike_loader():
    # manifoldx.native's spike loader looks for `manifoldx._native`; the
    # production extension must keep a different name so the nbody/boids
    # spikes keep loading beside it.
    native = (ROOT / "src" / "manifoldx" / "native.py").read_text()
    assert "from manifoldx import _native" in native
    assert "_core_native" not in native


def test_distribution_stays_single_with_its_extras_and_entry_point():
    project = _load("pyproject.toml")["project"]
    assert project["name"] == "manifold-gfx"
    assert project["scripts"] == {"manifoldx": "manifoldx.cli:main"}
    for extra in ("desktop", "offline", "viz", "svg", "audio", "all"):
        assert extra in project["optional-dependencies"], extra
