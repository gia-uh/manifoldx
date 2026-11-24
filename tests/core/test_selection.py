"""Startup selection contracts, not native ECS parity."""

import importlib
import sys
import sysconfig

import pytest

import manifoldx
import manifoldx.engine as engine_module
from manifoldx import Engine
from manifoldx._core import _selection
from manifoldx.ecs import EntityStore
from manifoldx.systems import SystemRegistry


@pytest.fixture
def extension_path(monkeypatch, tmp_path):
    # Use the ordinary importer, including the real extension loader for broken binaries.
    monkeypatch.setattr(manifoldx, "__path__", [str(tmp_path)])
    monkeypatch.delitem(sys.modules, "manifoldx._core_native", raising=False)
    importlib.invalidate_caches()
    return tmp_path


@pytest.mark.parametrize("value", [None, "", "python"])
def test_python_constructs_existing_engine_without_native_import(monkeypatch, value):
    if value is None:
        monkeypatch.delenv("MANIFOLDX_CORE", raising=False)
    else:
        monkeypatch.setenv("MANIFOLDX_CORE", value)

    class RejectNative:
        def find_spec(self, fullname, path=None, target=None):
            if fullname == "manifoldx._core_native":
                pytest.fail("Python selection imported the native extension")

    monkeypatch.delitem(sys.modules, "manifoldx._core_native", raising=False)
    monkeypatch.setattr(sys, "meta_path", [RejectNative(), *sys.meta_path])
    engine = Engine("selection", max_entities=16)
    assert type(engine.store) is EntityStore
    assert type(engine.systems) is SystemRegistry
    assert engine._device is None
    assert _selection() == ("python", None)


@pytest.mark.parametrize("value", ["invalid", "Rust", " python", " "])
def test_invalid_selection(monkeypatch, value):
    monkeypatch.setenv("MANIFOLDX_CORE", value)
    with pytest.raises(ValueError, match="Invalid MANIFOLDX_CORE"):
        Engine("selection")


@pytest.mark.parametrize("broken", [False, True])
def test_native_import_failure_is_early(monkeypatch, extension_path, broken):
    monkeypatch.setenv("MANIFOLDX_CORE", "rust")
    if broken:
        (extension_path / ("_core_native" + sysconfig.get_config_var("EXT_SUFFIX"))).write_bytes(
            b"broken"
        )
    calls = []

    def observe(*args, **kwargs):
        calls.append(True)
        raise AssertionError("Python initialization reached")

    monkeypatch.setattr(engine_module, "_warm_wgpu_types", observe)
    monkeypatch.setattr(engine_module, "EntityStore", observe)
    engine = Engine.__new__(Engine)
    message = "could not load" if broken else "requires manifoldx._core_native"
    with pytest.raises(ImportError, match=message) as failure:
        engine.__init__("selection")
    assert isinstance(failure.value.__cause__, ImportError)
    assert not calls
    assert vars(engine) == {}


def test_missing_native_dependency_is_not_reported_as_missing_extension(
    monkeypatch, extension_path
):
    monkeypatch.setenv("MANIFOLDX_CORE", "rust")
    (extension_path / "_core_native.py").write_text("import nonexistent_core_dependency\n")
    with pytest.raises(ImportError, match="could not load") as failure:
        _selection()
    assert failure.value.__cause__.name == "nonexistent_core_dependency"
