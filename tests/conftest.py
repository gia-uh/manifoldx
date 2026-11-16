"""pytest configuration: mock display-dependent modules, skip GPU tests without a GPU."""

import pytest
from unittest.mock import MagicMock, patch


@pytest.fixture(autouse=True)
def mock_rendercanvas(monkeypatch):
    """Mock rendercanvas to avoid window creation in tests.

    Note: We don't mock rendercanvas globally because we need the real
    modules for testing. Instead, we only mock glfw-specific imports.
    """
    import sys

    # Only mock glfw - let offscreen use real implementation
    if "rendercanvas.glfw" not in sys.modules:
        mock_glfw = MagicMock()
        mock_glfw.loop = MagicMock()
        sys.modules["rendercanvas.glfw"] = mock_glfw

    yield


@pytest.fixture(autouse=True)
def ibl_cache_in_tmp(monkeypatch, tmp_path_factory):
    """The IBL precompute caches to disk; tests never write to the user's ~/.cache."""
    monkeypatch.setenv("MANIFOLDX_CACHE_DIR", str(tmp_path_factory.getbasetemp() / "manifoldx-cache"))


def _gpu_adapter_available() -> bool:
    try:
        import wgpu

        return wgpu.gpu.request_adapter_sync() is not None
    except Exception:
        return False


def pytest_configure(config):
    """Without a GPU adapter (CI runners), every adapter request skips its test.

    Most render tests only guard on `get_offscreen_canvas`, which does not
    touch the GPU, so on a GPU-less machine they used to error inside
    `Engine._init_canvas`. Patching the adapter request turns every such
    path, in fixtures or test bodies, into a clean skip with a reason.
    """
    if _gpu_adapter_available():
        return
    import wgpu

    def _no_gpu(*args, **kwargs):
        pytest.skip("no GPU adapter on this machine")

    async def _no_gpu_async(*args, **kwargs):
        pytest.skip("no GPU adapter on this machine")

    # The first request above replaced wgpu.gpu with the backend's object.
    wgpu.gpu.request_adapter_sync = _no_gpu
    wgpu.gpu.request_adapter_async = _no_gpu_async



_tests_since_collect = 0


@pytest.fixture(autouse=True)
def collect_gpu_devices():
    """Run the collector every 25 tests, so engines' GPU devices are freed.

    A wgpu device and its queue point at each other, so only the collector
    frees a device. On free-threaded Python it runs too rarely for a suite
    that builds an engine per test, and after about 60 devices the next
    request fails. Engine.close() does the same for one engine, at 45 ms each.
    """
    global _tests_since_collect
    yield
    _tests_since_collect += 1
    if _tests_since_collect >= 25:
        _tests_since_collect = 0
        import gc

        gc.collect()
