import pytest


@pytest.fixture(scope="session")
def device():
    import wgpu

    try:
        adapter = wgpu.gpu.request_adapter_sync(power_preference="high-performance")
    except Exception as e:  # no Vulkan, Metal or D3D driver at all
        pytest.skip(f"no wgpu adapter: {e}")
    if adapter is None:
        pytest.skip("no wgpu adapter")
    from manifoldx.trace.tracer import _device

    return _device()
