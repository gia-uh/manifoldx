"""Engine.close(): free the GPU device an engine requested, without the garbage collector."""

import gc

import pytest

import manifoldx as mx


def _engine(w=32, h=24):
    try:
        from manifoldx.backends import get_offscreen_canvas

        get_offscreen_canvas(width=8, height=8)
    except Exception as e:
        pytest.skip(f"offscreen canvas unavailable: {e}")
    return mx.Engine("test", width=w, height=h)


def test_close_drops_an_owned_device():
    e = _engine()
    e.render_frame(supersample=1)
    assert e._owns_device
    e.close()
    assert getattr(e, "_device", None) is None
    e.close()  # a second close does nothing


def test_close_keeps_a_shared_device():
    a = _engine()
    a.render_frame(supersample=1)
    b = _engine()
    b._ensure_offscreen(1, gpu=(a._adapter, a._device, a._texture_format))
    assert not b._owns_device
    b.close()
    assert a.render_frame(supersample=1).shape == (24, 32, 3)


def test_close_fires_shutdown_once():
    e = _engine()
    calls = []

    @e.on("shutdown")
    def on_stop(payload):
        calls.append(1)

    e.render_frame(supersample=1)
    e.close()
    e.close()
    assert calls == [1]


def test_eighty_engines_in_a_row():
    gc.disable()
    try:
        for _ in range(80):  # without close the 64th device request fails
            e = _engine(8, 8)
            e.render_frame(supersample=1)
            e.close()
    finally:
        gc.enable()
