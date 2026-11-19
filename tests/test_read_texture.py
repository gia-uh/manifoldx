"""Engine.read_texture: a render_to_texture texture read back as RGB pixels."""

import numpy as np
import pytest

import manifoldx as mx
from manifoldx.camera import Camera


def _offscreen_engine():
    try:
        from manifoldx.backends import get_offscreen_canvas

        get_offscreen_canvas(width=8, height=8)
    except Exception as e:
        pytest.skip(f"offscreen canvas unavailable: {e}")
    eng = mx.Engine("t", width=32, height=32)
    eng.render_frame(supersample=1)  # gives the engine its offscreen device
    return eng


def _wait(eng, future, frames=4):
    for _ in range(frames):
        if future.done():
            break
        eng.render_frame(supersample=1)
    assert future.done(), f"read_texture did not resolve within {frames} frames"
    return future.result()


@pytest.mark.parametrize("live", [True, False])
def test_read_texture_returns_the_rendered_background(live):
    eng = _offscreen_engine()
    cam = Camera()
    cam.set_pose(position=(0, 0, 6), target=(0, 0, 0))
    # 70 px rows are 280 bytes: the readback has to drop the 256-byte alignment padding.
    # A live=False view has not rendered yet: the read waits for its first render.
    handle = eng.render_to_texture(cam, (70, 30), live=live, background="#336699")
    pixels = _wait(eng, eng.read_texture(handle))
    assert pixels.shape == (30, 70, 3) and pixels.dtype == np.uint8
    assert np.abs(pixels.astype(int) - (0x33, 0x66, 0x99)).max() <= 2


def test_read_texture_of_a_still_view_reads_its_last_image():
    eng = _offscreen_engine()
    handle = eng.render_to_texture(Camera(), (16, 16), live=False, background="#c08040")
    eng.render_frame(supersample=1)  # its one render
    pixels = _wait(eng, eng.read_texture(handle))
    assert np.abs(pixels.astype(int) - (0xC0, 0x80, 0x40)).max() <= 2
