"""Video export keeps the requested size, changes every frame and cleans up on failure."""

from __future__ import annotations

import numpy as np
import pytest

from manifoldx.engine import Engine


def _read_video(path):
    imageio_ffmpeg = pytest.importorskip("imageio_ffmpeg")
    reader = imageio_ffmpeg.read_frames(str(path))
    meta = next(reader)
    frames = [np.frombuffer(frame, np.uint8) for frame in reader]
    return meta, frames


@pytest.mark.parametrize(("width", "height"), [(160, 100), (161, 101)])
def test_video_writer_keeps_the_requested_size(tmp_path, width, height):
    output = tmp_path / "size.mp4"
    writer = Engine._get_video_writer(None, output, 10, width, height, "h264", "high")
    writer.send(np.zeros((height, width, 3), np.uint8))
    writer.close()

    meta, frames = _read_video(output)
    assert meta["size"] == (width, height)
    assert len(frames) == 1


def test_video_writer_reports_an_encoder_failure(tmp_path):
    output = tmp_path / "bad.mp4"
    writer = Engine._get_video_writer(None, output, 10, 64, 64, "not-a-codec", "high")
    with pytest.raises(RuntimeError, match="video encoder"):
        writer.close()
    assert not output.exists()


def test_video_writer_explains_a_missing_dependency(tmp_path, monkeypatch):
    import sys

    monkeypatch.setitem(sys.modules, "imageio_ffmpeg", None)
    with pytest.raises(ImportError, match=r"manifold-gfx\[offline\]"):
        Engine._get_video_writer(None, tmp_path / "missing.mp4", 10, 64, 64, "h264", "high")


def test_video_writer_rejects_an_unknown_quality(tmp_path):
    with pytest.raises(ValueError, match="quality"):
        Engine._get_video_writer(None, tmp_path / "bad-quality.mp4", 10, 64, 64, "h264", "huge")


def _native_engine(width=160, height=100):
    native = pytest.importorskip("manifoldx._core_native")
    if not hasattr(native, "_NativeGraphics"):
        pytest.skip("native graphics is unavailable")
    import manifoldx as mx

    class Toggle:
        def __init__(self):
            self.frames = 0

        def paint(self, query, dt):
            self.frames += 1
            engine.background_color = (1.0, 0.0, 0.0) if self.frames <= 2 else (0.0, 1.0, 0.0)

    engine = mx.Engine("video", width=width, height=height)
    engine.system(Toggle().paint)
    return engine


def test_native_video_changes_in_order(tmp_path, monkeypatch):
    monkeypatch.setenv("MANIFOLDX_CORE", "rust")
    engine = _native_engine()
    output = tmp_path / "native.mp4"
    engine.render(output=str(output), fps=10, frame_count=4, quality="high")

    meta, frames = _read_video(output)
    assert meta["size"] == (160, 100)
    assert meta["fps"] == 10.0
    assert len(frames) == 4
    corner = [frame.reshape(100, 160, 3)[4, 4] for frame in frames]
    assert all(pixel[0] > 200 and pixel[1] < 30 for pixel in corner[:2])
    assert all(pixel[1] > 200 and pixel[0] < 30 for pixel in corner[2:])


def test_native_video_failure_removes_the_partial_file(tmp_path, monkeypatch):
    monkeypatch.setenv("MANIFOLDX_CORE", "rust")
    native = pytest.importorskip("manifoldx._core_native")
    if not hasattr(native, "_NativeGraphics"):
        pytest.skip("native graphics is unavailable")
    import manifoldx as mx

    engine = mx.Engine("video fail", width=160, height=100)

    @engine.system
    def boom(query, dt):
        if engine.elapsed >= 0.1:
            raise RuntimeError("deliberate frame failure")

    output = tmp_path / "fail.mp4"
    with pytest.raises(RuntimeError, match="deliberate frame failure"):
        engine.render(output=str(output), fps=10, frame_count=4, quality="high")
    assert not output.exists()
