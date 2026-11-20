import sys
import time
import wave

import numpy as np
import pytest

from manifoldx.audio import Audio

RATE = 48000


class Sine:
    def __init__(self, read, name=None, f=440.0):
        self.read, self.name, self.f, self.ph = read, name, f, 0.0

    def block(self, n):
        f = self.read(self.name) if self.name else self.f
        ph = self.ph + 2 * np.pi * f * np.arange(1, n + 1) / RATE
        self.ph = float(ph[-1])
        return (0.5 * np.sin(ph)).astype(np.float32)


class Const:
    def __init__(self, value, seconds):
        self.value, self.duration = value, seconds

    def block(self, n):
        return np.full(n, self.value, np.float32)


def sine_440(ctl):
    return Sine(ctl)


def sine_from(ctl):
    return Sine(ctl, "freq")


def long_click(**values):
    return Const(0.01, 5.0)


def read_wav(path):
    with wave.open(str(path)) as w:
        x = np.frombuffer(w.readframes(w.getnframes()), "<i2").reshape(-1, 2) / 32767
    return x


def peak_hz(x):
    x = x[:, 0] if x.ndim == 2 else x
    w = np.abs(np.fft.rfft(x * np.hanning(len(x))))
    return float(np.fft.rfftfreq(len(x), 1 / RATE)[np.argmax(w)])


def wait_alive(a, timeout=20):
    t = time.time()
    while time.time() - t < timeout and not a.state()["alive"]:
        time.sleep(0.05)
    assert a.state()["alive"]


def test_the_file_backend_records_a_source(tmp_path):
    a = Audio(backend="file", path=tmp_path / "out.wav")
    wait_alive(a)
    a.source("tone", sine_440, "effects")
    time.sleep(1.2)
    a.close()
    x = read_wav(tmp_path / "out.wav")
    assert len(x) / RATE >= 1.0 and abs(peak_hz(x[RATE // 4:]) - 440) < 3


def test_controls_reach_a_source(tmp_path):
    a = Audio(backend="file", path=tmp_path / "out.wav", controls=["freq"])
    a.set("freq", 880.0)
    a.tick()
    wait_alive(a)
    a.source("tone", sine_from, "effects")
    time.sleep(1.0)
    a.close()
    x = read_wav(tmp_path / "out.wav")
    assert abs(peak_hz(x[RATE // 4:]) - 880) < 3


def test_stems_write_one_file_per_bus(tmp_path):
    a = Audio(backend="file", path=tmp_path / "drive.wav", stems=True)
    wait_alive(a)
    a.source("tone", sine_440, "music")
    time.sleep(0.6)
    a.close()
    music, engine = read_wav(tmp_path / "drive.music.wav"), read_wav(tmp_path / "drive.engine.wav")
    assert np.abs(music).max() > 0.1 and np.abs(engine).max() == 0.0


def test_events_play_voices_up_to_their_cap():
    a = Audio(backend="null")
    wait_alive(a)
    a.voice("hit", long_click, "effects", cap=4)
    for _ in range(6):
        a.event("hit")
    time.sleep(0.5)
    s = a.state()
    a.close()
    assert s["voices"]["effects"] == 4


def test_a_missing_device_falls_back_to_null(monkeypatch):
    monkeypatch.setenv("MANIFOLDX_AUDIO_DEVICE", "no-such-device-xyz")
    a = Audio(backend="device")
    wait_alive(a)
    s = a.state()
    a.close()
    assert s["backend"] == "null" and s["error"]


def test_a_dead_process_does_not_block_the_app():
    a = Audio(backend="null", controls=["rpm"])
    wait_alive(a)
    a._process.kill()
    a._process.join(5)
    t = time.perf_counter()
    a.set("rpm", 1.0)
    a.tick()
    a.event("hit", x=0.0)
    a.gain("music", 0.5)
    assert time.perf_counter() - t < 0.01 and not a.state()["alive"]
    a.close()


def test_a_stream_adds_chunks_with_overlap(tmp_path):
    a = Audio(backend="file", path=tmp_path / "out.wav", streams={"music": "music"})
    w = a.stream("music")
    w.push(0.0, np.full(int(1.5 * RATE), 0.1, np.float32))
    w.push(1.0, np.full(int(1.5 * RATE), 0.1, np.float32))
    wait_alive(a)
    time.sleep(2.0)
    clock = w.clock()
    a.close()
    x = read_wav(tmp_path / "out.wav")[:, 0]
    assert abs(np.median(x[int(1.1 * RATE):int(1.4 * RATE)]) - 0.2) < 0.01
    assert abs(np.median(x[int(0.3 * RATE):int(0.8 * RATE)]) - 0.1) < 0.01
    assert clock > 1.5


def test_reload_uses_the_edited_factory(tmp_path, monkeypatch):
    mod = tmp_path / "audio_reload_probe.py"
    src = (
        "import numpy as np\n"
        "class T:\n"
        "    def __init__(self, f): self.f, self.i = f, 0\n"
        "    def block(self, n):\n"
        "        t = (self.i + np.arange(n)) / 48000; self.i += n\n"
        "        return (0.5 * np.sin(2 * np.pi * self.f * t)).astype(np.float32)\n"
        "def tone(ctl): return T(FREQ)\n"
    )
    mod.write_text(src.replace("FREQ", "440.0"))
    monkeypatch.syspath_prepend(str(tmp_path))
    import audio_reload_probe

    a = Audio(backend="file", path=tmp_path / "out.wav")
    wait_alive(a)
    a.source("tone", audio_reload_probe.tone, "effects")
    time.sleep(0.8)
    mod.write_text(src.replace("FREQ", "660.0"))
    a.reload(["audio_reload_probe"])
    a.source("tone", audio_reload_probe.tone, "effects")
    time.sleep(1.0)
    a.close()
    x = read_wav(tmp_path / "out.wav")
    assert abs(peak_hz(x[int(0.2 * RATE):int(0.7 * RATE)]) - 440) < 3
    assert abs(peak_hz(x[-int(0.5 * RATE):]) - 660) < 3
    sys.modules.pop("audio_reload_probe", None)


def test_no_underruns_under_a_light_load():
    a = Audio(backend="null")
    wait_alive(a)
    a.source("tone", sine_440, "music")
    time.sleep(3)
    s = a.state()
    a.close()
    assert s["underruns"] == 0 and 0 < s["cpu"] < 0.6


def test_a_file_backend_needs_a_path():
    with pytest.raises(ValueError):
        Audio(backend="file")
