import numpy as np

from manifoldx.audio.mixer import BUSES, CEILING, Mixer, Placed, Pose

RATE = 48000
LIS = Pose(0, 0, 0, 0, 0, 0, yaw=0.0)


class tone:
    """A numpy sine with block(n): manifoldx's tests do not use a synthesis library."""

    def __init__(self, f=1000.0):
        self.f, self.i = f, 0

    def block(self, n):
        t = (self.i + np.arange(n)) / RATE
        self.i += n
        return np.sin(2 * np.pi * self.f * t).astype(np.float32)


def rms(x):
    return float(np.sqrt(np.mean(np.square(x))))


def peak_hz(x):
    w = np.abs(np.fft.rfft(x * np.hanning(len(x))))
    return float(np.fft.rfftfreq(len(x), 1 / RATE)[np.argmax(w)])


def run(p, pose, blocks=10, n=4800):
    return np.concatenate([p.block(n, pose, LIS) for _ in range(blocks)])


def test_gain_is_full_within_ref_then_falls_as_one_over_d():
    near = Placed(tone(), ref=5).block(4800, Pose(5, 0, 0, 0, 0, 0), LIS)
    far = Placed(tone(), ref=5).block(4800, Pose(20, 0, 0, 0, 0, 0), LIS)
    assert abs(rms(near) - 0.5) < 0.02  # a sine of RMS 0.707, centred: 0.707 on each side
    assert abs(rms(far) / rms(near) - 0.25) < 0.02


def test_silent_past_far():
    assert rms(Placed(tone(), far=300).block(4800, Pose(400, 0, 0, 0, 0, 0), LIS)) == 0.0


def test_a_source_on_the_right_is_heard_on_the_right():
    x = Placed(tone()).block(4800, Pose(0, 0, 10, 0, 0, 0), LIS)
    assert rms(x[:, 1]) > 10 * rms(x[:, 0])


def test_a_source_ahead_is_centred():
    x = Placed(tone()).block(4800, Pose(10, 0, 0, 0, 0, 0), LIS)
    assert abs(rms(x[:, 0]) - rms(x[:, 1])) < 1e-3


def test_turning_the_listener_moves_the_source_across():
    x = Placed(tone()).block(4800, Pose(10, 0, 0, 0, 0, 0), Pose(0, 0, 0, 0, 0, 0, yaw=-np.pi / 2))
    assert rms(x[:, 1]) > 10 * rms(x[:, 0])  # facing -z, a source at +x is on the right


def test_far_sources_lose_their_highs():
    near = run(Placed(tone(8000.0), far=2000), Pose(30, 0, 0, 0, 0, 0))
    far = run(Placed(tone(8000.0), far=2000), Pose(600, 0, 0, 0, 0, 0))
    assert rms(far) / rms(near) < 30 / 600 * 0.7  # weaker than distance alone


def test_doppler_raises_an_approaching_tone_by_c_over_c_minus_v():
    x = run(Placed(tone(1000.0)), Pose(100, 0, 0, -20, 0, 0))[:, 0]
    assert abs(peak_hz(x) / (1000.0 * 343 / 323) - 1) < 0.005


def test_doppler_lowers_a_receding_tone():
    x = run(Placed(tone(1000.0)), Pose(100, 0, 0, 20, 0, 0))[:, 0]
    assert abs(peak_hz(x) / (1000.0 * 343 / 363) - 1) < 0.005


def test_doppler_is_clamped_for_a_teleport():
    x = run(Placed(tone(1000.0)), Pose(100, 0, 0, -2000, 0, 0))[:, 0]
    assert peak_hz(x) <= 1000.0 * 343 / 283 * 1.005


def test_a_position_jump_alone_changes_no_pitch():
    p = Placed(tone(1000.0))
    a = np.concatenate([p.block(4800, Pose(100, 0, 0, 0, 0, 0), LIS) for _ in range(5)])
    b = np.concatenate([p.block(4800, Pose(5000, 0, 0, 0, 0, 0), LIS) for _ in range(5)])
    assert abs(peak_hz(a[:, 0]) - 1000) < 5 and np.all(np.isfinite(b))


def test_the_limiter_holds_minus_one_dbfs():
    m = Mixer()
    x = 4 * np.sin(np.arange(48000) / 8)
    m["music"].add(np.stack([x, x], 1))
    y = m.out(48000)
    assert np.abs(y).max() <= CEILING + 1e-3 and abs(CEILING - 10 ** (-1 / 20)) < 1e-4


def test_quiet_sound_passes_the_limiter_untouched():
    m = Mixer()
    x = 0.3 * np.sin(np.arange(4800) / 8)
    m["music"].add(np.stack([x, x], 1))
    y = m.out(4800)
    np.testing.assert_allclose(y[240:, 0], x[:-240], atol=1e-6)  # delayed by the 5 ms lookahead


def level(m, bus, n=4800):
    m[bus].add(np.full((n, 2), 0.1))
    return rms(m.out(n))


def test_a_duck_lowers_a_bus_by_6_db_then_recovers():
    m = Mixer()
    base = [level(m, "music") for _ in range(3)][-1]
    m.duck("music", 6.0, 1.5)
    at = {}
    for k in range(25):  # 2.5 s in blocks of 0.1 s
        at[round((k + 1) * 0.1, 1)] = level(m, "music")
    assert abs(20 * np.log10(at[1.0] / base) + 6) < 0.5
    assert abs(20 * np.log10(at[2.0] / base)) < 0.5


def test_a_bus_at_zero_gain_is_silent():
    m = Mixer()
    m.gain("music", 0.0)
    for _ in range(3):
        y = level(m, "music")
    assert y == 0.0


def test_buses_are_the_five():
    assert BUSES == ("engine", "effects", "ambience", "music", "ui") and set(Mixer().buses) == set(BUSES)


def test_the_master_lowpass_muffles():
    m = Mixer()
    m.lowpass(400.0)
    for _ in range(3):
        m["effects"].add(np.stack([tone(4000.0).block(4800)] * 2, 1))
        y = m.out(4800)
    assert rms(y) < 0.1 * 0.707
