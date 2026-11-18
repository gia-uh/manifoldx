"""The wall-clock dt is smoothed: steady steps from uneven frame intervals, summing to the wall clock.

Under vsync the time an image stays on screen does not follow the frame's own start-to-start interval, so
moving things by that raw interval judders. The engine gives systems a running mean instead, kept in step
with the wall clock, and lets a hitch through as it is."""

import numpy as np
import pytest

from manifoldx.frame_clock import SmoothClock


def dts(intervals):
    clock = SmoothClock()
    return np.array([clock.step(i) for i in intervals])


def test_steady_intervals_give_that_dt():
    assert dts([1 / 60] * 50)[-1] == pytest.approx(1 / 60)


def test_uneven_intervals_give_steady_dts():
    rng = np.random.default_rng(0)
    raw = rng.uniform(0.014, 0.026, 600)
    out = dts(raw)[60:]
    change = np.abs(np.diff(out)) / out[:-1]
    assert np.percentile(change, 95) < 0.05
    raw_change = np.abs(np.diff(raw[60:])) / raw[60:-1]
    assert np.percentile(raw_change, 95) > 0.3  # the input really was uneven


def test_dts_sum_to_the_wall_clock():
    rng = np.random.default_rng(1)
    raw = rng.uniform(0.010, 0.040, 1000)
    assert abs(dts(raw).sum() - raw.sum()) < 0.05


def test_a_hitch_passes_through():
    out = dts([1 / 60] * 30 + [0.5] + [1 / 60] * 5)
    assert out[30] == pytest.approx(0.5)
    assert out[31] == pytest.approx(1 / 60, rel=0.05)


def test_the_engine_smooths_dt_unless_asked_not_to(monkeypatch):
    import manifoldx as mx
    import manifoldx.engine as engine_module

    def run(**kwargs):
        engine = mx.Engine("clock", **kwargs)
        times = iter(np.cumsum([0.0] + [0.014, 0.026] * 50) * 1e9)
        monkeypatch.setattr(engine_module, "perf_counter_ns", lambda: int(next(times)))
        return [engine._compute_dt() for _ in range(101)][-20:]

    assert np.ptp(run()) < 0.002
    assert np.ptp(run(smooth_dt=False)) == pytest.approx(0.012)
