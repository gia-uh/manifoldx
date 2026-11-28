"""The native bounded diagnostic's GPU timings, read the way `Engine.gpu_timings()` returns them."""

from manifoldx._desktop import _NativeEngine


class _Report:
    """An engine carrying only what the accessor reads: the run report."""

    def __init__(self, report):
        self.desktop_report = report


def timings(report):
    return _NativeEngine.gpu_timings(_Report(report))


def measured(samples, state="measured"):
    return {"gpu_timings": {"state": state, "unit": "ns"}, "samples": samples}


def test_no_report_means_no_timings():
    assert timings(None) is None


def test_unmeasured_states_return_none():
    for state in ("not_bounded", "disabled", "unavailable"):
        assert timings(measured([{"gpu_render_ns": 9000}], state)) is None
    assert timings({"samples": [{"gpu_render_ns": 9000}]}) is None  # a report without the field


def test_measured_without_a_readback_returns_none():
    assert timings(measured([{"gpu_render_ns": None}])) is None
    assert timings(measured([])) is None


def test_newest_measured_frame_in_milliseconds_and_only_the_measured_pass():
    report = measured(
        [
            {"gpu_render_ns": 100_000},
            {"gpu_render_ns": None},  # a frame that did not present is not fabricated
            {"gpu_render_ns": 250_000},
        ]
    )
    assert timings(report) == {"main": 0.25, "total": 0.25}
