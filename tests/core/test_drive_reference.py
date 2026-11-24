"""Test-only binary32 oracle and immutable input/output corpus for kernel IR v1."""

import json
import math
import struct
from pathlib import Path

import pytest

CORPUS = json.loads(Path(__file__).with_name("fixtures").joinpath("drive-v1.json").read_text())


def f32(value):
    return struct.unpack("<f", struct.pack("<f", value))[0]


def drive(speed, pos, target, forward, dt):
    speed, target, dt = map(f32, (speed, target, dt))
    lo, hi = f32(-3.0 * dt), f32(2.0 * dt)
    speed = f32(speed + min(max(f32(target - speed), lo), hi))
    pos = [
        f32(f32(p) + f32(f32(f32(direction) * speed) * dt)) for p, direction in zip(pos, forward)
    ]
    return speed, pos


def check_step(actual, expected):
    values = [actual[0], *actual[1]]
    reference = [expected[0], *expected[1]]
    for value, ref in zip(values, reference, strict=True):
        assert math.isfinite(value)
        assert abs(value - ref) <= 1e-5 + 1e-5 * abs(ref)


@pytest.mark.parametrize("case", CORPUS["cases"], ids=lambda case: case["name"])
def test_drive_reference(case):
    assert drive(*case["input"]) == tuple(case["expected"])


def test_drive_accumulated_reference():
    sequence = CORPUS["sequence"]
    speed, pos = sequence["initial"]
    forward, dt = sequence["forward"], sequence["dt"]
    for run in sequence["runs"]:
        for _ in range(run["frames"]):
            speed, pos = drive(speed, pos, run["target"], forward, dt)
        assert (speed, pos) == tuple(run["expected"])


def test_step_comparison_rejects_reversed_acceleration_and_nonfinite():
    case = next(case for case in CORPUS["cases"] if case["name"] == "upper-clamp")
    speed, pos, _, forward, dt = case["input"]
    check_step(drive(*case["input"]), case["expected"])
    with pytest.raises(AssertionError):
        check_step(drive(speed, pos, -30, forward, dt), case["expected"])
    with pytest.raises(AssertionError):
        check_step((math.nan, pos), case["expected"])
