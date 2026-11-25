"""Verify kernel v1 through an installed wheel. Run outside the checkout with -I."""

import argparse
import json
import math
import os
import platform
import struct
import sys
import time
from pathlib import Path
from runpy import run_path
from tempfile import TemporaryDirectory

import manifoldx as mx
from manifoldx import clamp, f32, vec3
from manifoldx.compute import ComputeShaderCompileError, GpuKernelGroup

ROOT = Path(__file__).resolve().parents[2]
FIXTURES = ROOT / "tests/core/fixtures"
CORPUS = json.loads((FIXTURES / "drive-v1.json").read_text())
APP = run_path(str(ROOT / "examples/kernel_drive.py"))
Car, Player, drive = (APP[name] for name in ("Car", "Player", "drive"))
make_world, install_steering, steering_target = (
    APP[name] for name in ("make_world", "install_steering", "steering_target")
)


@mx.kernel
def changed_clamp(car: Car, dt: f32) -> None:
    car.speed += clamp(car.target_speed - car.speed, -2.0 * dt, 3.0 * dt)
    car.pos += car.forward * car.speed * dt


@mx.kernel
def advance_x(car: Car, dt: f32) -> None:
    car.pos += car.forward * dt


@mx.kernel
def reverse_x(car: Car, dt: f32) -> None:
    car.pos -= dt * car.forward


@mx.kernel
def compare_speed(car: Car, dt: f32) -> None:
    step: f32 = dt * 2.0
    if car.speed > 0.0:
        car.speed += step
    else:
        car.speed -= step


@mx.kernel
def store_local(car: Car, dt: f32) -> None:
    direction: vec3 = car.forward
    direction *= dt
    car.pos = car.pos + direction


def compare(actual, expected, label):
    for value, ref in zip(actual, expected, strict=True):
        assert math.isfinite(value) and abs(value - ref) <= 1e-5 + 1e-5 * abs(ref), (
            f"drive oracle mismatch ({label}): {actual} vs {expected}"
        )


def diagnostics():
    # E0's source fixtures declare an external Car for their source spans.
    mx.Car = Car
    try:
        fixed = run_path(str(FIXTURES / "kernel-v1.py"))["drive"]
        assert fixed.body[0].span == (8, 5, 8, 74), fixed.body[0].span
        assert fixed.wgsl() == drive.wgsl()
        expect_diagnostic(
            FIXTURES / "kernel-v1-invalid.py", 8, "unsupported-construct", "import numpy"
        )
        with TemporaryDirectory() as scratch:
            source = Path(scratch) / "bad_type.py"
            source.write_text(
                "from manifoldx import Car, f32, kernel\n\n"
                "def fixture():\n"
                "    @kernel\n"
                "    def wrong(car: Car, dt: f32) -> None:\n"
                "        car.pos = dt\n"
                "fixture()\n"
            )
            expect_diagnostic(source, 6, "type-error", "car.pos = dt", column=9)
            source.write_text(
                "from manifoldx import Car, f32, kernel\n"
                "import numpy as np\n"
                "@kernel\n"
                "def wrong(car: Car, dt: f32) -> None:\n"
                "    car.speed = np.float32(dt)\n"
            )
            expect_diagnostic(
                source, 5, "unsupported-construct", "car.speed = np.float32(dt)", column=17
            )
    finally:
        del mx.Car


def expect_diagnostic(source, line, category, snippet, *, column=5):
    try:
        run_path(str(source))
    except ComputeShaderCompileError as error:
        assert error.filename == str(source) and error.line == line, str(error)
        assert error.col == column and error.category == category, str(error)
        assert error.source_line.strip() == snippet, str(error)
        print(str(error))
    else:
        raise AssertionError(f"invalid fixture imported: {source.name}")


def car_row(speed, pos, target, forward):
    return Car(speed=speed, target_speed=target, pos=pos, forward=forward)


def pose(array):
    return (array[0], *array[2:5])


def f32_value(value):
    return struct.unpack("<f", struct.pack("<f", value))[0]


def oracle(speed, pos, target, forward, dt):
    # Test-only scalar model with an explicit binary32 rounding per operation.
    speed, target, dt = map(f32_value, (speed, target, dt))
    lo, hi = f32_value(-3.0 * dt), f32_value(2.0 * dt)
    speed = f32_value(speed + min(max(f32_value(target - speed), lo), hi))
    position = tuple(
        f32_value(f32_value(p) + f32_value(f32_value(f32_value(d) * speed) * dt))
        for p, d in zip(pos, forward, strict=True)
    )
    return speed, position


def checkpoints():
    seq = CORPUS["sequence"]
    speed, pos = seq["initial"]
    frame = 0
    for run in seq["runs"]:
        for _ in range(run["frames"]):
            speed, pos = oracle(speed, pos, run["target"], seq["forward"], seq["dt"])
            frame += 1
            yield (speed, *pos), run["target"]
        expected_speed, expected_pos = run["expected"]
        assert (speed, *pos) == (expected_speed, *expected_pos), f"invalid oracle at frame {frame}"


def isolated_cpu(program):
    for case in CORPUS["cases"]:
        speed, pos, target, forward, dt = case["input"]
        world = make_world(3)
        world.spawn(Player())
        world.spawn(car_row(speed, pos, target, forward))
        world.spawn(Player())
        assert world.dispatch_kernel(program, dt=dt, target="cpu") == 1
        expected_speed, expected_pos = case["expected"]
        compare(pose(world.query(Car).numpy()[0]), (expected_speed, *expected_pos), case["name"])

    seq = CORPUS["sequence"]
    expected = list(checkpoints())
    speed, pos = seq["initial"]
    world = make_world(3)
    (controlled,) = world.spawn(car_row(speed, pos, 0, seq["forward"]), Player())
    (secondary,) = world.spawn(Car(forward=(1, 0, 0)))
    world.spawn()
    install_steering(world, lambda frame: {"w"} if expected[frame][1] else set())
    world.kernel(program, target="cpu")
    observed = []

    @world.system(reads=(Car,))
    def observe(snapshot, commands):
        data = snapshot.query(Car).read()
        assert data.entities == (controlled, secondary)
        rows = data.numpy()
        assert pose(rows[1]) == (0, 0, 0, 0)
        observed.append((pose(rows[0]), rows[0, 1], data.version))

    report = world.run(frames=len(expected), paced=False)
    assert report["kernel_backend"] == "rust-cpu"
    assert report["kernel_dispatches"] == len(expected)
    assert report["kernel_rows"] == len(expected) * 2
    assert report["system_commands"] == len(expected)
    assert len(observed) == len(expected)
    assert all(a[2] < b[2] for a, b in zip(observed, observed[1:]))
    for frame, ((actual, target, _), (reference, expected_target)) in enumerate(
        zip(observed, expected, strict=True)
    ):
        compare(actual, reference, f"CPU frame {frame}")
        assert target == expected_target
        assert max(abs(a - b) for a, b in zip(actual[1:], reference[1:], strict=True)) <= 1e-3
    print(
        json.dumps(
            {
                "backend": "rust-cpu",
                "fixture_cases": len(CORPUS["cases"]),
                "sequence_frames": len(observed),
                "native_dispatches": report["kernel_dispatches"],
                "native_rows": report["kernel_rows"],
                "commands": report["system_commands"],
            }
        )
    )


def setup_gpu(program):
    import wgpu

    adapter = wgpu.gpu.request_adapter_sync(power_preference="high-performance")
    if adapter is None:
        raise RuntimeError("WGSL backend requires an available adapter")
    features = ["timestamp-query"] if "timestamp-query" in adapter.features else []
    device = adapter.request_device_sync(required_features=features)
    print(
        json.dumps(
            {
                "backend": "wgsl",
                "adapter": dict(adapter.info),
                "features": features,
                "wgpu": wgpu.__version__,
                "flags": {
                    name: os.environ.get(name)
                    for name in ("WGPU_BACKEND_TYPE", "WGPU_POWER_PREF", "VK_DRIVER_FILES")
                },
            }
        )
    )
    return GpuKernelGroup(device, program, rows=65, target="wgsl"), device


def gpu_upload(group, speed, pos, target, forward):
    row = (speed, target, *pos, *forward)
    group.load(struct.pack("<520f", *(row * 65)), struct.pack("<65I", 1, *([0] * 63), 1))
    return tuple(map(f32_value, row))


def gpu_pose(snapshot, untouched):
    values = struct.unpack("<520f", snapshot.data)
    assert values[8:-8] == untouched * 63, "WGSL changed unselected slots"
    assert values[:8] == values[-8:], "WGSL skipped a selected slot in the partial workgroup"
    return pose(values)


def isolated_gpu(program):
    group, _ = setup_gpu(program)
    for case in CORPUS["cases"]:
        speed, pos, target, forward, dt = case["input"]
        untouched = gpu_upload(group, speed, pos, target, forward)
        group.dispatch(program, dt=dt)
        expected_speed, expected_pos = case["expected"]
        compare(
            gpu_pose(group.readback(), untouched), (expected_speed, *expected_pos), case["name"]
        )

    seq = CORPUS["sequence"]
    expected = list(checkpoints())
    speed, pos = seq["initial"]
    target = None
    for frame, (reference, fixture_target) in enumerate(expected):
        current = steering_target({"w"} if fixture_target else set())
        if current != target:
            target = current
            untouched = gpu_upload(group, speed, pos, target, seq["forward"])
        group.dispatch(program, dt=seq["dt"])
        actual = gpu_pose(group.readback(), untouched)
        speed, *pos = actual
        compare(actual, reference, f"WGSL frame {frame}")
        assert max(abs(a - b) for a, b in zip(pos, reference[1:], strict=True)) <= 1e-3
    print(json.dumps({"backend": "wgsl", "sequence_frames": len(expected), **group.counters()}))


def timestamps(group, device, program):
    import wgpu

    if "timestamp-query" not in device.features:
        return {"available": False}
    query = device.create_query_set(type="timestamp", count=2)
    resolved = device.create_buffer(
        size=16,
        usage=wgpu.BufferUsage.QUERY_RESOLVE | wgpu.BufferUsage.COPY_SRC,
    )
    group.dispatch(
        program,
        dt=0.125,
        timestamp_writes={
            "query_set": query,
            "beginning_of_pass_write_index": 0,
            "end_of_pass_write_index": 1,
        },
    )
    encoder = device.create_command_encoder()
    encoder.resolve_query_set(query, 0, 2, resolved, 0)
    device.queue.submit([encoder.finish()])
    begin, end = struct.unpack("<QQ", device.queue.read_buffer(resolved, 0, 16))
    assert 0 < begin < end, (begin, end)
    return {"available": True, "begin": begin, "end": end, "unit": "adapter timestamp ticks"}


def residency(program, control):
    group, device = setup_gpu(program)
    untouched = gpu_upload(group, 0, (0, 0, 0), 30, (1, 0, 0))
    first = group.counters()
    kernels = (program, advance_x, reverse_x, compare_speed, store_local)
    for i, declared in enumerate(kernels):
        if control != "dispatch" or i != 2:
            group.dispatch(declared, dt=0.125)
        if control == "readback" and i == 0:
            group.readback()
    group.wait()
    after = group.counters()
    print(json.dumps({"backend": "wgsl", "before": first, "after": after}))
    assert after["dispatches"] == 5, "WGSL dispatch count differs from scheduled kernels"
    assert after["readback_calls"] == 0, "GPU-to-CPU transfer between resident dispatches"
    assert after["upload_bytes"] == first["upload_bytes"], "component re-upload between dispatches"
    sample = group.readback()
    compare(gpu_pose(sample, untouched), (0.5, 0.15625, 0, 0), "resident dispatch order")
    assert group.readback() is sample and group.counters()["readback_calls"] == 1
    assert group.counters()["readback_consumers"] == 2
    for _ in range(64):
        group.dispatch(program, dt=0)
    before_rejection = group.counters()
    try:
        group.dispatch(program, dt=0)
    except RuntimeError as error:
        assert "submission credit exhausted" in str(error), str(error)
    else:
        raise AssertionError("WGSL accepted a submission beyond its credit")
    assert group.counters() == before_rejection
    group.wait()
    stamp = timestamps(group, device, program)
    latest = group.readback()
    assert latest.version > sample.version
    compare(gpu_pose(sample, untouched), (0.5, 0.15625, 0, 0), "earlier owned version")
    if stamp["available"]:
        compare(gpu_pose(latest, untouched), (0.75, 0.25, 0, 0), "latest owned version")
    print(json.dumps({"backend": "wgsl", "timestamp_probe": stamp, **group.counters()}))


def performance(frames):
    rows = 100_000
    world = make_world(rows)
    world.spawn(Car(forward=(1, 0, 0), target_speed=30), count=rows)
    for program in (drive, advance_x, reverse_x, compare_speed, store_local):
        world.kernel(program, target="cpu")
    start = time.perf_counter_ns()
    report = world.run(frames=frames, paced=True)
    duration_ms = (time.perf_counter_ns() - start) / 1e6
    intervals = sorted(report["frame_intervals_ns"])
    assert len(intervals) == frames - 1 and report["frames"] == frames
    assert report["kernel_dispatches"] == frames * 5
    assert report["kernel_rows"] == frames * rows * 5
    assert world.rows_read == (0, 0, 0) and report["snapshot_copies"] == 0
    p99_ms = intervals[math.ceil(0.99 * len(intervals)) - 1] / 1e6
    over_33 = sum(interval > 33e6 for interval in intervals)
    print(
        json.dumps(
            {
                "backend": report["kernel_backend"],
                "frames": frames,
                "entities": rows,
                "kernels": 5,
                "p99_ms": p99_ms,
                "max_ms": intervals[-1] / 1e6,
                "over_33ms": over_33,
                "duration_ms": duration_ms,
                "intervals_ns": report["frame_intervals_ns"],
                "system_metrics": report["systems"],
                "snapshots": report["snapshot_copies"],
                "rows_read": world.rows_read,
                "kernel_rows": report["kernel_rows"],
            }
        )
    )
    assert p99_ms <= 16.7 and over_33 == 0, f"GW-2 timing: p99={p99_ms} ms, >33 ms={over_33}"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("cpu", "gpu", "residency", "performance", "diagnostics"))
    parser.add_argument("--control", choices=("clamp", "dispatch", "readback", "backend"))
    parser.add_argument("--frames", type=int, default=600)
    args = parser.parse_args()
    assert sys.flags.isolated and "site-packages" in Path(mx.__file__).parts, mx.__file__
    print(
        json.dumps(
            {
                "mode": args.mode,
                "control": args.control,
                "module": mx.__file__,
                "python": platform.python_version(),
                "gil": sys._is_gil_enabled(),
                "clock": time.get_clock_info("perf_counter").implementation,
            }
        )
    )
    program = changed_clamp if args.control == "clamp" else drive
    if args.control == "backend":
        GpuKernelGroup(None, program, rows=1, target="wgsl")
    if args.mode == "diagnostics":
        diagnostics()
    elif args.mode == "cpu":
        isolated_cpu(program)
    elif args.mode == "gpu":
        isolated_gpu(program)
    elif args.mode == "residency":
        residency(program, args.control)
    else:
        performance(args.frames)


if __name__ == "__main__":
    main()
