"""Exercise the public native World, or measure its installed 60 Hz clock."""

import argparse
import gc
import json
import sys
import threading
import time
from contextlib import contextmanager

import numpy as np

from manifoldx import Float, PythonGraphicsBridge, World
from manifoldx.components import Component


class Speed(Component):
    value: Float = 0.0


@contextmanager
def rejects(error, message):
    try:
        yield
    except error as exc:
        assert message in str(exc), str(exc)
    else:
        raise AssertionError(f"expected {error.__name__}: {message}")


def rows(*values):
    return np.array(values, dtype="f4").reshape(-1, 1)


def outcomes(world):
    result = {}
    while (outcome := world.outcome()) is not None:
        sequence, frame, error = outcome
        result[sequence] = (frame, error)
    return result


def contract(drop_first):
    world = World(4, queue_capacity=6, batches_per_frame=6)
    world.register(Speed)
    world.register("flags", kind="u32", width=1)
    speed = world.field(Speed, "value")
    flags = world.field("flags")
    cars = world.query(Speed)
    first, second = world.spawn(
        Speed(value=[1, 2]), count=2, payload={"flags": np.array([[3], [5]], "u4")}
    )
    (third,) = world.spawn(Speed(value=3))
    explicit = cars.read()
    assert explicit.numpy()[:, 0].tolist() == [1, 2, 3]

    buffer_calls = []

    class Exporter:
        def __buffer__(self, flags):
            buffer_calls.append(("get", cars.read().version))
            return memoryview(rows(1))

        def __release_buffer__(self, view):
            buffer_calls.append(("release", cars.read().version))

    world.replace(Speed, [first], Exporter())
    assert [name for name, _ in buffer_calls] == ["get", "release"]
    with rejects(TypeError, "dtype f32"):
        world.replace(Speed, [first], np.zeros((1, 1), dtype="f8"))
    with rejects(ValueError, "C-contiguous"):
        world.replace(Speed, [first, second], rows(8, 9, 10, 11)[::2])

    before = world.rows_read
    first_id = None if drop_first else world.set(cars.where(speed > 1), speed, 4)
    # The second live match lacks flags and must reject the entire write.
    bad_id = world.set(cars.where(speed > 3), flags, 9)
    destroy_id = world.destroy([third])
    spawn_id = world.submit_spawn(Speed(value=7))
    stale_id = world.destroy([third])
    set_id = world.set(cars.where(speed.eq(7)), speed, 8)
    if not drop_first:
        with rejects(RuntimeError, "queue is full"):
            world.submit_spawn(payload={Speed: Exporter()})
        assert len(buffer_calls) == 2
    assert world.rows_read == before, "creating or queuing a selection read rows"

    snapshots = []
    writes = []

    @world.system(reads=[Speed], writes=[speed])
    def earlier(snapshot, commands):
        assert snapshot.clock.frame == 0
        assert abs(snapshot.clock.dt - 1 / 60) < 1e-8
        with rejects(ValueError, "not declared"):
            snapshot.read("flags")
        with rejects(RuntimeError, "command lease"):
            world.set(cars, speed, 100)
        snapshots.append((snapshot, commands, snapshot.read(Speed)))
        writes.append(commands.emit)
        # Free the credits while still inside the first frame.
        seen = outcomes(world)
        assert first_id in seen, f"first command ID missing: {first_id!r} in {seen!r}"
        assert seen[first_id] == (0, None), seen
        assert "does not carry" in seen[bad_id][1], seen
        assert seen[destroy_id] == seen[spawn_id] == seen[set_id] == (0, None), seen
        assert "stale" in seen[stale_id][1].lower(), seen
        assert snapshots[-1][2].numpy()[:, 0].tolist() == [1, 4, 8]
        writes.append(commands.set(snapshot.query(Speed).where(speed.eq(8)), speed, 10))
        commands.emit("earlier", "committed")

    @world.system(reads=[Speed], writes=[speed])
    def later(snapshot, commands):
        assert snapshot.read(Speed).numpy()[:, 0].tolist() == [1, 4, 10]
        writes.append(commands.set(snapshot.query(Speed).where(speed.eq(10)), speed, 11))
        commands.emit("later", "committed")

    bridge_values = []

    def present(snapshot):
        assert threading.current_thread() is threading.main_thread()
        bridge_values.append(snapshot.read(Speed).numpy()[:, 0].tolist())

    result = world.run(frames=1, paced=False, graphics=PythonGraphicsBridge(present, reads=[Speed]))
    assert result["frames"] == 1
    assert result["events"] == [(0, "earlier", "committed"), (0, "later", "committed")]
    assert list(outcomes(world)) == writes[1:]
    assert bridge_values == [[1, 4, 11]]
    assert result["bridge"][:3] == (1, 3, 63)
    assert result["system_commands"] == 2
    assert world.queue_full == 1
    leased, commands, saved = snapshots[0]
    with rejects(RuntimeError, "lease expired"):
        leased.read(Speed)
    with rejects(RuntimeError, "lease expired"):
        commands.set(cars, speed, 0)
    with rejects(RuntimeError, "queue is stopped"):
        world.set(cars, speed, 9)
    with rejects(ValueError, "read-only"):
        saved.numpy()[0, 0] = 0
    assert saved.numpy()[:, 0].tolist() == [1, 4, 8]
    current = cars.read()
    assert current.entities[2].slot == third.slot
    assert current.entities[2].generation > third.generation
    assert current.version > saved.version > explicit.version
    assert world.query("flags").numpy()[:, 0].tolist() == [3, 5]
    foreign = World(1)
    foreign.register(Speed)
    with rejects(ValueError, "different World"):
        foreign.destroy([first])

    failures = []

    def foreign_thread():
        try:
            world.run(frames=1)
        except RuntimeError as error:
            failures.append(str(error))

    thread = threading.Thread(target=foreign_thread)
    thread.start()
    thread.join()
    assert failures == ["operation requires the World owner thread"]
    print(
        json.dumps(
            {
                "commands": result["system_commands"],
                "events": result["events"],
                "buffer": buffer_calls,
                "bridge": result["bridge"],
                "readback": world.rows_read,
                "queue_full": world.queue_full,
            }
        )
    )


def overlap():
    world = World(2)
    world.register(Speed)
    world.register("other", kind="u32", width=1)
    world.spawn(Speed(value=1))
    rendezvous = threading.Barrier(2, timeout=1)
    successful = []

    @world.system(hz=10, reads=[Speed], writes=[Speed])
    def first(snapshot, commands):
        if snapshot.clock.frame == 0:
            rendezvous.wait()
            time.sleep(0.15)
            successful.append("first")

    @world.system(hz=10, reads=["other"], writes=["other"])
    def independent(snapshot, commands):
        if snapshot.clock.frame == 0:
            rendezvous.wait()
            successful.append("independent")

    @world.system(hz=10, writes=[Speed])
    def conflicts(snapshot, commands):
        assert snapshot.clock.frame != 0

    report = world.run(frames=14)
    assert sorted(successful) == ["first", "independent"]
    assert report["systems"][2][2] > 0 and report["systems"][2][3] > 0, report
    assert report["snapshot_copies"] == 2, report
    print(
        json.dumps(
            {"independent_systems": report["systems"], "snapshot_copies": report["snapshot_copies"]}
        )
    )


def shutdown():
    world = World(1)
    world.register(Speed)
    world.spawn(Speed(value=1))
    speed = world.field(Speed, "value")
    staged = threading.Event()
    finish = threading.Event()
    accepted = []

    @world.system(hz=10, writes=[Speed])
    def pending(snapshot, commands):
        accepted.append(commands.set(snapshot.query(Speed), speed, 5))
        staged.set()
        assert finish.wait(2)

    @world.system
    def stop(snapshot, commands):
        if snapshot.clock.frame == 1:
            assert staged.wait(2)
            world.stop()
            finish.set()

    world.run(frames=10)
    closed = outcomes(world)
    assert len(accepted) == 1 and closed[accepted[0]][1] == "cancelled on stop", closed
    assert world.query(Speed).numpy()[0, 0] == 1
    print(json.dumps({"shutdown": closed}))


def stress(mode, frames):
    world = World(1)
    durations = []
    forced_gc_ns = []
    running = threading.Event()
    ended = threading.Event()

    def collect():
        gc.disable()
        try:
            while not ended.wait(0.7):
                trash = [[None] for _ in range(250_000)]
                for item in trash:
                    item[0] = item
                del item, trash
                start = time.perf_counter_ns()
                gc.collect()
                forced_gc_ns.append(time.perf_counter_ns() - start)
        finally:
            gc.enable()

    if mode != "base":

        @world.system(hz=10)
        def busy(snapshot, commands):
            start = time.perf_counter_ns()
            running.set()
            value = 1
            while time.perf_counter_ns() - start < 300_000_000:
                value = (value * 3 + 1) % 1_000_003
            durations.append(time.perf_counter_ns() - start)

    if mode == "sync-gc":

        @world.system
        def framed(snapshot, commands):
            if snapshot.clock.frame % 30 == 0:
                time.sleep(0.035)

    collector = None
    if mode.endswith("gc"):
        collector = threading.Thread(target=collect)
        collector.start()
    try:
        report = world.run(frames=frames)
    finally:
        ended.set()
        if collector:
            collector.join()
    intervals = sorted(report["frame_intervals_ns"])
    p99 = intervals[(99 * len(intervals) + 99) // 100 - 1] / 1e6
    assert report["frames"] == frames and len(intervals) == frames - 1
    assert report["bridge"][0] == world.rows_read[0] == 0
    if mode == "base":
        assert report["systems"] == []
    else:
        assert running.is_set() and min(durations) >= 300_000_000
        assert report["systems"][0][3] > 0, report
    if collector:
        assert forced_gc_ns
    if mode == "sync-gc":
        assert report["systems"][1][0].endswith("framed") and report["systems"][1][-1] >= 1
    print(
        json.dumps(
            {
                "mode": mode,
                "gil": sys._is_gil_enabled(),
                "frames": frames,
                "p99_ms": p99,
                "max_ms": intervals[-1] / 1e6,
                "over_33ms": sum(ns > 33e6 for ns in intervals),
                "async_ms": [ns / 1e6 for ns in durations],
                "gc_ms": [ns / 1e6 for ns in forced_gc_ns],
                "systems": report["systems"],
            }
        )
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--drop-first", action="store_true")
    parser.add_argument("--stress", choices=("base", "async", "async-gc", "sync-gc"))
    parser.add_argument("--frames", type=int, default=360)
    args = parser.parse_args()
    if args.stress:
        stress(args.stress, args.frames)
    else:
        contract(args.drop_first)
        overlap()
        shutdown()


if __name__ == "__main__":
    main()
