"""Run with an installed native wheel, from any cwd: python -I native_runtime.py.

NumPy prepares data once. Rust owns the World and executes every frame on the
calling thread. The separate async producer owns its thread and joins it.
"""

import argparse
import asyncio
import gc
import importlib.machinery
import threading
import time

import numpy as np
from manifoldx import _core_native as native


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def handles(result):
    return np.frombuffer(result["handles"], dtype="<u8").reshape(-1, 2).copy()


def values(result):
    return np.frombuffer(result["data"], dtype="<f4").copy()


def world(count, width=1):
    w = native._NativeWorld(count)
    w.register_component("position", "f32", width)
    w.register_component("velocity", "f32", width)
    ids = handles(
        w.spawn(
            count,
            {
                "position": np.zeros((count, width), np.float32),
                "velocity": np.ones((count, width), np.float32),
            },
        )
    )
    return w, ids


def rejected(call, text):
    try:
        call()
    except (RuntimeError, ValueError) as error:
        require(text in str(error), str(error))
    else:
        raise AssertionError(f"expected rejection: {text}")


def drain(runtime):
    results = []
    while (outcome := runtime.take_outcome()) is not None:
        results.append(outcome)
    return results


def deterministic(drop_first=False):
    w, ids = world(2)
    runtime = native.NativeRuntime(w, "position", "velocity", capacity=8, batches_per_frame=1)
    rejected(w.snapshot, "owned")
    rejected(lambda: native.NativeRuntime(w, "position", "velocity"), "owned")
    source = np.array([[100], [200]], dtype=np.float32)
    first = {"sequence": 1} if drop_first else runtime.submit_rows("position", ids, source)
    require(first["sequence"] == 1, "first ID")
    source[:] = 999  # caller mutation cannot reach admitted data
    require(runtime.request_read("position", ids)["sequence"] == 2, "read ID")
    require(
        runtime.submit_rows("position", ids, np.array([[5], [6]], np.float32))["sequence"] == 3,
        "update ID",
    )
    runtime.request_read("position", ids)  # pending at bounded end
    s = runtime.run(3, 1)
    outcomes = drain(runtime)
    require([o["sequence"] for o in outcomes] == [1, 2, 3, 4], "no lost/reordered IDs")
    require([o["frame"] for o in outcomes] == [0, 1, 2, 3], "one batch per boundary")
    require(outcomes[-1]["status"] == "cancelled", "pending work cancelled")
    require(np.array_equal(values(outcomes[1]), [101, 201]), "FIFO read before integration")
    saved = outcomes[1]["data"]
    require(
        (s["frames"], s["native_steps"], s["rows_integrated"]) == (3, 3, 6), "native work counts"
    )
    rejected(lambda: runtime.run(1, 1), "single-shot")
    require(runtime.request_read("position", ids)["status"] == "stopped", "stopped admission")
    runtime.close()
    require(
        np.array_equal(values(w.read_component("position", ids)), [6, 7]), "same World returned"
    )
    require(np.array_equal(np.frombuffer(saved, "<f4"), [101, 201]), "snapshot ownership")

    # A stale identity mixed with a valid one must not partially mutate row 0.
    runtime = native.NativeRuntime(w, "position", "velocity", capacity=4, batches_per_frame=4)
    stale = ids.copy()
    stale[1, 1] += 1
    runtime.submit_rows("position", stale, np.full((2, 1), 900, np.float32))
    runtime.request_read("position", ids)
    runtime.submit_rows("position", ids, np.array([[10], [20]], np.float32))
    s = runtime.run(1, 1)
    outcomes = drain(runtime)
    require(outcomes[0]["status"] == "rejected", "stale rejected")
    require(np.array_equal(values(outcomes[1]), [6, 7]), "stale atomicity")
    require((s["rejected"], s["applied"]) == (1, 2) and s["first_error"], "first error visible")
    runtime.close()
    require(
        np.array_equal(values(w.read_component("position", ids)), [11, 21]), "later valid applied"
    )

    runtime = native.NativeRuntime(w, "position", "velocity", capacity=1)
    runtime.request_read("position", ids)
    require(runtime.request_read("position", ids)["status"] == "full", "queue full")
    runtime.stop()  # must work even with every credit occupied
    require(runtime.phase == "stopped", "stop before run")
    require(drain(runtime)[0]["status"] == "cancelled", "stop cancels pending")
    runtime.close()
    runtime = native.NativeRuntime(w, "position", "velocity", capacity=1)
    runtime.request_read("position", ids)
    del runtime
    gc.collect()
    require(w.alive_count == 2, "drop restores World without worker")

    # Scalar-byte bound applies before acquiring the large values buffer.
    wide, wide_ids = world(5000, width=256)
    runtime = native.NativeRuntime(wide, "position", "velocity", max_rows=5000)
    rejected(lambda: runtime.request_read("position", wide_ids), "limit")

    class NeverRead:
        def __buffer__(self, flags):
            raise AssertionError("oversized payload buffer was acquired")

    rejected(lambda: runtime.submit_rows("position", wide_ids, NeverRead()), "limit")
    runtime.close()

    # Bounds and copy must use the same acquisition; arbitrary exporters can
    # return different buffers on repeated calls.
    w, ids = world(1)
    runtime = native.NativeRuntime(w, "position", "velocity")

    class OneExport:
        calls = 0

        def __buffer__(self, flags):
            self.calls += 1
            require(self.calls == 1, "handles acquired twice before bounded copy")
            return memoryview(ids)

    once = OneExport()
    require(
        runtime.request_read("position", once)["status"] == "admitted",
        "single acquisition admission",
    )
    require(once.calls == 1, "one handle acquisition")
    runtime.close()

    # A buffer exporter reenters while a World update is preparing input,
    # transferring ownership before commit. Revalidated commit must reject.
    w, ids = world(1)
    owned = []

    class Transfer:
        def __buffer__(self, flags):
            owned.append(native.NativeRuntime(w, "position", "velocity"))
            return memoryview(np.array([[44]], np.float32))

    rejected(lambda: w.update_rows("position", ids, Transfer()), "owned")
    owned.pop().close()
    require(
        values(w.read_component("position", ids))[0] == 0, "no mutation after ownership transfer"
    )
    print(
        "deterministic native gate: FIFO, atomic stale rejection, ownership, full, stop, pending, drop and byte bounds passed"
    )


def paced(count, seconds):
    w, ids = world(count)
    runtime = native.NativeRuntime(w, "position", "velocity", capacity=8, batches_per_frame=2)
    samples, submitted, failures = [], [], []
    original_thread = threading.get_ident()

    async def producer():
        deadline = time.monotonic() + 5
        while runtime.phase != "running":
            require(time.monotonic() < deadline, "run never started")
            await asyncio.sleep(0.001)
        rejected(runtime.close, "running")
        rejected(lambda: runtime.run(1, 1), "running")
        rejected(w.snapshot, "owned")
        for value, stall in [(1000, "async sleep"), (2000, "Python busy")]:
            result = runtime.submit_rows("position", ids[:1], np.array([[value]], np.float32))
            require(result["status"] == "admitted", str(result))
            submitted.append((result["sequence"], value))
            runtime.request_read("position", ids[:1])
            before = runtime.stats()["frames"]
            start = time.monotonic()
            if stall == "async sleep":
                await asyncio.sleep(seconds / 2)
            else:
                until = start + seconds / 2
                while time.monotonic() < until:
                    pass
            after = runtime.stats()["frames"]
            require(after > before, f"native frames stopped during {stall}")
            samples.append((stall, time.monotonic() - start, before, after))
        # Exhaust credits, race submission with stop, without requiring a slot
        # for stop. Some commands may apply before stop; all IDs still complete.
        while (result := runtime.request_read("position", ids[:1]))["status"] == "admitted":
            pass
        require(result["status"] == "full", "outcome credits bound memory")
        runtime.stop()
        require(
            runtime.request_read("position", ids[:1])["status"] == "stopped", "post-stop admission"
        )

    def own_thread():
        try:
            asyncio.run(producer())
        except BaseException as error:
            failures.append(error)
            runtime.stop()

    thread = threading.Thread(target=own_thread, name="native-command-producer")
    thread.start()
    try:
        s = runtime.run(max(600, int(seconds * 240)), 1, pace_hz=120)
    finally:
        runtime.stop()
        thread.join(timeout=6)
    require(not thread.is_alive(), "producer leaked")
    if failures:
        raise failures[0]
    require(threading.get_ident() == original_thread, "loop changed caller thread")
    outcomes = drain(runtime)
    require(
        [o["sequence"] for o in outcomes] == list(range(1, s["admitted"] + 1)),
        "lost/reordered completion",
    )
    require(
        s["admitted"] == s["applied"] + s["rejected"] + s["cancelled"], "all admitted work disposed"
    )
    require(s["rejected"] == 0 and s["backlog"] == 0, "no unexpected rejection/backlog")
    for sequence, value in submitted:
        update, read = outcomes[sequence - 1 : sequence + 1]
        require(update["status"] == read["status"] == "applied", "updates and snapshots applied")
        require(values(read)[0] == value + read["frame"] - update["frame"], "exact FIFO read state")
    last_update = outcomes[submitted[-1][0] - 1]
    runtime.close()
    final = values(w.read_component("position", ids[[0, -1]]))
    require(
        np.array_equal(final, [2000 + s["frames"] - last_update["frame"], s["frames"]]),
        "exact final state",
    )
    require(s["rows_integrated"] == s["frames"] * count, "all entities natively integrated")
    print("| producer stall | seconds | frames before | frames after | advanced |")
    print("|---|---:|---:|---:|---:|")
    for name, elapsed, before, after in samples:
        print(f"| {name} | {elapsed:.3f} | {before} | {after} | {after - before} |")
    print(
        "| entities | frames/steps | admitted/applied/rejected/cancelled | backlog/full | native ms | frame work ms | paced ms | wall ms | max frame work ms | in/readback bytes |"
    )
    print("|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    print(
        f"| {count} | {s['frames']}/{s['native_steps']} | {s['admitted']}/{s['applied']}/{s['rejected']}/{s['cancelled']} | {s['backlog']}/{s['full']} | "
        f"{s['native_ns'] / 1e6:.3f} | {s['frame_ns'] / 1e6:.3f} | {s['paced_ns'] / 1e6:.3f} | {s['wall_ns'] / 1e6:.3f} | {s['max_frame_ns'] / 1e6:.3f} | {s['bytes_in']}/{s['readback_bytes']} |"
    )
    print(
        f"exact final selected positions: {final.tolist()}; producer joined; calling-thread native run"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--entities", type=int, default=10000)
    parser.add_argument("--seconds", type=float, default=2)
    parser.add_argument("--deterministic-only", action="store_true")
    parser.add_argument(
        "--control-drop-first",
        action="store_true",
        help="negative control: omit first native submission; must fail",
    )
    args = parser.parse_args()
    require(
        any(native.__file__.endswith(s) for s in importlib.machinery.EXTENSION_SUFFIXES),
        "compiled extension required",
    )
    require(2 <= args.entities <= 100000 and 0.2 <= args.seconds <= 5, "bounded sample arguments")
    deterministic(args.control_drop_first)
    if not args.deterministic_only:
        paced(args.entities, args.seconds)


if __name__ == "__main__":
    main()
