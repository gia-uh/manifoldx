#!/usr/bin/env python3
"""Exercise the installed private native component World.

Run this only from the native verifier's disposable cwd, with `-I`, after the
wheel is installed. It is both the runnable example and the installed-wheel
gate: it asserts the compiled-extension provenance, the native schema and
transaction contract, then prints the measured bulk-transfer table.

The example mirrors the intended first consumer: NumPy prepares positions and
mesh ids once, a synchronous setup uploads a native world, then bulk commands
update, query, read back, destroy and reuse entities. No Python reference
backend, parity adapter or graphics stack is involved.

The boundary is explicit: each submitted NumPy array is copied once into typed
Rust storage; each readback is an owned copy encoded as little-endian bytes.
Nothing here claims zero-copy or a frame rate.
"""

import importlib.machinery
import importlib.util
import sys
import time

import numpy as np

checks = 0


def check(condition, label):
    global checks
    if not condition:
        raise AssertionError(label)
    checks += 1
    print(f"ok {checks}  {label}")


def expect(error_type, call, contains, label):
    global checks
    try:
        call()
    except error_type as error:
        check(contains in str(error), f"{label}: {error}")
    else:
        raise AssertionError(f"{label}: no {error_type.__name__}")


def timed(function):
    start = time.perf_counter()
    value = function()
    return (time.perf_counter() - start) * 1000.0, value


def handles_from(result):
    handles = np.frombuffer(result["handles"], dtype="<u8").reshape(-1, 2)
    check(handles.shape == (result["count"], 2), f"owned handles shape {handles.shape}")
    return handles


def provenance():
    import manifoldx

    spec = importlib.util.find_spec("manifoldx._core_native")
    check(
        spec is not None and isinstance(spec.loader, importlib.machinery.ExtensionFileLoader),
        f"manifoldx._core_native is a compiled extension: {getattr(spec, 'origin', None)}",
    )
    check(spec.origin.startswith(sys.prefix), f"extension origin is installed: {spec.origin}")
    check(manifoldx.__file__.startswith(sys.prefix), f"package origin is installed: {manifoldx.__file__}")
    import manifoldx._core_native as native

    world = native._NativeWorld
    check(world.__module__ == "manifoldx._core_native", f"native class provenance: {world.__module__}")
    return native._NativeWorld


def make_world(world_class, capacity):
    world = world_class(capacity)
    world.register_component("transform.position", "f32", 3)
    world.register_component("mesh.id", "u32", 1)
    world.register_component("material.id", "u32", 1)
    world.register_component("custom.velocity", "f32", 3)
    world.register_component("flags.marker", "u32", 0)
    return world


def prepare(count):
    rng = np.random.default_rng(1234)
    return {
        "transform.position": rng.random((count, 3), dtype=np.float32),
        "mesh.id": np.arange(count, dtype=np.uint32).reshape(-1, 1),
        "material.id": np.full((count, 1), 1, dtype=np.uint32),
        "custom.velocity": np.zeros((count, 3), dtype=np.float32),
        "flags.marker": None,
    }


def schema_contract(world_class):
    world = make_world(world_class, 8)
    before = world.snapshot()
    expect(
        ValueError,
        lambda: world.register_component("mesh.id", "u32", 1),
        "already registered",
        "duplicate schema rejected",
    )
    check(world.snapshot() == before, "duplicate schema leaves live state unchanged")
    expect(
        ValueError,
        lambda: world.register_component("bad", "f64", 1),
        "unsupported component element type",
        "unsupported element type rejected",
    )
    ids = handles_from(world.spawn(1, {}))
    expect(
        ValueError,
        lambda: world.register_component("late", "f32", 1),
        "after entities are published",
        "schema registration locks after the first publish",
    )
    check(world.alive_count == 1 and ids.shape == (1, 2), "one entity is live after the lock check")


def world_roundtrip(world_class):
    world = make_world(world_class, 32)
    payload = prepare(3)
    payload.pop("custom.velocity")
    expected = {name: value.copy() for name, value in payload.items() if value is not None}

    elapsed, spawned = timed(lambda: world.spawn(3, payload))
    check(elapsed >= 0.0, "spawn timed")
    ids = handles_from(spawned).copy()
    check(spawned["bytes_in"] == sum(value.nbytes for value in expected.values()), "spawn bytes counted")
    check(world.alive_count == 3, "three entities live")

    # Defect control: mutate the caller's arrays after submission. A binding
    # that retained a pointer would show -1 here.
    for value in payload.values():
        if value is not None:
            value.fill(0)
    position = world.read_component("transform.position", ids)
    values = np.frombuffer(position["data"], dtype="<f4").reshape(3, 3)
    check(np.array_equal(values, expected["transform.position"]), "caller mutation cannot alias native rows")
    membership = np.frombuffer(position["membership"], dtype=np.uint8)
    check(np.array_equal(membership, np.ones(3, dtype=np.uint8)), "position membership is true")
    marker = world.read_component("flags.marker", ids)
    check(len(marker["data"]) == 0, "zero-width marker has no data bytes")
    check(
        np.array_equal(np.frombuffer(marker["membership"], dtype=np.uint8), np.ones(3, dtype=np.uint8)),
        "zero-width marker membership is true",
    )

    # A component not in the payload is a non-member with a zeroed owned row.
    velocity = world.read_component("custom.velocity", ids)
    check(
        np.array_equal(np.frombuffer(velocity["membership"], dtype=np.uint8), np.zeros(3, dtype=np.uint8)),
        "omitted component is a non-member",
    )
    check(np.all(np.frombuffer(velocity["data"], dtype="<f4") == 0.0), "omitted component row is zeroed")

    # Whole-row update only touches selected members.
    updated = expected["transform.position"][[2, 0]] + np.array([10.0, 20.0, 30.0], dtype=np.float32)
    result = world.update_rows("transform.position", ids[[2, 0]], updated)
    check(result["rows"] == 2 and result["bytes_in"] == updated.nbytes, "row update counted")
    after = np.frombuffer(
        world.read_component("transform.position", ids)["data"], dtype="<f4"
    ).reshape(3, 3)
    check(np.array_equal(after[[2, 0]], updated), "row update wrote selected rows")
    check(np.array_equal(after[1], expected["transform.position"][1]), "row update left an unselected row alone")

    # One field update.
    material = np.array([7], dtype=np.uint32)
    result = world.update_field("material.id", 0, ids[[1]], material)
    check(result["rows"] == 1 and result["bytes_in"] == material.nbytes, "field update counted")
    material_after = np.frombuffer(
        world.read_component("material.id", ids)["data"], dtype="<u4"
    ).reshape(3, 1)
    check(material_after[1, 0] == 7 and material_after[0, 0] == 1, "field update changed one member")

    # Malformed batches: a stale identity and a wrong length reject the whole
    # batch before any value changes. This is the batch-validation control.
    state_before = world.snapshot()
    rows_before = world.read_component("transform.position", ids)
    stale = ids.copy()
    stale[0, 1] += 1
    expect(
        RuntimeError,
        lambda: world.update_rows("transform.position", stale[[0, 1]], np.zeros((2, 3), dtype=np.float32)),
        "stale entity slot",
        "stale batch rejected",
    )
    check(world.snapshot() == state_before, "stale batch left the world version unchanged")
    check(
        world.read_component("transform.position", ids)["data"] == rows_before["data"],
        "stale batch left every row unchanged",
    )
    expect(
        ValueError,
        lambda: world.update_rows("transform.position", ids, np.zeros((2, 3), dtype=np.float32)),
        "must have shape",
        "wrong row count rejected",
    )
    check(world.snapshot() == state_before, "malformed length left the world unchanged")
    expect(
        ValueError,
        lambda: world.update_rows("custom.velocity", ids, np.zeros((3, 3), dtype=np.float32)),
        "has no",
        "update of a non-member rejected",
    )
    check(world.snapshot() == state_before, "non-member update left the world unchanged")

    # Destroy and reuse: membership and values reset, the old generation is
    # stale for every later command.
    result = world.destroy(ids[[0]])
    check(result["entities"] == 1 and result["bytes_in"] == 16, "destroy counted")
    reused = handles_from(world.spawn(1, {"transform.position": np.full((1, 3), 9.0, dtype=np.float32)}))
    check(reused[0, 0] == ids[0, 0], "destroyed slot is reused")
    check(reused[0, 1] > ids[0, 1], "reused slot carries a new generation")
    check(
        np.frombuffer(world.has("mesh.id", reused)["flags"], dtype=np.uint8)[0] == 0,
        "reused slot has no old membership",
    )
    check(
        np.all(np.frombuffer(world.read_component("mesh.id", reused)["data"], dtype="<u4") == 0),
        "reused slot has zeroed default values",
    )
    expect(
        RuntimeError,
        lambda: world.destroy(ids[[0]]),
        "stale entity slot",
        "stale destroy rejected",
    )


def bulk_example(world_class, count):
    payload = prepare(count)
    world = make_world(world_class, count + 16)
    expected_positions = payload["transform.position"].copy()

    spawn_ms, spawned = timed(lambda: world.spawn(count, payload))
    handles = handles_from(spawned).copy()
    check(world.alive_count == count, f"{count} entities live")

    has_ms, flags = timed(lambda: world.has("transform.position", handles))
    check(np.all(np.frombuffer(flags["flags"], dtype=np.uint8) == 1), "bulk membership query is true")

    read_ms, read = timed(lambda: world.read_component("transform.position", handles))
    values = np.frombuffer(read["data"], dtype="<f4").reshape(count, 3)
    check(np.array_equal(values, expected_positions), "bulk readback matches the uploaded positions")

    # A substantial bulk row update and one field update.
    rows = min(5_000, count)
    updated = (expected_positions[:rows] + np.array([0.0, 1.0, 0.0], dtype=np.float32)).astype(
        np.float32, copy=False
    )
    rows_ms, rows_result = timed(lambda: world.update_rows("transform.position", handles[:rows], updated))
    check(rows_result["rows"] == rows, "bulk row update reports its row count")

    fields = min(2_000, count)
    materials = np.full(fields, 7, dtype=np.uint32)
    field_ms, field_result = timed(lambda: world.update_field("material.id", 0, handles[2000 : 2000 + fields], materials))
    check(field_result["rows"] == fields, "bulk field update reports its row count")

    after_read_ms, after = timed(lambda: world.read_component("transform.position", handles[:3]))
    after_values = np.frombuffer(after["data"], dtype="<f4").reshape(3, 3)

    # Destroy and reuse a subset.
    destroy_slots = np.arange(0, count, max(1, count // 1_000))[:1_000]
    destroy_handles = handles[destroy_slots]
    destroy_ms, destroyed = timed(lambda: world.destroy(destroy_handles))
    check(destroyed["entities"] == len(destroy_slots), "destroy reports the subset size")
    reuse_ms, reused = timed(
        lambda: world.spawn(len(destroy_slots), {"transform.position": np.full((len(destroy_slots), 3), 42.0, dtype=np.float32)})
    )
    reused_handles = handles_from(reused).copy()
    check(set(reused_handles[:, 0]) == set(destroy_handles[:, 0]), "reuse consumed the destroyed slots")
    old_generations = {int(slot): int(generation) for slot, generation in destroy_handles}
    check(
        all(old_generations[int(slot)] < int(generation) for slot, generation in reused_handles),
        "every reused slot has a newer generation",
    )
    check(
        np.all(np.frombuffer(world.has("mesh.id", reused_handles)["flags"], dtype=np.uint8) == 0),
        "reused entities have no old mesh membership",
    )

    return {
        "count": count,
        "spawn_ms": spawn_ms,
        "has_ms": has_ms,
        "read_ms": read_ms,
        "rows_ms": rows_ms,
        "rows": rows_result["rows"],
        "field_ms": field_ms,
        "fields": field_result["rows"],
        "after_read_ms": after_read_ms,
        "destroy_ms": destroy_ms,
        "destroyed": destroyed["entities"],
        "reuse_ms": reuse_ms,
        "bytes_in": spawned["bytes_in"]
        + rows_result["bytes_in"]
        + field_result["bytes_in"]
        + destroyed["bytes_in"],
        "bytes_out": spawned["bytes_out"]
        + flags["bytes_out"]
        + read["bytes_out"]
        + after["bytes_out"],
        "before": expected_positions[:3],
        "after": after_values,
    }


def overflow_contract(world_class):
    world = make_world(world_class, 4)
    world.spawn(4, {})
    before = world.snapshot()
    expect(
        RuntimeError,
        lambda: world.spawn(1, {}),
        "Max entities 4",
        "over-capacity spawn rejected",
    )
    check(world.snapshot() == before, "over-capacity spawn left the world unchanged")
    expect(
        RuntimeError,
        lambda: world.spawn(2**32 - 1, {}),
        "Max entities 4",
        "extreme spawn count rejected without allocation",
    )
    check(world.snapshot() == before, "extreme spawn count left the world unchanged")


def main():
    world_class = provenance()
    schema_contract(world_class)
    world_roundtrip(world_class)
    overflow_contract(world_class)

    print("\nBulk boundary example")
    print(
        "| entities | spawn ms | membership ms | readback ms | rows ms | field ms | destroy ms | reuse ms |"
    )
    print("|---:|---:|---:|---:|---:|---:|---:|---:|")
    runs = []
    for count in (10_000, 100_000):
        run = bulk_example(world_class, count)
        runs.append(run)
        print(
            f"| {run['count']:,} | {run['spawn_ms']:.2f} | {run['has_ms']:.2f} | "
            f"{run['read_ms']:.2f} | {run['rows_ms']:.2f} | {run['field_ms']:.2f} | "
            f"{run['destroy_ms']:.2f} | {run['reuse_ms']:.2f} |"
        )

    print("\nBoundary transfers (sum over both worlds)")
    total_in = sum(run["bytes_in"] for run in runs)
    total_out = sum(run["bytes_out"] for run in runs)
    print(f"- input arrays copied once at submission: {total_in:,} bytes")
    print(f"- owned readback bytes returned: {total_out:,} bytes")
    print("- copies: 1 per submitted array in, 1 owned copy per readback out; no zero-copy claim")
    print("- synchronous setup/command boundary; no frame-rate claim")

    run = runs[-1]
    print(f"\nObserved before/after rows (100k world, first three entities)")
    print("| entity | before position | after position |")
    print("|---:|---|---|")
    for row in range(3):
        before = run["before"][row]
        after = run["after"][row]
        print(
            f"| {row} | ({before[0]:.4f}, {before[1]:.4f}, {before[2]:.4f}) | "
            f"({after[0]:.4f}, {after[1]:.4f}, {after[2]:.4f}) |"
        )
    print(f"\n{checks} checks passed")


if __name__ == "__main__":
    main()
