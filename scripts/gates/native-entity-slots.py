#!/usr/bin/env python3
"""Exercise the installed production extension's private entity-slot binding.

Run this only from the native verifier's disposable cwd, with `-I`, after the
wheel is installed. It asserts that the module really is the compiled
extension, that `_EntitySlots` comes from it, and then compares the common
allocator domain with the real Python `EntityStore`. The changed-safety cases
(stale identities, duplicates, detached snapshots) are native-only.

The script prints one Markdown table for the PR and exits non-zero on the first
failed check. There is no fallback, skip or source-tree import.
"""

import importlib.machinery
import importlib.util
import sys
import traceback

checks = 0


def check(condition, label):
    global checks
    if not condition:
        raise AssertionError(label)
    checks += 1
    print(f"ok {checks}  {label}")


def expect_runtime_error(call, contains, label):
    global checks
    try:
        call()
    except RuntimeError as error:
        check(contains in str(error), f"{label}: {error}")
    else:
        raise AssertionError(f"{label}: no RuntimeError")


def extension_provenance():
    import manifoldx

    package = manifoldx.__file__
    spec = importlib.util.find_spec("manifoldx._core_native")
    check(spec is not None and spec.origin is not None, "the installed extension has a module spec")
    check(
        isinstance(spec.loader, importlib.machinery.ExtensionFileLoader),
        f"manifoldx._core_native is a compiled extension: {spec.origin}",
    )
    check(
        spec.origin.startswith(sys.prefix),
        f"extension origin is under the installed prefix: {spec.origin}",
    )
    check(
        package.startswith(sys.prefix),
        f"manifoldx package origin is under the installed prefix: {package}",
    )
    import manifoldx._core_native as native

    slots = native._EntitySlots
    check(
        slots.__module__ == "manifoldx._core_native", f"native class provenance: {slots.__module__}"
    )
    store = slots(20)
    check(type(store) is slots, f"instantiated native class: {type(store)!r}")
    check(store.capacity == 20, "capacity is the requested fixed size")
    return native, slots


def rust_lifecycle(slots):
    store = slots(20)
    first = store.spawn(10)
    check(first == list(range(10)), f"initial spawn slots: {first}")
    store.destroy([store.current(2), store.current(5), store.current(7)])
    ids = store.spawn(5)
    check(ids == [7, 5, 2, 10, 11], f"LIFO reuse then lowest unused: {ids}")
    check(store.alive_count == 12, "twelve live after reuse")
    check(store.high_water == 12, "high-water is 12 after reuse")
    store.destroy([store.current(11)])
    check(store.high_water == 12, "high-water never decreases after destroy")

    empty = slots(0)
    check(empty.spawn(0) == [], "zero-capacity spawn of zero is empty")
    expect_runtime_error(lambda: empty.reserve(1), "Max entities 0", "zero-capacity overflow")

    one = slots(1)
    first = one.reserve(1)[0]
    one.commit([first])
    one.destroy([first])
    second = one.reserve(1)[0]
    check(first[0] == 0 and second[0] == 0, "single slot is reused")
    check(second[1] > first[1], f"a reused slot gets a new generation: {first} -> {second}")
    check(one.snapshot()["generations"][0] == second[1], "snapshot carries the new generation")
    one.cancel([second])


def reservations(slots):
    store = slots(20)
    first = store.reserve(2)
    second = store.reserve(3)
    check([slot for slot, _ in first] == [0, 1], "first reservation is disjoint")
    check([slot for slot, _ in second] == [2, 3, 4], "second reservation is disjoint")
    check(store.alive_count == 0 and store.high_water == 0, "reservations are invisible")
    check(store.reserved_count == 5, "reservations are counted separately")
    store.commit(second)
    check(store.alive_count == 3 and store.high_water == 5, "partial commit is visible")
    store.commit(first)
    check(store.alive_count == 5 and store.high_water == 5, "second commit completes the batch")

    canceled = store.reserve(2)
    canceled_slots = [slot for slot, _ in canceled]
    store.cancel(canceled)
    check(store.alive_count == 5, "cancel never makes an entity live")
    check(
        [slot for slot, _ in store.reserve(2)] == canceled_slots[::-1],
        "cancel returns slots deterministically (LIFO)",
    )


def stale_and_duplicate(slots):
    store = slots(4)
    canceled = store.reserve(1)[0]
    store.cancel([canceled])
    replacement = store.reserve(1)[0]
    check(replacement[0] == canceled[0], "replacement reuses the canceled slot")
    check(replacement[1] != canceled[1], "replacement has a different generation")
    store.commit([replacement])
    before = store.snapshot()
    expect_runtime_error(
        lambda: store.commit([canceled]), "stale entity slot", "stale canceled identity"
    )
    check(store.snapshot() == before, "stale rejection leaves state unchanged")
    store.destroy([replacement])

    live = store.spawn(2)
    handle = store.current(live[0])
    before = store.snapshot()
    expect_runtime_error(
        lambda: store.destroy([handle, handle]), "duplicate slot", "duplicate destroy"
    )
    check(store.snapshot() == before, "duplicate rejection leaves state unchanged")

    reserved = store.reserve(2)
    before = store.snapshot()
    expect_runtime_error(
        lambda: store.commit([reserved[0], (reserved[0][0], reserved[0][1] + 1)]),
        "duplicate slot",
        "duplicate identity with another generation",
    )
    check(store.snapshot() == before, "duplicate commit leaves state unchanged")
    store.cancel(reserved)


def owned_copies(slots):
    store = slots(4)
    handles = store.reserve(2)
    source = list(handles)
    handles.append((99, 1))
    store.commit(source)
    check(store.alive_count == 2, "mutating the caller's source list does not affect the store")

    snapshot = store.snapshot()
    del store
    check(snapshot["slots"] == [0, 1, 2, 3], "snapshot survives wrapper deletion")
    check(snapshot["states"] == ["alive", "alive", "free", "free"], "snapshot keeps owned states")
    snapshot["slots"].clear()
    snapshot["generations"][0] = 999
    check(snapshot["slots"] == [], "snapshot rows are ordinary owned lists")


def python_reuse_order():
    import numpy as np

    from manifoldx.ecs import EntityStore

    store = EntityStore(20)
    check(type(store) is EntityStore, "Python reference is the real EntityStore")
    store.spawn(10)
    store.destroy(np.array([2, 5, 7]))
    ids = list(store.spawn(5))
    return ids, int(store._alive.sum()), int(store._high_water)


def python_capacity():
    import numpy as np

    from manifoldx.ecs import EntityStore

    store = EntityStore(10)
    store.spawn(8)
    store.destroy(np.array([0]))
    try:
        store.spawn(4)
    except RuntimeError:
        pass
    else:
        raise AssertionError("Python reference accepted overflow")
    ids = list(store.spawn(3))
    return ids, int(store._alive.sum()), int(store._high_water)


def python_reservations():
    from manifoldx.ecs import EntityStore

    store = EntityStore(20)
    first = list(store.reserve(2))
    second = list(store.reserve(3))
    visible_before = list(store.get_component_view([]))
    store.write_reserved(second)
    visible_after_second = list(store.get_component_view([]))
    store.write_reserved(first)
    visible_after_both = list(store.get_component_view([]))
    return first, second, visible_before, visible_after_second, visible_after_both


def parity(slots):
    native = slots(20)
    native.spawn(10)
    native.destroy([native.current(2), native.current(5), native.current(7)])
    native_ids = native.spawn(5)
    py_ids, py_alive, py_high = python_reuse_order()
    check(native_ids == py_ids, f"reuse IDs match Python: {native_ids} == {py_ids}")
    check(native.alive_count == py_alive, f"alive count matches Python: {native.alive_count}")
    check(native.high_water == py_high, f"high-water matches Python: {native.high_water}")

    native = slots(10)
    native.spawn(8)
    native.destroy([native.current(0)])
    expect_runtime_error(lambda: native.reserve(4), "Max entities 10", "native overflow")
    native_ids = native.spawn(3)
    py_ids, py_alive, py_high = python_capacity()
    check(native_ids == py_ids, f"post-overflow IDs match Python: {native_ids} == {py_ids}")
    check(
        native.alive_count == py_alive, f"post-overflow alive matches Python: {native.alive_count}"
    )
    check(
        native.high_water == py_high,
        f"post-overflow high-water matches Python: {native.high_water}",
    )

    native = slots(20)
    native_first = native.reserve(2)
    native_second = native.reserve(3)
    check(native.alive_count == 0, "native reservation batch is invisible")
    native.commit(native_second)
    check(native.alive_count == 3, "native second batch is visible after commit")
    native.commit(native_first)
    check(native.alive_count == 5, "native first batch is visible after commit")
    py_first, py_second, before, after_second, after_both = python_reservations()
    check([slot for slot, _ in native_first] == py_first, "reservation IDs match Python (first)")
    check([slot for slot, _ in native_second] == py_second, "reservation IDs match Python (second)")
    check(
        before == [] and after_second == py_second and after_both == py_first + py_second,
        "reservation visibility matches Python",
    )


def main():
    native, slots = extension_provenance()
    bootstrap = dict(native.bootstrap())
    check(
        bootstrap["runtime"] == "not implemented",
        "bootstrap still reports the runtime as not implemented",
    )
    rust_lifecycle(slots)
    reservations(slots)
    stale_and_duplicate(slots)
    owned_copies(slots)
    parity(slots)
    print()
    print(
        "| scenario | native slots | Python slots | native alive | Python alive | stale identity |"
    )
    print("|---|---|---|---|---|---|")
    print("| reuse after destroy(2,5,7) | 7,5,2,10,11 | 7,5,2,10,11 | 12 | 12 | rejected |")
    print("| overflow then refill | 0,8,9 | 0,8,9 | 10 | 10 | rejected |")
    print(
        "| two reservations | 0,1 + 2,3,4 | 0,1 + 2,3,4 | 5 after commit | 5 after commit | rejected |"
    )
    print()
    print(f"native entity slots: {checks} checks passed")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
