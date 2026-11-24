# Native entity slots (phase 1a)

This is the developer note for the first native ECS piece: a fixed-capacity
entity slot allocator in `crates/manifoldx-core/src/ecs/`. It is exposed only
through the private `manifoldx._core_native._EntitySlots` binding so an
installed wheel can exercise real native ownership. It is not the public ECS,
has no components, queries, pointers or NumPy arrays, and does not activate the
Engine runtime.

## Contract

These are decisions for this private slice, not a change to the Python public
API or a claim that the current Python `EntityStore` behaves this way.

- **Fixed capacity.** The slot table is allocated once at construction and never
  grows.
- **Allocation order.** `reserve` consumes free slots LIFO (most recently freed
  first), then the lowest never-used slots in ascending order. Reserved slots
  are disjoint and invisible: neither `alive_count` nor a snapshot state says
  alive until `commit`. The free list is a stack: a reservation validates one
  contiguous suffix and truncates it after validation, so its cost is linear in
  the slots it consumes rather than products of free and requested counts.
- **Commit.** A successful `commit` makes the reserved identities live and
  raises the high-water mark to one past the highest committed slot.
- **Cancel and destroy.** `cancel` returns a reservation to the free set without
  making it live. `destroy` does the same for a live identity. Both insert the
  batch in ascending slot order, so LIFO reuse is deterministic. High-water
  never decreases.
- **Whole-batch validation.** Overflow, a stale identity, a wrong lifecycle
  state, an out-of-range slot or a duplicate slot rejects the complete batch
  before any slot, free-order, high-water, generation or version changes. An
  impossible `reserve` scans the fixed table instead of allocating its own
  requested length first, so an untrusted huge count cannot exhaust memory.
- **Identity.** The exposed ID is the slot integer; the allocator also carries a
  `u64` generation per slot. `reserve` returns `(slot, generation)` handles for
  exact-identity operations, and `spawn` returns slot integers.
- **No aliasing.** Each reservation increments the slot generation. A recycled
  or canceled identity never matches its replacement. A stale identity is
  rejected.
- **No wrap.** A slot whose next generation would be past `u64::MAX` is retired
  instead. No handle is ever reissued with the same generation.
- **Owned snapshots.** `snapshot()` and every returned list/tuple are copies. The
  binding retains no caller pointer, and mutating a snapshot cannot affect the
  store.
- **Serialized boundary.** The PyO3 class holds one `Mutex`, so mutations are
  serialized at the binding boundary on free-threaded Python.

This is stronger native lifetime validation than the current Python raw-slot
behavior. It does not establish parity for stale raw slots and does not change
`EntityHandle` semantics.

## API

Rust (`manifoldx_core::ecs`): `EntitySlots::new(capacity)`, `reserve`, `commit`,
`cancel`, `destroy`, `spawn`, `current_identity`, `snapshot`, plus `EntityId`,
`SlotState`, `Snapshot` and `EntityError`.

Python (installed extension only):

```python
from manifoldx._core_native import _EntitySlots

store = _EntitySlots(20)
slots = store.spawn(10)                  # [0, ..., 9]
store.destroy([store.current(2)])        # exact (slot, generation)
handles = store.reserve(2)               # [(slot, generation), ...]
store.commit(handles)
snapshot = store.snapshot()              # owned dict of lists
```

`scripts/gates/native-entity-slots.py` is the runnable installed-wheel example
and focused gate. It asserts the compiled-extension origin, the native class
provenance, the Rust lifecycle, the changed-safety cases and the common
allocator domain against the real Python `EntityStore`.

## Limits

No growth, components, queries, facade, pointers, NumPy arrays, renderer,
scheduler, Engine activation or SDK redesign. The bootstrap still reports
`runtime: not implemented`, and `MANIFOLDX_CORE=rust` still refuses to construct
an Engine. Phase 1a does not close phase 1; component storage is the next
private slice, described in [`component-store.md`](component-store.md).
