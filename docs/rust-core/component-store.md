# Native component World (phase 1b)

This is the developer note for the first native component store:
`crates/manifoldx-core/src/ecs/world.rs`, built on the generation-bearing
entity slots from phase 1a. It is exposed only through the private
`manifoldx._core_native._NativeWorld` binding so an installed wheel can exercise
real native ownership. It is not the public ECS, has no scheduler, renderer or
IR interpreter, and does not enable the Engine Rust path.

The supported native API is deliberately small: explicit schemas over `f32`
and `u32` rows, bulk spawn/update/destroy, membership queries and owned
readback. It is not a promise of arbitrary NumPy dtype parity, a Python
reference backend or writable native aliases. Later element types are added for
actual consumers, one at a time.

## Contract

- **One allocator.** The World owns one `EntitySlots` table. There is no second
  entity allocator, and every mutation keeps its generation-bearing identity.
- **Explicit, frozen schema.** `register_component(name, kind, width)` registers
  a unique name with element kind `"f32"` or `"u32"` and a fixed width in
  scalars. Width `0` is a marker. Duplicate names, empty names, widths above
  `MAX_COMPONENT_WIDTH` and registration after the first published entity are
  rejected; a rejection does not reset or change live state.
- **Fixed capacity and fixed rows.** The slot table and every column are
  allocated once for the declared capacity. Rows never change width.
- **Membership is explicit.** Every live entity has a zeroed row in every
  registered column. A membership bit says whether the component is attached.
  A non-member row is zeroed and is not a place to read stale values.
- **Spawn initialises the row set.** For each new entity every column starts as
  non-member with a zeroed row; each payload component then receives the
  submitted values and becomes a member. A recycled slot never inherits an
  earlier life's values or membership.
- **Atomic batches.** A malformed descriptor, wrong kind, wrong length,
  duplicate component or identity, stale generation, non-member update or
  over-capacity spawn rejects the whole batch before any entity, membership or
  value changes. Validation is linear in the submitted batch, not in
  `capacity × batch`; no transaction clones the World.
- **Updates require membership.** Row and field updates never attach a
  component implicitly.
- **Destroy clears.** A successful destroy marks the entities dead and zeroes
  every column's row and membership for those slots. A stale destroy is
  rejected by the slot allocator and changes nothing.
- **Owned readback.** Snapshots and component readouts are copies. No pointer
  into Rust storage crosses the boundary; no canonical writable native memory
  is exported.
- **Typed storage.** Columns are `Vec<f32>` and `Vec<u32>`, never unchecked
  casts from byte storage. Values returned to Python are encoded as
  little-endian bytes, then read into a Python-owned array.

## Supported input boundary

Payloads and updates are bulk transfers, not per-entity calls:

| Call | Array shape | Dtype |
|---|---|---|
| `spawn(count, {name: array})` | `(count, width)` | `float32` or `uint32` matching the schema |
| `update_rows(name, handles, array)` | `(len(handles), width)` | same |
| `update_field(name, field, handles, array)` | `(len(handles),)` | same |
| handles in or out | `(n, 2)` | `uint64`, slot then generation |

A zero-width payload may be `None` or a zero-sized `(count, 0)` buffer. Arrays
must be C-contiguous; other strides, shapes and dtypes are rejected with the
expected shape and kind in the message. NumPy is a modeling and preparation
tool at this boundary, never a requirement inside the core crate.

Each submitted array is copied once into typed Rust storage. A caller may
mutate its array immediately after the call without changing native data. Each
readback is an owned copy. No zero-copy or frame-rate claim is made here.

## Python API

```python
from manifoldx._core_native import _NativeWorld

world = _NativeWorld(100_000)
world.register_component("transform.position", "f32", 3)
world.register_component("mesh.id", "u32", 1)
world.register_component("flags.marker", "u32", 0)

spawned = world.spawn(100_000, {
    "transform.position": positions,   # (100_000, 3) float32
    "mesh.id": mesh_ids,               # (100_000, 1) uint32
    "flags.marker": None,
})
handles = np.frombuffer(spawned["handles"], dtype="<u8").reshape(-1, 2)

world.update_rows("transform.position", handles[:5_000], new_positions)
world.update_field("mesh.id", 0, handles[2_000:4_000], new_mesh_ids)
flags = np.frombuffer(world.has("mesh.id", handles)["flags"], dtype=np.uint8)
readback = world.read_component("transform.position", handles[:3])
values = np.frombuffer(readback["data"], dtype="<f4").reshape(-1, 3)
world.destroy(handles[:1_000])
```

The result dictionaries also report `bytes_in` and `bytes_out`, so a caller can
count the boundary transfer without guessing. `snapshot()` returns the live
identities, schemas and counters as owned data.

## Evidence

`scripts/gates/native-component-world.py` is the runnable installed-wheel
example and focused gate. It asserts compiled-extension provenance, schema
uniqueness and locking, real values and zero defaults, membership, caller
mutation after submission, stale and malformed batch atomicity, destroy/reuse
clearing, capacity overflow and ordinary slot lifecycle. It then uploads 10k
and 100k entity worlds from NumPy, applies bulk row/field updates, queries and
reads back, destroys and reuses a subset, and prints the measured table and a
small before/after sample.

`scripts/gates/native-verify.sh` runs it from the disposable cwd after installing
the wheel, and again as a negative control with the extension removed.

## Limits

No scheduler, renderer, public ECS facade, general IR interpreter, GPU work,
Engine activation, Python reference backend, parity adapter, writable alias or
arbitrary dtype support. The bootstrap still reports `runtime: not implemented`,
and `MANIFOLDX_CORE=rust` still refuses to construct an Engine.
