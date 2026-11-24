# Rust core contracts v1

The starting tree is `4a500635afeae673dbd9c7af594b90c404307cfd`. The wheel
(#349), startup selection (#351), Python ECS reference tests (#354), private
native World (#363), and calling-thread Runtime (#365) supply the starting
behavior. Python owns gameplay, components, scene declarations, widgets, and
kernel source. Rust owns engine mechanisms. An unavailable native capability
fails visibly at the requested entrypoint.

## Kernel AST and typed IR

Version 1 covers the initial drive slice. Parse the function from its source
file at decoration time, with no execution of the body. There is one annotated
component argument and one typed scalar uniform `dt: f32`. The return is
`None`. An invocation addresses one selected entity. Dispatch guards lanes
beyond the selection length before touching any column. Specialization binds
component and field identifiers to typed columns; an IR compiled for a
different field schema must be rejected before dispatch.

| IR node | Required operands and type rules |
| --- | --- |
| `literal` | A finite numeric literal rounded to `f32` at compilation. |
| `uniform`, `local`, `field` | A typed identifier. Field references carry the component and field identifiers. |
| `neg`, `add`, `sub`, `mul` | Scalar/scalar and same-shape vector/vector arithmetic. `mul` also accepts scalar/vector in either order. Vector work is componentwise. |
| `clamp` | Three `f32` operands and an `f32` result. |
| `compare` | Two `f32` operands, an explicit ordered comparison or equality operator, and a bool result for a branch. |
| `let`, `store` | A typed local declaration, or assignment to a local or the invocation's own field. A store requires the exact target type. |
| `if` | A bool expression and ordered then/else statement lists. Only the chosen list executes. |

The IR header carries version, kernel name, component/schema binding, uniform
types and the inferred field read/write sets. The body is an ordered statement
list. Expression nodes retain their result type. Every node carries a source
span using one-based lines and columns in the original file and an exclusive
end. Diagnostics retain file, line, column, category and source text. An
augmented assignment lowers to a load, the typed operation and a store. Later
expressions see the store, including the position update's read of the new
speed. The IR agreement is on nodes and semantics. Its serialization and any
public ABI are outside this version.

Reject mutable globals, imports, NumPy, unlisted calls, recursion, loops,
dynamic objects, effects outside the row, implicit casts and unsupported
operators at import. `vec3` fields contain three `f32`s. The initial forward
direction is normalized input supplied by the caller. `@shader` remains a
separate GPU-only declaration. The source fixtures in `tests/core/fixtures`
and the test-only drive reference are inputs for the later compiler and backend
tests.

### Numeric domain for drive

Round the literal and every arithmetic result to nearest binary32, ties to
even. Preserve expression order including the two multiplies in the position
update. Rust must avoid fused operations. A WGSL adapter may contract or flush
floating-point values according to its implementation. Record its effective
flags, backend and driver and claim this slice only where the observed
results meet the corpus tolerance. Inputs and intermediate nonzero values in
the corpus are finite and normal. Zero compares by value. Treat subnormals,
nonfinite values, overflow, transcendental functions, casts and integer
arithmetic as outside the supported slice. Require finite nonnegative `dt`
and ordered inclusive clamp bounds.

The fixed corpus `tests/core/fixtures/drive-v1.json` carries six local cases
and a deterministic 600-step sequence at 60 Hz. There is no random input. The
reference evaluator is test-only. Compare each speed and position component
after every step using `abs(actual - ref) <= 1e-5 + 1e-5 * abs(ref)`.
At each sequence checkpoint also require accumulated position error at most
`1e-3` units per component. A backend must report its actual path even on
failure. E2 consumes these fixtures on the selected Rust CPU and WGSL
backends. E0 executes the reference and its reversed-acceleration and NaN
controls.

## Selection, ownership and copies

Selections and commands have a separate typed IR. A selection is a lazy
predicate over typed columns. Evaluate it at the command's FIFO position
against the current committed values. Each queued command has an ID, its
owned inputs and a bounded credit. Validate the identities, types, extents and
capacities of an entire command before publishing a mutation. A rejected
command produces its ID and error without consuming slots or publishing rows.
Full queues reject new work. Every accepted command gets its result or a
defined shutdown disposition, and the credit returns with consumption or
discard. Preserve #365's installed first-command-drop control and stale
generation checks as starting evidence. The current runtime runs on the
calling thread and accepts supplied frame timing.

The residency identity is (World, column, committed version). Entity
generation is checked independently of column version. CPU owns the initial
columns. A barrier publishes a new version with one writer per column. A
write invalidates other copies of the older version. Multiple consumers of the
same version on a destination share its current copy. A read needing a newer
version gets a transfer only when that destination lacks it. GPU compute
outputs stay on that device for compute and draw. Explicit CPU readback
copies the declared columns at the declared barrier, waits for that version,
and returns owned bytes or an immutable view of owned bytes. It never yields a
writable store alias. A snapshot carries generation, column version and frame
or dt. Admission bounds the queue and in-flight bytes by transfer direction
and allocation type. Record rows, bytes, calls, direction, version, consumers
and waits at the boundary.

| Case for later runtime tests | Observable outcome |
| --- | --- |
| Two consumers of one changed column | One transfer for that version and destination. |
| A write between reads | The second read sees the new version; an owned earlier snapshot still has the old values. |
| Slot reused after a snapshot | A command with the old generation fails without changing the replacement row. |
| Full queue or transfer budget | Admission fails before mutation, allocation or credit loss. |
| GPU compute followed by draw on the same device | The draw observes compute output with zero GPU-to-CPU transfers. |

The #363 binding currently copies Python input to a typed Rust Vec then to
column storage. Output goes from the column through an owned readout, encoded
bytes and Python bytes. Those costs are counted as delivered. Future tests
measure the transfers under the new residency policy.

## Native ABI and wheel matrix

Use one mixed `manifold-gfx` wheel with PyO3 and `manifoldx._core_native`.
The v1 builder layout is the existing `#[repr(C)]` SDK `CellRequest`,
`BuildOutput`, and `BuilderFn`. A plugin hands off a
`manifoldx.builder.v1` capsule containing the function pointer. Registration
validates the name/version before taking the pointer. Call-scoped borrowed
buffers stay with their originating allocator. Output sinks copy into
engine-owned staging. Publish only after status 0. The SDK wrapper returns
1 for a reported error and 2 for a caught panic, with a copied error message.
A failing call leaves the active state usable. Plugin state belongs in the
store and survives reload; neither side frees the other's allocation.
`manifoldx.system.v1` is the contracted name for the separate system capsule.
Its call layout and production loader are part of the SDK delivery. The SDK
fixture checks the existing builder function and sinks through C ABI,
including recovery in the same process. It does not install a capsule loader.

| Platform | CPython 3.14 | CPython 3.14t |
| --- | --- | --- |
| Linux x86-64 | Native wheel CI | Native wheel CI |
| Windows x86-64 | Native wheel CI | Native wheel CI |
| macOS | Pending runner evidence | Pending runner evidence |

Native CI records the candidate wheel name, SHA256, compressed bytes, installed
module origins, wheel tags, pip-compatible tags and GIL state before and after
imports. In a disposable environment the same ordinary smoke succeeds,
fails when the extension is removed, and succeeds again after restoration.
The selector separately checks an installed bootstrap that refuses Rust
Engine startup, a missing module and an existing binary the loader cannot
load. The ordinary smoke and the selector use isolated Python. The runtime
negative control must fail on the read ID when the first submission is
dropped.

The pre-core measurement starts at
`7e53eafe4c0bccd3d133991c1abd46709374fcbc`. CI runs the same wheel build,
clean install, Python Engine construction and installed CLI help on each
tree, per interpreter and runner, with separate cold uv caches. Compare
elapsed workload time and compressed wheel bytes in `comparison.json`.
Native-only compilation checks and extension tests are reported separately.
This measures the common packaging workload. GW-7 also requires an activated
native Engine and promotion evidence.

## Python render references

For each port, record the input and asset paths plus SHA256, engine and fixture
revision, installed artifact SHA256, camera/projection, seed, frame or tick,
fixed dt, adapter/backend/driver, dimensions, target format, channel or
feature thresholds, and the negative control. Keep references fixed before
starting the corresponding port. A new tolerance needs review with the
observable it covers. The entries below identify where to seed the remaining
captures and tests. Each effect needs its own observation or saved frame.

| ID | Existing seed for the port-specific scene or fixture |
| --- | --- |
| FR-01 | `tests/test_render_mvp.py` and the render pass tests. Capture order and visible output. |
| FR-02 | `tests/test_gamma.py` for flat sRGB; `examples/pbr.py` and material-specific examples for the remaining families. |
| FR-03 | `examples/textures.py`, `examples/svg_icons.py`, `examples/ibl.py`, `examples/volume.py`. |
| FR-04 | `examples/shadows.py`, `examples/spotlight.py`, `examples/ibl.py` and `docs/examples/sky_and_water.py`. |
| FR-05 | `examples/skinning.py`, `examples/axes.py`, `examples/point_cloud.py`, `examples/volume.py` and line/deformation tests. |
| FR-06 | `tests/test_post.py` and a saved off/on frame for each post option. |
| FR-07 | `examples/render_to_texture.py` and the readback/refresh cases. |
| FR-08 | `examples/quad_streaming.py` with a pinned LOD transition. |
| FR-09 | `examples/gas_compute.py`, `examples/nbody_compute.py`, `examples/point_cloud_compute.py`, `examples/gpu_field.py`. |
| FR-10 | `examples/gui.py` and the pointer and hit-test cases. |
| FR-11 | The integrated desktop app in its final display configuration, including scene and GUI composition. |

The first captured scene uses `scripts/gates/python-render-reference.py` from
outside the checkout with the installed wheel's Python and `-I`. It uses a
procedural 10×10 plane and `FlatMaterial("#cc2222")`, with no external assets
or RNG. Camera position (0, 0, 3), target (0, 0, 0), default projection.
One `render_frame` call with `dt=1/30` and supersample 1 writes a 32×32 PPM
containing RGB uint8 sRGB. The established `test_gamma.py` threshold is
(204, 34, 34) ±1 in each center-pixel channel. The script reports its adapter
and output SHA256. `--without-material` removes the plane and must fail that
same pixel check. This is a Python reference capture with the public
`render_frame` API. Window present and desktop frame time use separate
application runs.

The public run inventory at the starting tree is 46 top-level examples and
five demos. Excluding `web_walk.py` and `web_crowd.py` leaves 49. That total
includes the installed `native_runtime.py` diagnostic and the offline
`gltf_export.py` example. The public run inventory is exercised with its
feature checks at the desktop tools and promotion gates.
