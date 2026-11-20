"""Compute systems — first-class GPU work as ECS extension.

Phase 1 of the compute-systems design. Users subclass `Compute`, declare
component bindings via class-level annotations, and override `compile()`
to return raw WGSL. The engine extracts the bind-group layout from the
annotations, compiles the shader once, and dispatches the compute pipeline
each frame as part of the run loop.

Phase 2 (separate spec) will add a Python-as-shader DSL on top of this:
the user overrides `def main(self, i)` with traced Python, and the base
class's default `compile()` transpiles `main` to WGSL. The Phase-1 API
shape (class shape, annotations, marker types, bind layout) is identical
between phases — only the kernel-body language differs.

Spec: `.knowledge/analysis/2026-05-06-compute-systems-design.md`.
"""
from __future__ import annotations

import inspect
from typing import TYPE_CHECKING, Any, Callable, Dict, Generic, Optional, TypeVar

import numpy as np
from dataclasses import dataclass, field
from manifoldx import counters


# =============================================================================
# Marker types
# =============================================================================
#
# Reads[X] / Writes[X] / ReadsWrites[X] / Uniform[T] are subscriptable type
# markers used in Compute subclass annotations. Subscripting just stashes the
# parameter; Compute.__init_subclass__ walks the annotations and records the
# binding direction. The engine never instantiates these — they are pure
# class-level metadata.
#
# At runtime (the `else` branch below), they are simple non-generic classes
# whose metaclass returns a `_ParameterizedMarker` on subscription. For
# static type checkers, we expose them as Generic[T] storage-binding proxies
# so kernel code like `self.transforms[i].pos` resolves properly, and
# `Uniform[T]` resolves to `T` so `G: Uniform[float] = 1.0` type-checks.


class _ParameterizedMarker:
    __slots__ = ("base", "parameter")

    def __init__(self, base, parameter):
        self.base = base
        self.parameter = parameter

    def __repr__(self):
        return f"{self.base.__name__}[{self.parameter!r}]"


T = TypeVar("T")


if TYPE_CHECKING:
    # Static-checker view: storage bindings are indexable proxies whose
    # `[i]` returns a Component instance, and Uniform[T] is a transparent
    # alias for T so class-level uniform defaults type-check.
    class Reads(Generic[T]):
        """Read-only storage binding. `self.binding[i] -> T`."""

        def __getitem__(self, idx: int) -> T: ...

    class Writes(Generic[T]):
        """Write-only storage binding. `self.binding[i] -> T`."""

        def __getitem__(self, idx: int) -> T: ...

    class ReadsWrites(Generic[T]):
        """Read-write storage binding. `self.binding[i] -> T`."""

        def __getitem__(self, idx: int) -> T: ...

    # PEP 695 type alias: Uniform[T] === T for type-checking purposes.
    # Inside kernel bodies, `self.<uniform>` resolves to T so arithmetic
    # on uniforms type-checks. The trade-off is that auto-bound
    # sentinel-string defaults (e.g. `dt: Uniform[float] = "frame_dt"`)
    # need a `# type: ignore[assignment]` at the class body — narrow T
    # is preferred over `T | str` because arithmetic inside the kernel
    # is the hot path and arises every line; the sentinel default lines
    # arise once per uniform.
    type Uniform[U] = U
else:
    class _MarkerMeta(type):
        """Metaclass so subclasses get a working __class_getitem__."""

        def __getitem__(cls, parameter):
            return _ParameterizedMarker(cls, parameter)

    class Reads(metaclass=_MarkerMeta):
        """Marker for read-only component bindings (`storage<read>`)."""

    class Writes(metaclass=_MarkerMeta):
        """Marker for write-only component bindings (`storage<read_write>`).

        The shader is expected to write each entity's slot. Since wgpu doesn't
        distinguish read-only-write from read-write at the binding level,
        `Writes` and `ReadsWrites` map to the same WGSL access mode; the
        distinction is documentation for readers.
        """

    class ReadsWrites(metaclass=_MarkerMeta):
        """Marker for read-write component bindings (`storage<read_write>`)."""

    class Uniform(metaclass=_MarkerMeta):
        """Marker for scalar uniform parameters (packed into a single uniform buffer).

        Defaults can be:
        - A literal value (constant for the pipeline's lifetime).
        - A sentinel string (`"frame_dt"`, `"entity_count"`, etc.) — re-uploaded
          each frame, looked up in `_AUTO_BOUND_UNIFORMS`.
        """


# =============================================================================
# Auto-bound uniform / dispatch symbol registries
# =============================================================================


_AUTO_BOUND_UNIFORMS: Dict[str, Callable[[Any], float]] = {
    "frame_dt": lambda engine: float(getattr(engine, "_last_dt", 1 / 60)),
    "entity_count": lambda engine: int(np.sum(engine.store._alive)),
    "frame_index": lambda engine: int(getattr(engine, "_frame_index", 0)),
}


_DISPATCH_SYMBOLS: Dict[str, Callable[[Any], int]] = {
    "entity_count": lambda engine: int(np.sum(engine.store._alive)),
    "max_entities": lambda engine: int(engine.store.max_entities),
}


# =============================================================================
# Compute base class
# =============================================================================


class Compute:
    """Base class for declarative GPU compute systems.

    Subclass and:
    - Declare component bindings via class-level annotations
      (`Reads[X]`, `Writes[X]`, `ReadsWrites[X]`).
    - Declare uniform parameters via `Uniform[T]` annotations (with optional
      class-level defaults, either literals or sentinel strings).
    - Override class-level `workgroup_size: int` (default 64) and
      `dispatch` (default `"entity_count"`).
    - Override `compile()` to return a WGSL string. (Phase 2: override
      `main(self, i)` instead and the base class's default `compile()`
      will transpile it.)

    Then register with the engine: `engine.compute(MyCompute)`.
    """

    workgroup_size: int = 64
    dispatch: Any = "entity_count"  # str symbol | int | callable(engine) → int

    # Populated by __init_subclass__ from the class annotations.
    _reads: Dict[str, Any] = {}
    _writes: Dict[str, Any] = {}
    _uniforms: Dict[str, Any] = {}
    _uniform_defaults: Dict[str, Any] = {}

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)

        reads: Dict[str, Any] = {}
        writes: Dict[str, Any] = {}
        uniforms: Dict[str, Any] = {}
        uniform_defaults: Dict[str, Any] = {}

        # inspect.get_annotations, not __dict__["__annotations__"]: Python 3.14
        # evaluates class annotations lazily and leaves that key unset.
        annotations = inspect.get_annotations(cls)
        for name, ann in annotations.items():
            if name.startswith("_"):
                continue
            if isinstance(ann, _ParameterizedMarker):
                base = ann.base
                if base is Reads:
                    reads[name] = ann.parameter
                elif base is Writes or base is ReadsWrites:
                    writes[name] = ann.parameter
                elif base is Uniform:
                    uniforms[name] = ann.parameter
                    if name in cls.__dict__:
                        uniform_defaults[name] = cls.__dict__[name]
                else:
                    raise TypeError(
                        f"{cls.__name__}: annotation {name!r} uses an "
                        f"unrecognized marker {base.__name__}"
                    )
            else:
                # Non-marker annotations (workgroup_size: int, etc.) — ignore.
                continue

        cls._reads = reads
        cls._writes = writes
        cls._uniforms = uniforms
        cls._uniform_defaults = uniform_defaults

    # --- API ---------------------------------------------------------------

    def compile(self) -> str:
        """Default: trace `main` (and any helpers) to WGSL via the Phase-2 transpiler.

        Override for hand-written WGSL kernels.
        """
        from manifoldx.compute.transpile import transpile_compute
        return transpile_compute(type(self))

    # --- Internals (used by the engine) ------------------------------------

    @classmethod
    def _bind_group_layout(cls) -> list[dict]:
        """Compute the bind-group layout from the class annotations.

        Slot 0 is the packed uniform buffer IF the class declares any
        Uniform[T] annotations; otherwise slot 0 is the first Reads buffer.
        Reads come first (in declaration order), then Writes / ReadsWrites
        (in declaration order). Returns a list of dicts; the engine consumes
        them to build the actual wgpu bind-group layout.
        """
        layout: list[dict] = []
        slot = 0
        if cls._uniforms:
            layout.append(
                {"binding": slot, "name": "_uniforms", "kind": "uniform", "access": "read"}
            )
            slot += 1
        for name in cls._reads:
            layout.append(
                {"binding": slot, "name": name, "kind": "storage", "access": "read"}
            )
            slot += 1
        for name in cls._writes:
            layout.append(
                {
                    "binding": slot,
                    "name": name,
                    "kind": "storage",
                    "access": "read_write",
                }
            )
            slot += 1
        return layout


__all__ = [
    "Compute",
    "ComputeRunner",
    "Reads",
    "Writes",
    "ReadsWrites",
    "Uniform",
]


# =============================================================================
# Engine-side runner — owns pipelines, dispatches each frame
# =============================================================================


class ComputeRunner:
    """Engine-side manager for registered Compute classes.

    Lifecycle:
    - `register(cls)`: append a Compute class to the registry. No GPU work yet.
    - `run_all(dt)`: called once per frame after CPU command flush. For each
      registered class, compile-on-first-use, sync mirrored Reads from CPU
      to GPU, dispatch the compute pipeline, sync mirrored Writes back.

    Phase 1 v1 simplifications (documented in the design doc):
    - Each compute pass uses its own command encoder + submit (no batching).
    - Mirrored components round-trip CPU→GPU→CPU each frame; gpu_only
      components stay on GPU.
    - One bind group, group=0, slot 0 = uniforms, 1..K = Reads, K+1.. = Writes.
    """

    def __init__(self, engine):
        self._engine = engine
        self._registered: list[type] = []
        self._compiled: dict[type, _CompiledCompute] = {}

    def register(self, cls: type) -> None:
        if not isinstance(cls, type) or not issubclass(cls, Compute):
            raise TypeError(
                f"engine.compute(...) expects a Compute subclass; got {cls!r}"
            )
        if cls not in self._registered:
            self._registered.append(cls)

    def run_all(self, dt: float) -> None:
        if not self._registered or self._engine._device is None:
            return
        for cls in self._registered:
            if cls not in self._compiled:
                self._compile(cls)
            self._dispatch(cls, dt)

    def _compile(self, cls: type) -> None:
        import wgpu
        engine = self._engine
        device = engine._device

        instance = cls()
        wgsl = instance.compile()

        # Build the bind-group layout from the class annotations.
        layout = cls._bind_group_layout()
        bgl_entries = []
        for slot in layout:
            if slot["kind"] == "uniform":
                bgl_entries.append(
                    {
                        "binding": slot["binding"],
                        "visibility": wgpu.ShaderStage.COMPUTE,
                        # min_binding_size=0 disables the runtime check so the
                        # shader's static-analysis "expected size" doesn't
                        # need to match the bound buffer exactly.
                        "buffer": {
                            "type": wgpu.BufferBindingType.uniform,
                            "min_binding_size": 0,
                        },
                    }
                )
            else:
                bbt = (
                    wgpu.BufferBindingType.read_only_storage
                    if slot["access"] == "read"
                    else wgpu.BufferBindingType.storage
                )
                bgl_entries.append(
                    {
                        "binding": slot["binding"],
                        "visibility": wgpu.ShaderStage.COMPUTE,
                        "buffer": {
                            "type": bbt,
                            "min_binding_size": 0,
                        },
                    }
                )

        bind_group_layout = device.create_bind_group_layout(entries=bgl_entries)
        pipeline_layout = device.create_pipeline_layout(
            bind_group_layouts=[bind_group_layout]
        )
        shader_module = device.create_shader_module(code=wgsl)
        pipeline = device.create_compute_pipeline(
            layout=pipeline_layout,
            compute={"module": shader_module, "entry_point": "main"},
        )

        # Uniform buffer — only allocated when the class declares uniforms.
        if cls._uniforms:
            u_size = _round_up(4 * len(cls._uniforms), 16)
            uniform_buffer = device.create_buffer(
                size=u_size,
                usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST,
            )
        else:
            uniform_buffer = None

        # Storage buffers — one per component referenced by Reads/Writes.
        component_buffers: dict[str, Any] = {}
        for binding_name, comp_type in {**cls._reads, **cls._writes}.items():
            store_array = engine.store._components[comp_type.__name__]
            n_bytes = int(store_array.nbytes)
            buf = device.create_buffer(
                size=n_bytes,
                usage=(
                    wgpu.BufferUsage.STORAGE
                    | wgpu.BufferUsage.COPY_SRC
                    | wgpu.BufferUsage.COPY_DST
                ),
            )
            component_buffers[binding_name] = (comp_type, buf, n_bytes)

        self._compiled[cls] = _CompiledCompute(
            instance=instance,
            pipeline=pipeline,
            bind_group_layout=bind_group_layout,
            pipeline_layout=pipeline_layout,
            uniform_buffer=uniform_buffer,
            component_buffers=component_buffers,
            wgsl=wgsl,
        )

    def _dispatch(self, cls: type, dt: float) -> None:
        import wgpu
        engine = self._engine
        device = engine._device
        cc = self._compiled[cls]

        # 1. Sync mirrored components from CPU numpy → GPU storage buffer.
        # Both Reads and ReadsWrites need their initial state uploaded each
        # frame (Reads so the shader has fresh data; ReadsWrites so the
        # shader can read the current state before mutating it). Pure Writes
        # don't strictly need an upload, but Phase 1 doesn't distinguish
        # Writes from ReadsWrites — both go through _writes. The cost is
        # small so we upload regardless. Only rows below the store's high-water
        # mark move: no entity has ever lived above it.
        rows = engine.store._high_water
        all_input_bindings = list(cls._reads) + list(cls._writes)
        for binding_name in all_input_bindings:
            comp_type, buf, n_bytes = cc.component_buffers[binding_name]
            comp_name = comp_type.__name__
            if rows and not getattr(comp_type, "_gpu_only", False):
                arr = engine.store._components[comp_name]
                counters.write_buffer(device.queue, buf, 0, arr[:rows])

        # 2. Resolve uniforms + pack into bytes.
        engine._last_dt = dt  # auto-bound 'frame_dt' uses this
        uniform_values = []
        for name in cls._uniforms:
            default = cls._uniform_defaults.get(name)
            if isinstance(default, str):
                resolver = _AUTO_BOUND_UNIFORMS.get(default)
                if resolver is None:
                    raise ValueError(
                        f"Unknown auto-bound uniform sentinel {default!r} "
                        f"for {cls.__name__}.{name}"
                    )
                value = resolver(engine)
            elif default is None:
                value = 0.0
            else:
                value = default
            uniform_values.append(float(value))

        if uniform_values and cc.uniform_buffer is not None:
            packed = np.array(uniform_values, dtype=np.float32).tobytes()
            # Pad to 16-byte alignment.
            pad = (-len(packed)) % 16
            if pad:
                packed = packed + b"\x00" * pad
            counters.write_buffer(device.queue, cc.uniform_buffer, 0, packed)

        # 3. Build the bind group fresh each dispatch (cheap; lets buffers move).
        entries = []
        slot = 0
        if cc.uniform_buffer is not None:
            entries.append(
                {
                    "binding": slot,
                    "resource": {
                        "buffer": cc.uniform_buffer,
                        "offset": 0,
                        "size": cc.uniform_buffer.size,
                    },
                }
            )
            slot += 1
        for binding_name in {**cls._reads, **cls._writes}:
            _comp_type, buf, n_bytes = cc.component_buffers[binding_name]
            entries.append(
                {
                    "binding": slot,
                    "resource": {"buffer": buf, "offset": 0, "size": n_bytes},
                }
            )
            slot += 1
        bind_group = device.create_bind_group(
            layout=cc.bind_group_layout, entries=entries
        )

        # 4. Resolve dispatch size + workgroup count.
        dispatch_value = cls.dispatch
        if isinstance(dispatch_value, str):
            resolver = _DISPATCH_SYMBOLS.get(dispatch_value)
            if resolver is None:
                raise ValueError(
                    f"Unknown dispatch symbol {dispatch_value!r} for {cls.__name__}"
                )
            n_threads = resolver(engine)
        elif callable(dispatch_value):
            n_threads = int(dispatch_value(engine))
        else:
            n_threads = int(dispatch_value)
        workgroup_count = (n_threads + cls.workgroup_size - 1) // cls.workgroup_size
        if workgroup_count <= 0:
            return

        # 5. Encode + submit compute pass.
        encoder = device.create_command_encoder()
        compute_pass = encoder.begin_compute_pass()
        compute_pass.set_pipeline(cc.pipeline)
        compute_pass.set_bind_group(0, bind_group, [], 0, 0)
        compute_pass.dispatch_workgroups(workgroup_count, 1, 1)
        compute_pass.end()

        # 6. Sync mirrored Writes back to CPU before submit completes.
        # We read each write-buffer back through a staging buffer; for v1
        # we just queue.read_buffer (sync).
        for binding_name in cls._writes:
            comp_type, buf, n_bytes = cc.component_buffers[binding_name]
            comp_name = comp_type.__name__
            if getattr(comp_type, "_gpu_only", False) or not rows:
                continue
            # Use copy_buffer_to_buffer staging approach is cleaner, but for
            # simplicity v1 reads back via queue.read_buffer after submit.
            cc.pending_readbacks.append(binding_name)

        device.queue.submit([encoder.finish()])

        # 7. Read back mirrored writes (sync; blocks until GPU done).
        for binding_name in list(cc.pending_readbacks):
            comp_type, buf, n_bytes = cc.component_buffers[binding_name]
            comp_name = comp_type.__name__
            arr = engine.store._components[comp_name][:rows]
            data = device.queue.read_buffer(buf, 0, arr.nbytes)
            np.copyto(arr, np.frombuffer(data, dtype=arr.dtype).reshape(arr.shape))
        cc.pending_readbacks.clear()


def _round_up(value: int, multiple: int) -> int:
    return ((value + multiple - 1) // multiple) * multiple




@dataclass
class _CompiledCompute:
    instance: "Compute"
    pipeline: Any
    bind_group_layout: Any
    pipeline_layout: Any
    uniform_buffer: Any
    component_buffers: dict
    wgsl: str
    pending_readbacks: list = field(default_factory=list)
