"""Public native World and its calling-thread runtime."""

from dataclasses import dataclass, field
from functools import wraps
from importlib import import_module
from operator import index
from struct import iter_unpack, pack
from threading import current_thread, local, main_thread


@dataclass(frozen=True, slots=True)
class Entity:
    """An entity identity tied to one World and one slot generation."""

    slot: int
    generation: int
    _owner: object = field(repr=False)


@dataclass(frozen=True, slots=True)
class Clock:
    """The native simulation clock at a frame boundary."""

    frame: int
    time: float
    dt: float


@dataclass(frozen=True, slots=True)
class Readback:
    """Owned data for one committed column version."""

    version: int
    entities: tuple[Entity, ...]
    membership: bytes
    _data: bytes = field(repr=False)
    _kind: str
    _width: int

    def numpy(self):
        """View the owned bytes as a read-only NumPy array."""
        import numpy as np

        dtype = {"f32": "<f4", "u32": "<u4"}[self._kind]
        return np.frombuffer(self._data, dtype=dtype).reshape(len(self.entities), self._width)


@dataclass(frozen=True, slots=True)
class _Field:
    world: object
    component: int
    offset: int
    width: int
    kind: str

    def scalar(self, value):
        if self.width != 1:
            raise ValueError("selection and assignment require a scalar field")
        return (value, None) if self.kind == "f32" else (None, index(value))

    def __lt__(self, value):
        return self._compare("lt", value)

    def __le__(self, value):
        return self._compare("le", value)

    def __gt__(self, value):
        return self._compare("gt", value)

    def __ge__(self, value):
        return self._compare("ge", value)

    def eq(self, value):
        return self._compare("eq", value)

    def ne(self, value):
        return self._compare("ne", value)

    def _compare(self, operation, value):
        return _Predicate(self.world, (self.component, self.offset, operation, self.scalar(value)))


@dataclass(frozen=True, slots=True)
class _Predicate:
    world: object
    ir: tuple

    def __bool__(self):
        raise TypeError("pass predicates separately to where()")


@dataclass(frozen=True, slots=True)
class _Selection:
    world: object
    component: int
    predicates: tuple = ()

    def where(self, *predicates):
        if any(not isinstance(p, _Predicate) or p.world is not self.world for p in predicates):
            raise ValueError("selection predicates must belong to this World")
        if len(self.predicates) + len(predicates) > 16:
            raise ValueError("too many predicates")
        return _Selection(
            self.world, self.component, (*self.predicates, *(p.ir for p in predicates))
        )

    def read(self, column=None):
        component = self.component if column is None else self.world._column(column)
        resident = self.world._resident_closed
        if component == resident or any(predicate[0] == resident for predicate in self.predicates):
            raise RuntimeError("resident column is unavailable after the desktop device closes")
        return self.world._readback(
            self.world._loop.read(self.component, self.predicates, component)
        )

    def numpy(self, name=None):
        if name is None:
            return self.read().numpy()
        column = self.world._field_by_index(self.component, name)
        return self.read().numpy()[:, column.offset : column.offset + column.width]


class _Snapshot:
    def __init__(self, world, lease):
        self._world = world
        self._lease = lease

    @property
    def clock(self):
        return Clock(*self._lease.clock)

    @property
    def overruns(self):
        return self._lease.overruns

    def query(self, component):
        # A selection remains a lazy command input. The lease controls reading.
        self._lease.clock
        return self._world.query(component)

    def read(self, component):
        return self._world._readback(self._lease.read(self._world._column(component)))


class _Commands:
    def __init__(self, world, lease):
        self._world = world
        self._lease = lease

    def set(self, selection, column, value):
        return self._lease.set(*self._world._assignment(selection, column, value))

    def replace(self, component, entities, rows):
        return self._lease.replace(
            self._world._column(component),
            lambda: self._world._handles(entities, command=True),
            rows,
        )

    def spawn(self, *components, count=1, payload=None):
        count = self._world._checked_count(count)
        return self._lease.spawn(count, lambda: self._world._payload(count, components, payload))

    def destroy(self, entities):
        return self._lease.destroy(lambda: self._world._handles(entities, command=True))

    def emit(self, name, value):
        self._lease.emit(name, value)


class PythonGraphicsBridge:
    """Explicit main-thread graphics consumer for the transition to native rendering.

    The Python renderer receives a single snapshot lease after each barrier.
    It owns its window, surface, device, queue and renderer as one unit.
    """

    def __init__(self, present, *, reads):
        if current_thread() is not main_thread():
            raise RuntimeError("create the graphics bridge on the main thread")
        if not callable(present):
            raise TypeError("graphics bridge requires a present callback")
        self._present = present
        self._reads = tuple(reads)

    def _bind(self, world):
        if current_thread() is not main_thread():
            raise RuntimeError("run the graphics bridge on the main thread")
        reads = world._columns(self._reads)

        def present(snapshot):
            world._callback.active = True
            try:
                self._present(_Snapshot(world, snapshot))
            finally:
                world._callback.active = False

        return reads, present


class World:
    """Fixed-capacity native ECS and frame runtime with one owner thread.

    Setup takes place on the owner thread. Producers can submit operations
    and take outcomes while Rust runs. Systems access declared snapshot
    columns and publish bounded commands between frames.
    """

    def __init__(self, capacity, *, queue_capacity=64, batches_per_frame=8, max_command_rows=4096):
        native = import_module("manifoldx._core_native")
        self._thread = current_thread()
        self._callback = local()
        self._structural = native._STRUCTURAL_ACCESS
        self._capacity = capacity
        self._max_command_rows = max_command_rows
        self._identity = object()
        self._native = native._NativeWorld(capacity)
        self._loop = native.FrameLoop(
            self._native, queue_capacity, batches_per_frame, max_command_rows
        )
        self._schemas = {}
        self._metadata = []
        self._systems = []
        self._running = False
        self._started = False
        self._resident_closed = None

    def _owner(self):
        if current_thread() is not self._thread:
            raise RuntimeError("operation requires the World owner thread")
        if self._running:
            raise RuntimeError("use a snapshot or command lease during a frame")

    def register(self, component, *, kind=None, width=None):
        """Register a Component class or an explicitly typed component name."""
        self._owner()
        if self._started:
            raise RuntimeError("registration closes when the runtime starts")
        if isinstance(component, str):
            name, layout = component, {}
            if kind not in ("f32", "u32") or width is None:
                raise ValueError("named components require a kind and width")
        else:
            from manifoldx.components import Component

            if not isinstance(component, type) or not issubclass(component, Component):
                raise TypeError("register a Component subclass or an explicit name")
            if kind is not None or width is not None or component._gpu_only:
                raise ValueError("Component classes require their declared CPU layout")
            name, layout = component.__name__, component._layout.copy()
            kind = {"f4": "f32", "u4": "u32"}.get(component._dtype.str[1:])
            if kind is None or not component._dtype.isnative:
                raise TypeError("native World supports native-endian float32 and uint32")
            width = component._shape[0]
        column = self._native.register_component(name, kind, width)
        self._schemas[component] = column
        self._metadata.append((name, kind, width, layout))
        return column

    def _column(self, component):
        try:
            return self._schemas[component]
        except KeyError:
            raise ValueError(f"unregistered component {component!r}") from None

    def _columns(self, components):
        return list(dict.fromkeys(self._field_column(c) for c in components))

    def _field_column(self, component):
        if isinstance(component, _Field):
            if component.world is not self:
                raise ValueError("field belongs to a different World")
            return component.component
        return self._column(component)

    def _field_by_index(self, component, name):
        _, kind, width, layout = self._metadata[component]
        if isinstance(name, str):
            try:
                offset, count = layout[name]
            except KeyError:
                raise ValueError(f"unknown field {name!r}") from None
        else:
            offset, count = index(name), 1
        if offset < 0 or offset + count > width:
            raise ValueError("field outside the component row")
        return _Field(self, component, offset, count, kind)

    def field(self, component, name=0):
        """Identify a scalar or vector field without reading its values."""
        return self._field_by_index(self._column(component), name)

    def _entities(self, data):
        return tuple(
            Entity(slot, generation, self._identity)
            for slot, generation in iter_unpack("<QQ", data)
        )

    def _handles(self, entities, *, command=False):
        data = bytearray()
        limit = min(self._capacity, self._max_command_rows) if command else self._capacity
        for row, entity in enumerate(entities):
            if row >= limit:
                raise ValueError("entity handles exceed the row limit")
            if not isinstance(entity, Entity) or entity._owner is not self._identity:
                raise ValueError("entity belongs to a different World")
            data.extend(pack("=QQ", entity.slot, entity.generation))
        if data:
            return memoryview(data).cast("Q", shape=(len(data) // 16, 2))
        import numpy as np

        return np.empty((0, 2), dtype="uint64")

    def _readback(self, data):
        return Readback(
            data["version"],
            self._entities(data["handles"]),
            data["membership"],
            data["data"],
            data["kind"],
            data["width"],
        )

    def _checked_count(self, count, *, command=True):
        count = index(count)
        limit = min(self._capacity, self._max_command_rows) if command else self._capacity
        if not 1 <= count <= limit:
            raise ValueError(f"entity count must be between 1 and {limit}")
        return count

    def _payload(self, count, values, payload):
        rows = {self._column(component): data for component, data in (payload or {}).items()}
        for value in values:
            component = self._column(type(value))
            if component in rows:
                raise ValueError("duplicate component payload")
            rows[component] = value.get_data(count)
        return rows

    def spawn(self, *components, count=1, payload=None):
        """Publish setup entities with Component instances or typed input buffers."""
        self._owner()
        count = self._checked_count(count, command=False)
        rows = self._payload(count, components, payload)
        named = {self._metadata[column][0]: data for column, data in rows.items()}
        return self._entities(self._native.spawn(count, named)["handles"])

    def replace(self, component, entities, data):
        """Copy whole rows into the idle World after validating the complete batch."""
        self._owner()
        name = self._metadata[self._column(component)][0]
        return self._native.update_rows(name, self._handles(entities), data)

    def query(self, component):
        """Build a lazy membership selection over a registered component."""
        return _Selection(self, self._column(component))

    def _assignment(self, selection, column, value):
        if not isinstance(selection, _Selection) or selection.world is not self:
            raise ValueError("selection belongs to a different World")
        if not isinstance(column, _Field) or column.world is not self:
            raise ValueError("field belongs to a different World")
        return (
            selection.component,
            selection.predicates,
            column.component,
            column.offset,
            column.scalar(value),
        )

    def _external(self):
        if getattr(self._callback, "active", False):
            raise RuntimeError("submit through the command lease during a callback")

    def set(self, selection, column, value):
        """Enqueue a typed assignment over a selection evaluated at the barrier."""
        self._external()
        return self._loop.set(*self._assignment(selection, column, value))

    def destroy(self, entities):
        """Enqueue destruction using each entity's generation."""
        self._external()
        return self._loop.destroy(lambda: self._handles(entities, command=True))

    def submit_spawn(self, *components, count=1, payload=None):
        """Enqueue a producer spawn for the next barrier."""
        self._external()
        count = self._checked_count(count)
        return self._loop.spawn(count, lambda: self._payload(count, components, payload))

    def outcome(self):
        """Take a completion as (command ID, frame, error) and free its credit."""
        return self._loop.outcome()

    @property
    def rows_read(self):
        """Counts of explicit readback rows, bytes and calls."""
        return self._loop.reads

    @property
    def queue_full(self):
        return self._loop.full

    def system(self, function=None, *, hz=None, reads=(), writes=(), structural=False):
        """Register a Python system. A rate opts into snapshot-based async work."""
        self._owner()
        if self._started:
            raise RuntimeError("registration closes when the runtime starts")
        read_columns, write_columns = self._columns(reads), self._columns(writes)
        if structural:
            write_columns.append(self._structural)

        def register(callback):
            self._owner()
            if self._started:
                raise RuntimeError("registration closes when the runtime starts")

            @wraps(callback)
            def invoke(snapshot, commands):
                self._callback.active = True
                try:
                    return callback(_Snapshot(self, snapshot), _Commands(self, commands))
                finally:
                    self._callback.active = False

            self._systems.append((callback.__qualname__, hz, read_columns, write_columns, invoke))
            return callback

        return register if function is None else register(function)

    def _compile_kernel(self, program, target):
        from manifoldx.compute.kernel import _Kernel, _native_words

        self._owner()
        if not isinstance(program, _Kernel):
            raise TypeError("expected a declaration compiled by @kernel")
        if target != "cpu":
            raise ValueError("World supports target='cpu'; WGSL requires an explicit GPU group")
        column = self._column(program.component)
        name, kind, width, layout = self._metadata[column]
        if kind != "f32" or width != program.width or tuple(layout.items()) != program.layout:
            raise ValueError("kernel column differs from the compiled schema")
        native = import_module("manifoldx._core_native")
        program_type = getattr(native, "_KernelProgram", None)
        if program_type is None:
            raise RuntimeError("Rust CPU kernel backend is unavailable")
        compiled = program_type(program.version, program.name, name, width, _native_words(program))
        return column, compiled

    def dispatch_kernel(self, program, *, dt, target):
        """Apply one v1 kernel to the idle World, without copying its rows."""
        column, compiled = self._compile_kernel(program, target)
        return self._loop.dispatch_kernel(column, compiled, (program.version, dt))

    def kernel(self, program, *, target):
        """Schedule a compiled kernel on this World's native CPU frame lane."""
        self._owner()
        if self._started:
            raise RuntimeError("registration closes when the runtime starts")
        column, compiled = self._compile_kernel(program, target)
        writes = [column] if program.writes else []
        self._systems.append((program.name, None, [], writes, (column, compiled)))
        return program

    def run(self, *, frames, hz=60, paced=True, graphics=None, _profile=False):
        """Run the native clock until the frame limit or a stop request."""
        self._owner()
        bridge = None
        if graphics is not None:
            if not isinstance(graphics, PythonGraphicsBridge):
                raise TypeError("pass an explicit PythonGraphicsBridge")
            bridge = graphics._bind(self)
        self._running = True
        try:
            result = self._loop.run(frames, hz, paced, self._systems, bridge, _profile)
        finally:
            self._running = False
            self._started = self._loop.stopped
        return result

    def stop(self):
        """Stop admission and ask the native runtime to finish at its boundary."""
        self._loop.stop()
