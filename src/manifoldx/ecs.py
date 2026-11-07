"""Core Entity Component System: EntityStore, ComponentView, Query, decorators."""

import numpy as np
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from manifoldx.commands import CommandBuffer


# Global validation flag - set to False in production to disable warnings
ENABLE_VALIDATION = True


# =============================================================================
# EntityStore - Central storage for all entity data (SoA layout)
# =============================================================================


class EntityStore:
    """
    Central storage for all entity data using Structure of Arrays (SoA).

    Data layout: Each component has its own array, not interleaved.

    Attributes:
        max_entities: Maximum number of entities supported
        _alive: Boolean array indicating which entities are alive
        _components: dict of component_name -> np.ndarray (max_entities, component_size)
    """

    def __init__(self, max_entities: int = 100_000):
        self.max_entities = max_entities
        self._alive = np.zeros(max_entities, dtype=bool)
        self._components: dict[str, np.ndarray] = {}
        # Per component, which slots have it: set by the spawn that filled the slot.
        self._members: dict[str, np.ndarray] = {}
        self._free_list: list[int] = []  # Reusable dead entity indices
        # One past the highest slot ever spawned into: every live entity sits
        # below it, so GPU syncs move only rows [0, _high_water).
        self._high_water = 0
        # Called with the indices of every spawn and destroy, so per-entity
        # state kept outside the store (joint blocks) never outlives its entity.
        self._release_hooks: list = []

    def register_component(self, name: str, dtype: np.dtype, shape: tuple):
        """
        Register a component type, allocate storage array.

        Args:
            name: Component name
            dtype: NumPy dtype for component
            shape: Shape of each component value (e.g., (3,) for Vector3)
        """
        full_shape = (self.max_entities,) + shape
        self._components[name] = np.zeros(full_shape, dtype=dtype)
        self._members[name] = np.zeros(self.max_entities, dtype=bool)

    def spawn(self, n: int, **component_data) -> np.ndarray:
        """
        Spawn n entities, return indices.

        Args:
            n: Number of entities to spawn
            **component_data: Dict of {component_name: np.ndarray}
                              Arrays must be (n, component_size) or broadcastable

        Returns:
            np.ndarray of entity indices
        """
        # Free list first, popped from its end, then the lowest never-used slots.
        # Every free-list slot is dead, so the remaining dead slots exclude them.
        k = min(n, len(self._free_list))
        reused = self._free_list[len(self._free_list) - k :][::-1]
        dead = np.flatnonzero(~self._alive)
        if len(dead) < n:
            raise RuntimeError(
                f"Max entities {self.max_entities} reached: spawning {n} with "
                f"{len(dead)} free slots; raise it with Engine(max_entities=...)"
            )
        fresh = dead[~np.isin(dead, reused)][: n - k] if k else dead[:n]
        indices = np.concatenate([np.asarray(reused, dtype=np.int32), fresh.astype(np.int32)])
        del self._free_list[len(self._free_list) - k :]
        self._alive[indices] = True
        if len(indices):
            self._high_water = max(self._high_water, int(indices.max()) + 1)
        for hook in self._release_hooks:
            hook(indices)
        # A filled slot has the components given and Transform, and forgets the rest.
        for name, members in self._members.items():
            members[indices] = name in component_data or name == "Transform"

        # Write component data to storage
        for comp_name, data in component_data.items():
            if comp_name not in self._components:
                raise ValueError(f"Component {comp_name} not registered")

            data = np.asarray(data, dtype=self._components[comp_name].dtype)

            # Broadcast scalar to array
            if data.size == 1 and n > 1:
                data = np.full((n,) + data.shape, data.item(), dtype=data.dtype)
            elif data.shape[0] != n:
                raise ValueError(f"Component {comp_name}: expected {n} rows, got {data.shape[0]}")

            self._components[comp_name][indices] = data

        return indices

    def destroy(self, indices: np.ndarray):
        """
        Mark entities as dead.

        Args:
            indices: Array of entity indices to destroy
        """
        if len(indices) == 0:
            return
        # Each index once and only if alive, so a twice-destroyed entity does not
        # enter the free list twice (two spawns would then share its slot).
        indices = np.unique(np.asarray(indices))
        indices = indices[self._alive[indices]]
        self._alive[indices] = False
        for hook in self._release_hooks:
            hook(indices)
        # Add to free list for reuse
        self._free_list.extend(indices.tolist())

    def get_component_view(self, component_names: list[str], engine=None) -> "ComponentView":
        """A view of the alive entities that have every named component, ascending.

        A name that is not registered matches nothing; no names give every alive entity.
        """
        mask = self._alive.copy()
        for name in component_names:
            members = self._members.get(name)
            if members is None:
                mask[:] = False
                break
            mask &= members
        return ComponentView(self, component_names, np.flatnonzero(mask), engine)

    def get_component_data(self, component_name: str, indices: np.ndarray) -> np.ndarray:
        """Get component data for specific entity indices."""
        if component_name not in self._components:
            raise ValueError(f"Component {component_name} not registered")
        return self._components[component_name][indices]


# =============================================================================
# TransformFieldView - Writable view into a Transform sub-field
# =============================================================================


class _FieldView:
    """View into a sub-field of a component.

    Supports +=, -=, <=, *, etc. Knows its absolute entity indices.
    """

    def __init__(
        self,
        store: "EntityStore",
        indices: np.ndarray,
        component_name: str,
        col_start: int,
        col_end: int,
        field_name: str = "",
        commands=None,
    ):
        self._store = store
        self._indices = indices
        self._component_name = component_name
        self._col_start = col_start
        self._col_end = col_end
        self._field_name = field_name
        self._commands = commands

    def __setattr__(self, name, value):
        """Handle assignment: field_view.pos = (x, y, z) queues update command."""
        # Let private attributes go through normally
        if name.startswith("_"):
            super().__setattr__(name, value)
            return

        # For public attributes (like 'x', 'y', 'z' or vector fields),
        # convert to numpy array and call _set_data
        # This handles both scalar assignments and tuple/array assignments
        if self._col_start is not None and self._col_end is not None:
            # Convert value to numpy array (handles tuples, lists, scalars)
            value = np.asarray(value, dtype=np.float32)
            self._set_data(value)
        else:
            # Fall back to normal attribute storage
            super().__setattr__(name, value)

    @property
    def data(self) -> np.ndarray:
        """Get current data."""
        return self._store._components[self._component_name][
            self._indices, self._col_start : self._col_end
        ]

    def _get_data(self) -> np.ndarray:
        return self._store._components[self._component_name][
            self._indices, self._col_start : self._col_end
        ]

    def _set_data(self, value):
        # Validate data before writing
        value = np.asarray(value, dtype=np.float32)

        if ENABLE_VALIDATION:
            if np.any(np.isnan(value)):
                import warnings

                warnings.warn(
                    f"⚠️ NaN detected in {self._component_name}.{self._field_name} "
                    f"for {len(self._indices)} entities"
                )

            if np.any(np.isinf(value)):
                import warnings

                warnings.warn(
                    f"⚠️ Inf detected in {self._component_name}.{self._field_name} "
                    f"for {len(self._indices)} entities"
                )

            # Check for invalid rotation (zero quaternion)
            if self._field_name == "rotation" and self._component_name == "Transform":
                rot_magnitude = np.linalg.norm(value.reshape(-1, 4), axis=1)
                if np.any(rot_magnitude < 0.01):
                    import warnings

                    warnings.warn(
                        f"⚠️ Near-zero rotation detected in Transform.rotation "
                        f"for {(rot_magnitude < 0.01).sum()} entities"
                    )

        if self._commands is not None:
            from manifoldx.commands import Command, CommandType

            self._commands.append(
                Command(
                    CommandType.UPDATE_COMPONENT,
                    {
                        "component_name": self._component_name,
                        "indices": self._indices.copy(),
                        "new_data": value.copy(),
                        "col_start": self._col_start,
                        "col_end": self._col_end,
                    },
                )
            )
        else:
            self._store._components[self._component_name][
                self._indices, self._col_start : self._col_end
            ] = value

    def __iadd__(self, other):
        """In-place add. For rotation, composes quaternions."""
        other = np.asarray(other, dtype=np.float32)

        if self._field_name == "rotation" and self._component_name == "Transform":
            current = self._get_data().copy()
            result = _quat_multiply(other, current)
            self._set_data(result)
        else:
            self._set_data(self._get_data() + other)
        return self

    def __isub__(self, other):
        other = np.asarray(other, dtype=np.float32)
        self._set_data(self._get_data() - other)
        return self

    def __imul__(self, other):
        other = np.asarray(other, dtype=np.float32)
        self._set_data(self._get_data() * other)
        return self

    def __mul__(self, other):
        """Multiply field data by other (scalar, array, or field_view)."""
        if isinstance(other, _FieldView):
            other = other._get_data()
        return self._get_data() * np.asarray(other, dtype=np.float32)

    def __rmul__(self, other):
        """other * field_view (scalar, array, or field_view)."""
        return self.__mul__(other)

    def __add__(self, other):
        """field_view + other (scalar, array, or field_view)."""
        if isinstance(other, _FieldView):
            other = other._get_data()
        return self._get_data() + np.asarray(other, dtype=np.float32)

    def __sub__(self, other):
        """field_view - other (scalar, array, or field_view)."""
        if isinstance(other, _FieldView):
            other = other._get_data()
        return self._get_data() - np.asarray(other, dtype=np.float32)

    def __truediv__(self, other):
        """field_view / other."""
        return self._get_data() / np.asarray(other, dtype=np.float32)

    def __radd__(self, other):
        """other + field_view (scalar, array, or field_view)."""
        if isinstance(other, _FieldView):
            other = other._get_data()
        return np.asarray(other, dtype=np.float32) + self._get_data()

    def __rsub__(self, other):
        """other - field_view (scalar or array)."""
        return np.asarray(other, dtype=np.float32) - self._get_data()

    def __rtruediv__(self, other):
        """other / field_view (scalar)."""
        return np.asarray(other, dtype=np.float32) / self._get_data()

    def __le__(self, other):
        """Comparison returns absolute entity indices where condition is true."""
        mask = (self._get_data() <= other).flatten()
        return self._indices[mask]

    def __lt__(self, other):
        mask = (self._get_data() < other).flatten()
        return self._indices[mask]

    def __ge__(self, other):
        mask = (self._get_data() >= other).flatten()
        return self._indices[mask]

    def __gt__(self, other):
        mask = (self._get_data() > other).flatten()
        return self._indices[mask]

    def __repr__(self):
        return (
            f"_FieldView({self._component_name}.{self._field_name}, {len(self._indices)} entities)"
        )


def _quat_multiply(q1, q2):
    """Multiply quaternions q1 * q2. Format: (x, y, z, w).

    Supports broadcasting: q1 can be (4,) and q2 can be (N, 4).
    """
    q1 = np.atleast_2d(q1)
    q2 = np.atleast_2d(q2)

    x1, y1, z1, w1 = q1[:, 0], q1[:, 1], q1[:, 2], q1[:, 3]
    x2, y2, z2, w2 = q2[:, 0], q2[:, 1], q2[:, 2], q2[:, 3]

    result = np.empty_like(q2)
    result[:, 0] = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2  # x
    result[:, 1] = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2  # y
    result[:, 2] = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2  # z
    result[:, 3] = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2  # w

    # Normalize to avoid drift
    norms = np.linalg.norm(result, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    result /= norms

    return result


# =============================================================================
# ComponentView - View into entity store for specific components
# =============================================================================


class ComponentView:
    """
    View into entity store for specific components.

    Read operations return CURRENT data.
    Write operations emit Update commands (never write directly).
    """

    def __init__(
        self, store: EntityStore, component_names: list[str], indices: np.ndarray, engine=None
    ):
        self._store = store
        self._component_names = component_names
        self._indices = indices
        self._len = len(indices)
        self._engine = engine

    def __len__(self) -> int:
        return self._len

    def __iter__(self):
        """Iterate over entity indices."""
        for idx in self._indices:
            yield idx

    def __getitem__(self, component_name) -> "ComponentAccessor":
        """Get accessor for reading/writing component data.

        Supports string name ('Cube') or class type (Cube)."""
        # Convert class to string name if needed
        if not isinstance(component_name, str):
            component_name = component_name.__name__

        accessor = ComponentAccessor(self._store, component_name, self._indices)

        # Pass command buffer if engine available
        if self._engine:
            accessor._commands = self._engine.commands

        # Try to find the component class for field lookup
        from manifoldx.ecs import _COMPONENT_REGISTRY

        if component_name in _COMPONENT_REGISTRY:
            comp_class = _COMPONENT_REGISTRY.get(component_name)
            if comp_class and hasattr(comp_class, "_component_start_idx"):
                accessor._component_class = comp_class

        return accessor

    def get_component_data(self, component_name: str) -> np.ndarray:
        """Get component data for all entities in this view."""
        return self._store.get_component_data(component_name, self._indices)


# =============================================================================
# ComponentAccessor - Read/write accessor for components
# =============================================================================


class ComponentAccessor:
    """
    Accessor for component data within a view.

    Supports:
    - Reading: returns current component data
    - Augmented assignment (+=, -=, *=, /=): emits Update command
    """

    def __init__(self, store: EntityStore, component_name: str, indices: np.ndarray):
        self._store = store
        self._component_name = component_name
        self._indices = indices
        self._component_class = None
        self._commands = None

    def __setattr__(self, name, value):
        """Handle assignment: accessor.pos = (x, y, z) queues update command."""
        if name.startswith("_"):
            super().__setattr__(name, value)
            return

        # If value is a _FieldView from += operation, ignore (it already queued its command)
        if isinstance(value, _FieldView):
            return

        # Create a temporary FieldView to resolve the field and set via command
        field_view = self.__getattr__(name)
        field_view._set_data(np.asarray(value, dtype=np.float32))

    def _queue_update(self, new_data):
        if self._commands is not None:
            from manifoldx.commands import Command, CommandType

            self._commands.append(
                Command(
                    CommandType.UPDATE_COMPONENT,
                    {
                        "component_name": self._component_name,
                        "indices": self._indices.copy(),
                        "new_data": new_data.copy(),
                    },
                )
            )
        else:
            self._store._components[self._component_name][self._indices] = new_data

    def __getattr__(self, name: str):
        """Access component fields by name: .position, .life, .velocity."""
        from manifoldx.ecs import _COMPONENT_REGISTRY

        if self._component_name not in self._store._components:
            raise ValueError(f"Component {self._component_name} not registered")

        # Check if it's Transform with built-in fields
        if self._component_name == "Transform":
            if name in ("position", "pos"):
                return _FieldView(
                    self._store, self._indices, "Transform", 0, 3, "position", self._commands
                )
            elif name in ("rotation", "rot"):
                return _FieldView(
                    self._store, self._indices, "Transform", 3, 7, "rotation", self._commands
                )
            elif name == "scale":
                return _FieldView(
                    self._store, self._indices, "Transform", 7, 10, "scale", self._commands
                )

        # Built-in viz scalar components (single-column floats).
        if self._component_name == "ScalarValue" and name == "value":
            return _FieldView(
                self._store, self._indices, "ScalarValue", 0, 1, "value", self._commands
            )
        if self._component_name == "Radius" and name == "radius":
            return _FieldView(
                self._store, self._indices, "Radius", 0, 1, "radius", self._commands
            )

        # A Component subclass's fields, from the layout its annotations give
        from manifoldx.components import _COMPONENT_CLASSES

        layout = getattr(_COMPONENT_CLASSES.get(self._component_name), "_layout", {})
        if name in layout:
            col, size = layout[name]
            return _FieldView(
                self._store,
                self._indices,
                self._component_name,
                col,
                col + size,
                name,
                self._commands,
            )

        # Check if component class has field positions
        comp_class = _COMPONENT_REGISTRY.get(self._component_name)
        if comp_class and hasattr(comp_class, "_component_start_idx"):
            field_info = comp_class._component_start_idx.get(name)
            if field_info:
                col, size = field_info
                return _FieldView(
                    self._store,
                    self._indices,
                    self._component_name,
                    col,
                    col + size,
                    name,
                    self._commands,
                )

        raise AttributeError(f"No field '{name}' in component '{self._component_name}'")

    @property
    def data(self) -> np.ndarray:
        """Get raw component data as numpy array."""
        return self._store.get_component_data(self._component_name, self._indices)

    def __getitem__(self, key):
        """Read: returns current data for entities at indices."""
        return self._store.get_component_data(self._component_name, self._indices)

    def __setitem__(self, key, value):
        """Write: NOT ALLOWED - use augmented assignment instead."""
        raise NotImplementedError("Use +=, -=, *=, /= for updates")

    def __iadd__(self, other):
        """Augmented assignment: computes new_value = current + other."""
        current = self._store.get_component_data(self._component_name, self._indices)

        if np.isscalar(other):
            new_data = current + other
        else:
            new_data = current + other

        self._queue_update(new_data)
        return self

    def __isub__(self, other):
        current = self._store.get_component_data(self._component_name, self._indices)
        new_data = current - other
        self._queue_update(new_data)
        return self

    def __imul__(self, other):
        current = self._store.get_component_data(self._component_name, self._indices)
        new_data = current * other
        self._queue_update(new_data)
        return self

    def __itruediv__(self, other):
        current = self._store.get_component_data(self._component_name, self._indices)
        new_data = current / other
        self._queue_update(new_data)
        return self


# =============================================================================
# Global Component Registry
# =============================================================================

# Registry of component classes created via `@engine.component` /
# `_make_component_class`. Keyed by class name; value is the class itself.
# Read by ComponentAccessor / ComponentView for field lookup.
_COMPONENT_REGISTRY: dict[str, type] = {}


def _infer_shape_from_type(tp) -> tuple:
    """Infer array shape from type hint."""
    from manifoldx.types import Float, Vector3, Vector4

    if tp == Float or tp == float:
        return ()  # Scalar
    elif tp == Vector3:
        return (3,)
    elif tp == Vector4:
        return (4,)
    else:
        return ()  # Default to scalar


def _make_component_class(cls: type, engine) -> type:
    """Create a constructable component class that registers with engine's store."""
    from manifoldx.types import Float, Vector3, Vector4

    annotations = cls.__annotations__
    field_names = list(annotations.keys())
    field_shapes = {name: _infer_shape_from_type(annotations[name]) for name in field_names}

    # Register the component in engine's store
    if engine and engine.store:
        total_cols = sum(int(np.prod(s) if s else 1) for s in field_shapes.values())
        engine.store.register_component(cls.__name__, np.dtype("f4"), (total_cols,))

    # Also register in global registry for field lookup
    _COMPONENT_REGISTRY[cls.__name__] = cls

    # Create field position index
    _component_start_idx = {}
    col_offset = 0
    for field_name in field_names:
        shape = field_shapes.get(field_name, ())
        size = int(np.prod(shape)) if shape else 1
        _component_start_idx[field_name] = (col_offset, size)
        col_offset += size

    def __init__(self, **kwargs):
        self._field_values = kwargs

    def get_data(self, n: int, registry=None):
        total_size = sum(int(np.prod(field_shapes.get(f, (1,)))) for f in field_names)
        data = np.zeros((n, total_size), dtype=np.float32)

        col_offset = 0
        for field_name in field_names:
            value = self._field_values.get(field_name)
            shape = field_shapes.get(field_name, ())
            size = int(np.prod(shape)) if shape else 1

            if value is not None:
                value = np.asarray(value, dtype=np.float32)
                if value.shape == ():
                    # Scalar: broadcast to all n rows
                    data[:, col_offset : col_offset + size] = value.item()
                elif len(value.shape) == 1 and value.shape[0] == n and size == 1:
                    # (n,) array for scalar field: reshape to (n, 1)
                    data[:, col_offset : col_offset + size] = value.reshape(n, 1)
                elif len(value.shape) == 1 and value.shape[0] == size:
                    # 1D array matching field size: broadcast to all n rows
                    data[:, col_offset : col_offset + size] = value
                elif value.shape[0] == n:
                    # (n, size) or (n,) matching: reshape if needed
                    if len(value.shape) == 1:
                        value = value.reshape(n, -1)
                    data[:, col_offset : col_offset + size] = value[:, :size]
                else:
                    data[:, col_offset : col_offset + size] = value.flatten()[:size]

            col_offset += size

        return data

    cls.__init__ = __init__
    cls.get_data = get_data
    cls._component_name = cls.__name__
    cls._component_fields = field_names
    cls._component_start_idx = _component_start_idx

    return cls


__all__ = [
    "EntityStore",
    "ComponentView",
    "ComponentAccessor",
    "_COMPONENT_REGISTRY",
]
