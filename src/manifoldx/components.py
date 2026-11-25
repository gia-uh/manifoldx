"""Built-in components: Component (base), Transform, Mesh, Material."""

import inspect
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from manifoldx.compute.shader import vec3, vec4


# =============================================================================
# Component Base Class
# =============================================================================

# Registry of all Component subclasses, keyed by class name. Populated by
# Component.__init_subclass__ at class-definition time. The Engine consults
# this on spawn so callers don't have to register components by hand.
_COMPONENT_CLASSES: dict = {}


def _shape_from_annotation(tp) -> tuple:
    """Map a type-marker annotation to a per-entity numpy shape.

    Accepts both the runtime numpy types (`Float`, `Vector3`, `Vector4`
    from `manifoldx.types`) and the Phase-2 DSL types (`vec3`, `vec4`
    from `manifoldx.compute.shader`). The DSL forms type-check cleanly
    against compute-kernel arithmetic; the numpy forms predate Phase 2
    and stay supported.
    """
    from manifoldx.compute.shader import f32 as _f32
    from manifoldx.compute.shader import vec3 as _vec3
    from manifoldx.compute.shader import vec4 as _vec4
    from manifoldx.types import Float, Vector3, Vector4

    if tp is Float or tp is float or tp is _f32:
        return (1,)
    if tp is Vector3 or tp is _vec3:
        return (3,)
    if tp is Vector4 or tp is _vec4:
        return (4,)
    raise TypeError(
        f"Unsupported component field type: {tp!r}. "
        f"Use Float, Vector3/Vector4 (manifoldx.types), or vec3/vec4 "
        f"(manifoldx.compute.shader)."
    )


class Component:
    """Base class for ECS components with annotation-driven storage.

    Subclasses declare per-entity storage via class-level annotations,
    pydantic-style:

        class Radius(Component):
            radius: Float = 1.0          # 1 float per entity, default 1.0

        class Velocity(Component):
            velocity: Vector3            # 3 floats per entity, default 0

        class PointCloud(Component):
            \"\"\"Marker — no fields.\"\"\"   # zero-width

    On subclass creation `__init_subclass__` reads the annotations,
    computes total dtype + shape, and registers the class so the Engine
    can auto-register it with the store on first spawn.

    The default `__init__` accepts each annotated field as a keyword
    argument; the default `get_data` broadcasts scalars across all
    entities and accepts (n,) or (n, size) arrays for per-entity values.
    Subclasses may override either method for richer semantics.
    """

    _dtype: np.dtype = np.dtype("f4")
    _shape: tuple = (0,)
    _field_specs: tuple = ()
    _field_defaults: dict = {}
    _layout: dict = {}

    _gpu_only: bool = False

    def __init_subclass__(cls, *, gpu_only: bool = False, **kwargs):
        super().__init_subclass__(**kwargs)
        cls._gpu_only = bool(gpu_only)
        # inspect.get_annotations, not __dict__["__annotations__"]: Python 3.14
        # evaluates class annotations lazily and leaves that key unset.
        annotations = inspect.get_annotations(cls)
        fields = []
        defaults = {}
        for name, tp in annotations.items():
            if name.startswith("_"):
                continue
            shape = _shape_from_annotation(tp)
            fields.append((name, shape))
            default = cls.__dict__.get(name)
            if default is not None:
                defaults[name] = default
        cls._field_specs = tuple(fields)
        cls._field_defaults = defaults
        total = sum(int(np.prod(s)) for _, s in fields)
        cls._shape = (total,)
        # Derive per-field offset table for the Phase-2 transpiler. Each
        # field maps to (offset_in_floats, length_in_floats) within the
        # interleaved per-entity row.
        layout: dict[str, tuple[int, int]] = {}
        offset = 0
        for name, shape in fields:
            length = int(np.prod(shape))
            layout[name] = (offset, length)
            offset += length
        cls._layout = layout
        _COMPONENT_CLASSES[cls.__name__] = cls

    def __init__(self, **kwargs):
        # Each declared field accepts a keyword. Missing fields fall back
        # to the class-level default (or None → all-zeros).
        self._field_values = {}
        for name, _shape in self._field_specs:
            v = kwargs.get(name)
            if v is None:
                v = self._field_defaults.get(name)
            self._field_values[name] = v

    def get_data(self, n: int, registry=None) -> np.ndarray:
        """Pack this component's per-entity data into an (n, total_cols) array.

        Each declared field is filled by broadcasting:
        - scalar value → all rows get the scalar
        - (size,) array → all rows get the same vector (per-field broadcast)
        - (n,) array (size==1) → one value per row, reshaped to (n, 1)
        - (n, size) array → one row per entity verbatim
        Anything else raises ValueError.
        """
        cols = int(np.prod(self._shape)) if self._shape else 0
        data = np.zeros((n, cols), dtype=self._dtype)
        if cols == 0:
            return data
        col = 0
        for name, shape in self._field_specs:
            size = int(np.prod(shape))
            value = self._field_values.get(name)
            if value is not None:
                v = np.asarray(value, dtype=data.dtype)
                if v.ndim == 0:
                    data[:, col : col + size] = v.item()
                elif v.ndim == 1 and v.shape[0] == n and size == 1:
                    data[:, col : col + size] = v.reshape(n, 1)
                elif v.ndim == 1 and v.shape[0] == size:
                    data[:, col : col + size] = v
                elif v.shape[0] == n:
                    if v.ndim == 1:
                        v = v.reshape(n, -1)
                    data[:, col : col + size] = v[:, :size]
                else:
                    raise ValueError(
                        f"{type(self).__name__}: field {name!r} value shape "
                        f"{v.shape} incompatible with n={n}, field size={size}"
                    )
            col += size
        return data

    @classmethod
    def register(cls, store) -> None:
        """Register this component type with an EntityStore (idempotent)."""
        if cls.__name__ in store._components:
            return
        store.register_component(cls.__name__, cls._dtype, cls._shape)


# =============================================================================
# Transform Component
# =============================================================================


class Transform:
    """
    Transform component storing position, rotation, scale.

    Usage:
        Transform()                    # Default transform
        Transform(pos=(0,0,0))         # With position
        Transform(pos=(0,0,0), scale=(1,1,1))  # With position and scale

    Storage layout (10 floats per entity):
    - position: vec3 (columns 0-2)
    - rotation: vec4 quaternion (columns 3-6)
    - scale: vec3 (columns 7-9)

    Default values:
    - position: (0, 0, 0)
    - rotation: (0, 0, 0, 1) - identity quaternion
    - scale: (1, 1, 1)
    """

    # Combined shape: position(3) + rotation(4) + scale(3) = 10 floats
    _layout = {"pos": (0, 3), "rot": (3, 4), "scale": (7, 3)}

    if TYPE_CHECKING:
        # Field stubs for Phase-2 compute kernels: `self.transforms[i].pos`
        # type-checks against these. Runtime data lives in the SoA store,
        # not on instances — these annotations exist solely so kernel
        # bodies (which never run as Python) read coherently to mypy.
        pos: vec3
        rot: vec4
        scale: vec3

    def __init__(self, pos=None, rot=None, scale=None):
        """Create Transform with optional position/rotation/scale."""
        self._pos = pos
        self._rot = rot
        self._scale = scale

    def get_data(self, n: int, registry=None) -> np.ndarray:
        """Get Transform data array for n entities."""
        data = np.zeros((n, 10), dtype=np.float32)

        # Set defaults
        data[:, 7:10] = 1.0  # scale = (1, 1, 1)
        data[:, 6] = 1.0  # rotation w = 1 (quaternion identity)

        # Apply user-provided values
        if self._pos is not None:
            pos = np.asarray(self._pos, dtype=np.float32)
            data[:, 0:3] = pos

        if self._rot is not None:
            rot = np.asarray(self._rot, dtype=np.float32)
            data[:, 3:7] = rot

        if self._scale is not None:
            scale = np.asarray(self._scale, dtype=np.float32)
            data[:, 7:10] = scale

        # Broadcast to n entities if needed
        if n > 1 and (self._pos is not None or self._rot is not None or self._scale is not None):
            # If single values provided, broadcast them
            if self._pos is not None:
                data[:, 0:3] = np.broadcast_to(self._pos, (n, 3))
            if self._rot is not None:
                data[:, 3:7] = np.broadcast_to(self._rot, (n, 4))
            if self._scale is not None:
                data[:, 7:10] = np.broadcast_to(self._scale, (n, 3))

        return data

    @staticmethod
    def rotation(x=0, y=0, z=0, euler=None) -> np.ndarray:
        """Create quaternion(s) (x, y, z, w) from euler angles (radians).

        Uses intrinsic Tait-Bryan angles: rotate around X, then Y, then Z.

        Accepts scalars or arrays:
            Transform.rotation(x=0, y=0.5, z=0)          -> (4,) quaternion
            Transform.rotation(euler=angular_vel * dt)     -> (N, 4) quaternions
            Transform.rotation(x=np.array([...]), y=...) -> (N, 4) quaternions

        The `euler` parameter is an (N, 3) or (3,) array of [x, y, z] angles.
        """
        if euler is not None:
            euler = np.asarray(euler, dtype=np.float32)
            if euler.ndim == 1:
                x, y, z = euler[0], euler[1], euler[2]
            else:
                x, y, z = euler[:, 0], euler[:, 1], euler[:, 2]

        cx, sx = np.cos(x / 2), np.sin(x / 2)
        cy, sy = np.cos(y / 2), np.sin(y / 2)
        cz, sz = np.cos(z / 2), np.sin(z / 2)

        # Quaternion from euler (XYZ order)
        qx = sx * cy * cz - cx * sy * sz
        qy = cx * sy * cz + sx * cy * sz
        qz = cx * cy * sz - sx * sy * cz
        qw = cx * cy * cz + sx * sy * sz

        return np.stack([qx, qy, qz, qw], axis=-1).astype(np.float32)

    @staticmethod
    def register(store):
        """Register Transform component in entity store."""
        store.register_component("Transform", np.dtype("f4"), shape=(10,))

    @staticmethod
    def get_default_data(n: int) -> np.ndarray:
        """Get default Transform data for n entities."""
        # Default: position=(0,0,0), rotation=(0,0,0,1), scale=(1,1,1)
        data = np.zeros((n, 10), dtype=np.float32)
        data[:, 7:10] = 1.0  # Default scale to (1, 1, 1)
        # Default rotation is (0, 0, 0, 1) - already zeros with last element 1
        data[:, 6] = 1.0  # w component of quaternion
        return data


# =============================================================================
# Mesh Component
# =============================================================================


class Mesh:
    """
    Mesh component storing reference to geometry.

    Usage:
        Mesh(cube_geometry)  # Creates component data with geometry_id

    Storage: Single uint32 (geometry_id)
    """

    _layout = {"geometry_id": (0, 1)}

    def __init__(self, geometry):
        """Create Mesh component with geometry."""
        # Store geometry object for later registration
        self._geometry = geometry
        self._geometry_id = None

    def get_data(self, n: int, registry) -> np.ndarray:
        """Get component data array for n entities."""
        # Register geometry if not already, or again after release() freed it
        if self._geometry_id is None or registry.get(self._geometry_id) is None:
            self._geometry_id = registry.register(self._geometry)

        # Create array with geometry_id repeated n times
        return np.full((n, 1), self._geometry_id, dtype=np.uint32)

    @staticmethod
    def register(store):
        """Register Mesh component in entity store."""
        store.register_component("Mesh", np.dtype("u4"), shape=(1,))

    @staticmethod
    def get_default_data(n: int) -> np.ndarray:
        """Get default Mesh data (0 = no geometry)."""
        return np.zeros((n, 1), dtype=np.uint32)


# =============================================================================
# Material Component
# =============================================================================


class Material:
    """
    Material component storing reference to material.

    Usage:
        Material(phong_material)  # Creates component data with material_id

    Storage: Single uint32 (material_id)
    """

    _layout = {"material_id": (0, 1)}

    def __init__(self, material):
        """Create Material component with material."""
        self._material = material
        self._material_id = None

    def get_data(self, n: int, registry) -> np.ndarray:
        """Get component data array for n entities."""
        # Register material if not already
        if self._material_id is None:
            self._material_id = registry.register(self._material)

        return np.full((n, 1), self._material_id, dtype=np.uint32)

    @staticmethod
    def register(store):
        """Register Material component in entity store."""
        store.register_component("Material", np.dtype("u4"), shape=(1,))

    @staticmethod
    def get_default_data(n: int) -> np.ndarray:
        """Get default Material data (0 = no material)."""
        return np.zeros((n, 1), dtype=np.uint32)


# =============================================================================
# Built-in Colors
# =============================================================================


class Colors:
    """Color constants."""

    RED = "#ff0000"
    GREEN = "#00ff00"
    BLUE = "#0000ff"
    WHITE = "#ffffff"
    BLACK = "#000000"
    YELLOW = "#ffff00"
    CYAN = "#00ffff"
    MAGENTA = "#ff00ff"


__all__ = [
    "Component",
    "Transform",
    "Mesh",
    "Material",
    "Colors",
]


from manifoldx.types import Float  # noqa: E402


class Layer(Component):
    """The render layer an entity is on; entities without one are on layer 0.

    A camera draws the entities whose layer is in its `layers` set (None: all).
    Put a render_to_texture stage on its own layer and give its camera only that
    layer, and the view's cost follows the stage, not the world:

        engine.spawn(Mesh(head), Material(skin), Transform(...), Layer(layer=1))
        portrait_camera.layers = {1}
        engine.camera.layers = {0}   # the main camera skips the stage
    """

    layer: Float = 0.0


from manifoldx.types import Vector3  # noqa: E402


class Tint(Component):
    """A colour this entity's mesh is multiplied by, per instance; entities without one keep their colours.

    Only a StandardMaterial made with tinted=True reads it, so a thousand trees of one geometry and one
    material stay one instanced draw while each wears its own shade:

        leaf = StandardMaterial("#ffffff", vertex_colors=True, tinted=True)
        engine.spawn(Mesh(crown), Material(leaf), Transform(pos=spots), Tint(rgb=shades), n=len(spots))

    The colour is linear, like the shader's, and multiplies the vertex colour.
    """

    rgb: Vector3 = (1.0, 1.0, 1.0)


class Palette(Component):
    """Eight sRGB colours this entity's mesh wears by part, per instance; entities without one wear white.

    Only a StandardMaterial made with paletted=True reads it. Each vertex of surface PALETTE_SURFACE + k
    takes colour k times its own colour, so one figure geometry shared by a crowd dresses each walker
    differently in one instanced draw. A Figure's paletted_geometry() lays its slots out this way
    (skin, top, bottom, stripe, shoes, hair, hat; see human_dress.SLOTS) and its palette() gives the
    colours it was dressed in:

        mat = StandardMaterial("#ffffff", vertex_colors=True, paletted=True)
        engine.spawn(Mesh(fig.paletted_geometry()), Material(mat), Transform(pos=spots),
                     Palette(c1=shirts, c2=trousers), n=len(spots))

    The colours are sRGB, like a hex colour or a vertex colour.
    """

    c0: Vector3 = (1.0, 1.0, 1.0)
    c1: Vector3 = (1.0, 1.0, 1.0)
    c2: Vector3 = (1.0, 1.0, 1.0)
    c3: Vector3 = (1.0, 1.0, 1.0)
    c4: Vector3 = (1.0, 1.0, 1.0)
    c5: Vector3 = (1.0, 1.0, 1.0)
    c6: Vector3 = (1.0, 1.0, 1.0)
    c7: Vector3 = (1.0, 1.0, 1.0)


class Baked(Component):
    """Which layer of its material's baked texture array this entity's SURFACE_BAKED vertices draw, per
    instance; -1 (or no Baked) draws those vertices' own colours.

    Only a StandardMaterial made with baked= reads it. QuadStreamer gives each node it draws with an image
    a layer of its own, so every node of one material wears its own picture:

        tex = texture_array(engine, layers=64, size=512)
        mat = StandardMaterial("#ffffff", vertex_colors=True, baked=tex)
        write_layer(engine, tex, 3, layer_mips(image))
        engine.spawn(Mesh(node), Material(mat), Transform(pos=corner), Baked(layer=3))
    """

    layer: Float = -1.0


class Fade(Component):
    """How much of this entity's mesh draws, per instance: 1 all of it (as without a Fade), 0 none.

    Only a StandardMaterial made with faded=True reads it. The mesh drops pixels against a 4x4 pattern
    rather than blending, so nothing is sorted and the depth stays right; its shadow drops the same
    share. A negative fade draws the complementary pixels: an entity at -(1 - t) and one at t cover
    every pixel exactly once, so a coarse level of detail can give way to a fine one without a pop:

        lod = StandardMaterial("#ffffff", vertex_colors=True, faded=True)
        engine.spawn(Mesh(coarse), Material(lod), Transform(), Fade(value=-0.7))
        engine.spawn(Mesh(fine), Material(lod), Transform(), Fade(value=0.3))
    """

    value: Float = 1.0
