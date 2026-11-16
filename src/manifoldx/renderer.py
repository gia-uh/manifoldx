"""Render pipeline that executes after command buffer."""

import numpy as np
import wgpu

from manifoldx.render.passes import axis as _axis_pass
from manifoldx.render.passes import label as _label_pass
from manifoldx.render.passes import mesh as _mesh_pass
from manifoldx.render.passes import skybox as _skybox_pass
from manifoldx.render.passes import sprite as _sprite_pass
from manifoldx.render.passes import volume as _volume_pass
from manifoldx.render.passes.volume import VOLUME_SHADER_SOURCE
from manifoldx.skinning import (
    BLOCKS_BINDING,
    JOINT_LOCATION,
    JOINTS_BINDING,
    JointStore,
    skin_shader,
)
from manifoldx.viz.materials import ColormapMaterial


def _glow_target(fmt):
    """Alpha-blended color target for the 'glow' (atmosphere) material."""
    return {
        "format": fmt,
        "blend": {
            "color": {"src_factor": wgpu.BlendFactor.src_alpha,
                      "dst_factor": wgpu.BlendFactor.one_minus_src_alpha,
                      "operation": wgpu.BlendOperation.add},
            "alpha": {"src_factor": wgpu.BlendFactor.one,
                      "dst_factor": wgpu.BlendFactor.one_minus_src_alpha,
                      "operation": wgpu.BlendOperation.add},
        },
    }


def _scatter_target(fmt):
    """Additive color target for the 'scatter' (single-scattering atmosphere) material."""
    return {
        "format": fmt,
        "blend": {
            "color": {"src_factor": wgpu.BlendFactor.one,
                      "dst_factor": wgpu.BlendFactor.one,
                      "operation": wgpu.BlendOperation.add},
            "alpha": {"src_factor": wgpu.BlendFactor.one,
                      "dst_factor": wgpu.BlendFactor.one,
                      "operation": wgpu.BlendOperation.add},
        },
    }


MAX_BENDERS = 8  # the points swaying meshes part around (Engine.benders)

# Depth-only shadow-pass shader: transforms mesh geometry by the sun's
# light_view_proj into the shadow map. No fragment stage (depth writes only).
# Binds the shared Globals uniform (for light_view_proj) + a transforms storage.
_SHADOW_SHADER = """
struct Globals {
    vp:              mat4x4<f32>,
    view:            mat4x4<f32>,
    proj:            mat4x4<f32>,
    camera_pos:      vec3<f32>,
    _pad0:           f32,
    viewport_size:   vec2<f32>,
    _pad1:           vec2<f32>,
    ibl_intensity:   f32,
    ibl_enabled:     u32,
    _pad_ibl:        vec2<f32>,
    light_view_proj: mat4x4<f32>,
    sun_direction:   vec3<f32>,
    _pad_sun0:       f32,
    sun_color:       vec3<f32>,
    sun_intensity:   f32,
    shadow_enabled:  u32,
    shadow_bias:     f32,
    shadow_map_size: f32,
    shadow_pcf_radius: u32,
    spot_position:   vec3<f32>,
    spot_range:      f32,
    spot_direction:  vec3<f32>,
    spot_cos_inner:  f32,
    spot_color:      vec3<f32>,
    spot_intensity:  f32,
    spot_cos_outer:  f32,
    shadow_caster:   u32,
    _pad_spot0:      f32,
    _pad_spot1:      f32,
};

struct Transforms {
    models: array<mat4x4<f32>>,
};

@group(0) @binding(0) var<uniform> globals: Globals;
@group(0) @binding(1) var<storage, read> transforms: Transforms;

@vertex
fn vs_main(@location(0) position: vec3<f32>,
           @builtin(instance_index) instance: u32) -> @builtin(position) vec4<f32> {
    let model = transforms.models[instance];
    let world = (model * vec4<f32>(position, 1.0)).xyz;
    return globals.light_view_proj * vec4<f32>(world, 1.0);
}
"""


# =============================================================================
# Batch Buffers (per-batch GPU buffer management)
# =============================================================================


def vertex_bounds(positions, skinned=False):
    """A sphere (centre (3,), radius) around vertex positions, padded for a skinned
    geometry whose joints move them."""
    pos = np.asarray(positions, dtype=np.float64).reshape(-1, 3)
    lo, hi = pos.min(axis=0), pos.max(axis=0)
    c = (lo + hi) / 2
    r = float(np.linalg.norm(pos - c, axis=1).max())
    return c, (r * 1.5 + 0.5 if skinned else r)


def _groups(idx, *keys):
    """[(key tuple, local indices)] per distinct key over `idx` (ascending local indices), each
    key read from the `keys` arrays. Groups come in order of their first index, indices
    ascending within each, as a per-entity loop building dicts would give."""
    if not len(idx):
        return []
    order = idx[np.lexsort([k[idx] for k in reversed(keys)])]  # stable: index order kept
    split = np.flatnonzero(np.any([np.diff(k[order]) != 0 for k in keys], axis=0)) + 1
    out = [(tuple(int(k[v[0]]) for k in keys), v) for v in np.split(order, split)]
    return sorted(out, key=lambda kv: kv[1][0])


class MaterialGeometryMismatchError(ValueError):
    """A material requires geometry attributes (e.g. UVs) the geometry doesn't have."""


def _box_corners(points):
    """(8, 4) homogeneous corners of the points' axis-aligned box."""
    lo, hi = points.min(axis=0), points.max(axis=0)
    return np.array(
        [[x, y, z, 1.0] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])],
        dtype=np.float32,
    )


class _BatchBuffers:
    """Per-batch GPU buffers managed by RenderPipeline.

    Extends transform-only management to include optional per-instance
    scalar_values and radii buffers required by ColormapMaterial.
    """

    def __init__(self, device):
        self._device = device
        self.transforms_buf = None
        self.transforms_capacity = 0
        self.scalar_values_buf = None
        self.scalar_values_capacity = 0
        self.radii_buf = None
        self.radii_capacity = 0
        self.label_indices_buf = None
        self.label_indices_capacity = 0
        self.joint_blocks_buf = None
        self.joint_blocks_capacity = 0

    def upload_transforms(self, data: "np.ndarray"):
        n_bytes = data.nbytes
        if self.transforms_buf is None or self.transforms_capacity < n_bytes:
            self.transforms_buf = self._device.create_buffer(
                size=n_bytes,
                usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST,
            )
            self.transforms_capacity = n_bytes
        self._device.queue.write_buffer(self.transforms_buf, 0, data.tobytes())

    def upload_scalar_values(self, data: "np.ndarray"):
        n_bytes = data.nbytes
        if self.scalar_values_buf is None or self.scalar_values_capacity < n_bytes:
            self.scalar_values_buf = self._device.create_buffer(
                size=n_bytes,
                usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST,
            )
            self.scalar_values_capacity = n_bytes
        self._device.queue.write_buffer(self.scalar_values_buf, 0, data.tobytes())

    def upload_radii(self, data: "np.ndarray"):
        n_bytes = data.nbytes
        if self.radii_buf is None or self.radii_capacity < n_bytes:
            self.radii_buf = self._device.create_buffer(
                size=n_bytes,
                usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST,
            )
            self.radii_capacity = n_bytes
        self._device.queue.write_buffer(self.radii_buf, 0, data.tobytes())

    def upload_joint_blocks(self, data: "np.ndarray"):
        n_bytes = data.nbytes
        if self.joint_blocks_buf is None or self.joint_blocks_capacity < n_bytes:
            self.joint_blocks_buf = self._device.create_buffer(
                size=n_bytes,
                usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST,
            )
            self.joint_blocks_capacity = n_bytes
        self._device.queue.write_buffer(self.joint_blocks_buf, 0, data.tobytes())

    def upload_label_indices(self, data: "np.ndarray"):
        n_bytes = data.nbytes
        if self.label_indices_buf is None or self.label_indices_capacity < n_bytes:
            self.label_indices_buf = self._device.create_buffer(
                size=n_bytes,
                usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST,
            )
            self.label_indices_capacity = n_bytes
        self._device.queue.write_buffer(self.label_indices_buf, 0, data.tobytes())


# =============================================================================
# WGSL Shaders (default fallback - not used when materials provide shaders)
# =============================================================================


SHADER_SOURCE = """
struct Globals {
    vp: mat4x4<f32>,
    view: mat4x4<f32>,
    camera_pos: vec3<f32>,
    _pad: f32,
};

struct Transforms {
    models: array<mat4x4<f32>>,
};

@group(0) @binding(0) var<uniform> globals: Globals;
@group(0) @binding(1) var<storage, read> transforms: Transforms;

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @builtin(instance_index) instance: u32,
};

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) world_normal: vec3<f32>,
};

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    let model = transforms.models[in.instance];
    let world_pos = (model * vec4<f32>(in.position, 1.0)).xyz;
    out.position = globals.vp * vec4<f32>(world_pos, 1.0);
    // Transform normal by model matrix (ignore translation, assume uniform scale)
    out.world_normal = (model * vec4<f32>(in.normal, 0.0)).xyz;
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let normal = normalize(in.world_normal);
    let light_dir = normalize(vec3<f32>(0.5773, 0.5773, 0.5773));
    let diffuse = max(dot(normal, light_dir), 0.0);
    let brightness = 0.3 + 0.7 * diffuse;
    return vec4<f32>(vec3<f32>(0.8) * brightness, 1.0);
}
"""


# =============================================================================
# Transform Cache (With Dirty Flag Optimization)
# =============================================================================


class TransformCache:
    """
    Caches computed transform matrices.

    Stores: _matrix_cache (N, 16) - mat4x4<f32> per entity
            _dirty (N,) bool - needs recompute
            _seen (N, 10) - the Transform row each matrix was computed from

    refresh() compares the store's Transform rows with `_seen` and recomputes the rows that differ, so a
    write reaches the next frame whatever made it (a query, a handle, the array itself) and a static
    entity costs a comparison, not a matrix.
    """

    def __init__(self, max_entities: int):
        self._matrix_cache = np.zeros((max_entities, 16), dtype=np.float32)
        self._dirty = np.ones(max_entities, dtype=bool)  # All dirty initially
        self._seen = np.full((max_entities, 10), np.nan, dtype=np.float32)  # NaN: never computed
        self.version = 0  # counts the recomputes; _row_version holds the one that last touched each row
        self._row_version = np.zeros(max_entities, dtype=np.int64)

    def changed_since(self, version: int, indices: np.ndarray) -> bool:
        """Whether any of these rows was recomputed after `version`."""
        return len(indices) > 0 and int(self._row_version[indices].max()) > version

    def refresh(self, store):
        """Recompute the matrices of the alive rows whose Transform changed since they were computed."""
        hw = store._high_water
        if hw == 0:
            return
        data = store._components["Transform"][:hw]
        changed = (self._dirty[:hw] | (data != self._seen[:hw]).any(axis=1)) & store._alive[:hw]
        idx = np.flatnonzero(changed)
        if len(idx):
            self._recompute_matrices(store, idx)

    def mark_dirty(self, indices: np.ndarray):
        """Mark entities as needing recompute."""
        self._dirty[indices] = True

    def get_transforms(self, store, indices: np.ndarray) -> np.ndarray:
        """
        Get transform matrices for indices, recomputing if dirty.
        Returns (len(indices), 16) array of mat4x4<f32>.
        """
        dirty_mask = self._dirty[indices]
        if dirty_mask.any():
            self._recompute_matrices(store, indices[dirty_mask])

        return self._matrix_cache[indices]

    def _recompute_matrices(self, store, indices: np.ndarray):
        """Compute mat4x4 from pos/rot/scale using vectorized numpy."""
        if len(indices) == 0:
            return

        transform_data = store.get_component_data("Transform", indices)

        positions = transform_data[:, 0:3]  # (N, 3)
        quats = transform_data[:, 3:7]  # (N, 4) - quaternion (x, y, z, w)
        scales = transform_data[:, 7:10]  # (N, 3)

        n = len(indices)
        matrices = np.zeros((n, 16), dtype=np.float32)

        # Extract quaternion components
        qx, qy, qz, qw = quats[:, 0], quats[:, 1], quats[:, 2], quats[:, 3]

        xx, yy, zz = qx * qx, qy * qy, qz * qz
        xy, xz, yz = qx * qy, qx * qz, qy * qz
        wx, wy, wz = qw * qx, qw * qy, qw * qz

        sx, sy, sz = scales[:, 0], scales[:, 1], scales[:, 2]

        # Row 0 (scaled)
        matrices[:, 0] = (1 - 2 * (yy + zz)) * sx
        matrices[:, 1] = 2 * (xy - wz) * sy
        matrices[:, 2] = 2 * (xz + wy) * sz
        # Row 1 (scaled)
        matrices[:, 4] = 2 * (xy + wz) * sx
        matrices[:, 5] = (1 - 2 * (xx + zz)) * sy
        matrices[:, 6] = 2 * (yz - wx) * sz
        # Row 2 (scaled)
        matrices[:, 8] = 2 * (xz - wy) * sx
        matrices[:, 9] = 2 * (yz + wx) * sy
        matrices[:, 10] = (1 - 2 * (xx + yy)) * sz
        # Row 3
        matrices[:, 15] = 1.0
        # Translation stored at ROW 0, COLUMN 3 (indices 3, 7, 11)
        # This way, after transpose to column-major, it ends up at
        # COLUMN 3 (indices 12, 13, 14) where WGSL expects it
        matrices[:, 3] = positions[:, 0]
        matrices[:, 7] = positions[:, 1]
        matrices[:, 11] = positions[:, 2]

        # Validate computed matrices
        from manifoldx.ecs import ENABLE_VALIDATION

        if ENABLE_VALIDATION:
            if np.any(np.isnan(matrices)):
                import warnings

                warnings.warn(f"⚠️ NaN in transform matrices for {len(indices)} entities")
            if np.any(np.isinf(matrices)):
                import warnings

                warnings.warn(f"⚠️ Inf in transform matrices for {len(indices)} entities")

        self._matrix_cache[indices] = matrices
        self._dirty[indices] = False
        self._seen[indices] = transform_data
        self.version += 1
        self._row_version[indices] = self.version


def _color_hex_to_vec4(color_hex: str) -> np.ndarray:
    """Convert hex color string to vec4 float array."""
    color_hex = color_hex.lstrip("#")
    r = int(color_hex[0:2], 16) / 255.0
    g = int(color_hex[2:4], 16) / 255.0
    b = int(color_hex[4:6], 16) / 255.0
    return np.array([r, g, b, 1.0], dtype=np.float32)


# =============================================================================
# Render Pipeline
# =============================================================================


class RenderPipeline:
    """
    Instanced render pipeline.

    Groups entities by (geometry_id, material_type) into batches.
    Each batch uses a single draw_indexed call with instance_count.
    Per-instance transforms are uploaded via a storage buffer.
    Material-type specific pipelines are cached.
    """

    def __init__(self, store, device=None):
        self._store = store
        self._device = device
        self._transform_cache = TransformCache(store.max_entities)
        # Between run() and end_frame() the store does not change: matrices and shadow bounds
        # are computed once there and reused by every pass.
        self._in_frame = False
        self._frame_bounds = None  # (bounds,) once _scene_bounds ran this frame
        self._pipelines = {}  # (geometry_id, material_type) -> pipeline
        self._pipeline_layouts = {}  # material_type -> layout
        self._bind_group_layouts = {}  # material_type -> bind group layout
        self._material_buffers = {}  # (geometry_id, material_type) -> buffer
        # Per-material-id uniform buffer for the mesh pass. Distinct from
        # _material_buffers above, which is keyed by pipeline-cache key —
        # that one shares a buffer across all material instances of the same
        # type, which silently broke per-instance material data. The mesh
        # pass now uses this dict keyed by mat_id, one buffer per material.
        self._material_buffers_by_mat_id = {}  # mat_id -> buffer
        self._material_uniforms_by_mat_id = {}  # mat_id -> bytes last written to its buffer
        # (mat_id, has_joints) -> (resources, bind group): rebuilt only when a
        # bound resource changes (a grown buffer, a swapped texture).
        self._mesh_bind_groups = {}
        self._shadow_bind_group = None  # (view, bind group) for group(2)
        self._geom_bounds = {}  # geom_id -> (centre (3,), radius): a sphere around its rest vertices
        self.drawn = 0  # mesh instances the last frame drew, after frustum culling
        self._globals_buffer = None  # VP matrix + view matrix + camera_pos
        self._batch_buffers = None  # _BatchBuffers instance for per-batch storage
        self._sprite_batch_buffers = None  # _BatchBuffers instance for sprite per-batch storage
        self._label_batch_buffers = None  # _BatchBuffers instance for label per-batch storage
        self._axis_batch_buffers = None  # _BatchBuffers instance for axis per-batch storage
        self._lights_buffer = None  # External lights uniform buffer
        self._lut_textures = {}  # cmap_name -> (texture, sampler) for LUT caching
        self._shader_sources = {}  # pipeline key -> WGSL the mesh pipeline was built from
        self._joints = JointStore(store.max_entities)
        store._release_hooks.append(self._joints.release)
        self._initialized = False

    def _ensure_pipeline(self, device, texture_format):
        """Lazily initialize device and shared buffers on first use."""
        if getattr(self, "_initialized", False):
            return

        self._device = device

        # Create globals uniform buffer:
        #   vp(64) + view(64) + proj(64) + camera_pos(12) + pad(4)
        #   + viewport_size(8) + pad(8) + ibl_intensity(4) + ibl_enabled(4) + pad(8) = 240 bytes
        #   + shadow block: light_view_proj(64) + sun(32) + shadow params(16) = 352 bytes total
        self._globals_buffer = device.create_buffer(
            size=560,  # ... + fog block (start/end/color/enabled) = 432 + benders (8 vec4) = 560
            usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST,
        )

        # Create batch buffers for per-batch storage (transforms, etc.)
        self._batch_buffers = _BatchBuffers(device)

        # Create separate batch buffers for sprite path
        self._sprite_batch_buffers = _BatchBuffers(device)

        # Create separate batch buffers for label path
        self._label_batch_buffers = _BatchBuffers(device)

        # Create separate batch buffers for axis path
        self._axis_batch_buffers = _BatchBuffers(device)

        # Create lights uniform buffer (up to 4 lights, 32 bytes each = 128 bytes max)
        self._lights_buffer = device.create_buffer(
            size=128,
            usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST,
        )

        # IBL GPU resources — shared sampler + placeholder textures
        self._ibl_env_id = None
        self._ibl_sampler = device.create_sampler(
            min_filter="linear",
            mag_filter="linear",
            mipmap_filter="linear",
        )
        self._ibl_bind_group_layout = self._create_ibl_bind_group_layout(device)
        black_cube_view = self._create_black_cubemap(device)
        self._ibl_neutral_lut_view = self._create_brdf_lut_texture(device)
        self._ibl_placeholder_bind_group = self._create_ibl_bind_group(
            device, black_cube_view, black_cube_view, self._ibl_neutral_lut_view
        )
        self._ibl_active_bind_group = None

        # Shadow-mapping resources.
        self._shadow_map = None
        self._shadow_map_view = None
        self._shadow_map_size = 0
        self._shadow_batch_buffers = _BatchBuffers(device)
        self._shadow_sampler = device.create_sampler(compare=wgpu.CompareFunction.less)
        self._shadow_pipeline = None
        self._shadow_pipeline_layout = None
        self._shadow_bind_layout_vs = None
        self._shadow_bind_layout_skinned = None
        self._shadow_pipeline_layout_skinned = None
        self._shadow_shader_sources = {}  # (stride, skinned) -> WGSL of the shadow pipeline
        # 1x1 placeholder depth texture so the StandardMaterial pipeline's
        # group(2) always has something to bind when shadows are off.
        _placeholder = device.create_texture(
            size=(1, 1, 1),
            format=wgpu.TextureFormat.depth24plus,
            usage=wgpu.TextureUsage.RENDER_ATTACHMENT | wgpu.TextureUsage.TEXTURE_BINDING,
        )
        self._shadow_placeholder_view = _placeholder.create_view()
        self._shadow_bind_group_layout = device.create_bind_group_layout(
            entries=[
                {
                    "binding": 0,
                    "visibility": wgpu.ShaderStage.FRAGMENT,
                    "texture": {
                        "sample_type": wgpu.TextureSampleType.depth,
                        "view_dimension": wgpu.TextureViewDimension.d2,
                    },
                },
                {
                    "binding": 1,
                    "visibility": wgpu.ShaderStage.FRAGMENT,
                    "sampler": {"type": wgpu.SamplerBindingType.comparison},
                },
            ]
        )

        self._initialized = True

    def _create_ibl_bind_group_layout(self, device):
        return device.create_bind_group_layout(
            entries=[
                {
                    "binding": 0,
                    "visibility": wgpu.ShaderStage.FRAGMENT,
                    "sampler": {"type": wgpu.SamplerBindingType.filtering},
                },
                {
                    "binding": 1,
                    "visibility": wgpu.ShaderStage.FRAGMENT,
                    "texture": {
                        "sample_type": wgpu.TextureSampleType.float,
                        "view_dimension": wgpu.TextureViewDimension.cube,
                        "multisampled": False,
                    },
                },
                {
                    "binding": 2,
                    "visibility": wgpu.ShaderStage.FRAGMENT,
                    "texture": {
                        "sample_type": wgpu.TextureSampleType.float,
                        "view_dimension": wgpu.TextureViewDimension.cube,
                        "multisampled": False,
                    },
                },
                {
                    "binding": 3,
                    "visibility": wgpu.ShaderStage.FRAGMENT,
                    "texture": {
                        "sample_type": wgpu.TextureSampleType.float,
                        "view_dimension": wgpu.TextureViewDimension.d2,
                        "multisampled": False,
                    },
                },
            ]
        )

    def _create_black_cubemap(self, device):
        """1×1×6 rgba16float black cubemap, returns texture view."""
        tex = device.create_texture(
            size=(1, 1, 6),
            format=wgpu.TextureFormat.rgba16float,
            usage=wgpu.TextureUsage.TEXTURE_BINDING | wgpu.TextureUsage.COPY_DST,
            dimension=wgpu.TextureDimension.d2,
            mip_level_count=1,
            sample_count=1,
        )
        data = np.zeros((6, 1, 1, 4), dtype=np.float16).tobytes()
        device.queue.write_texture(
            {"texture": tex, "mip_level": 0, "origin": (0, 0, 0)},
            data,
            {"bytes_per_row": 8, "rows_per_image": 1},
            (1, 1, 6),
        )
        return tex.create_view(
            format=wgpu.TextureFormat.rgba16float,
            dimension=wgpu.TextureViewDimension.cube,
            base_mip_level=0,
            mip_level_count=1,
            base_array_layer=0,
            array_layer_count=6,
        )

    def _create_brdf_lut_texture(self, device):
        """Upload pre-baked BRDF LUT, returns texture view."""
        from manifoldx.ibl import load_brdf_lut

        lut = load_brdf_lut().astype(np.float16)
        tex = device.create_texture(
            size=(512, 512, 1),
            format=wgpu.TextureFormat.rg16float,
            usage=wgpu.TextureUsage.TEXTURE_BINDING | wgpu.TextureUsage.COPY_DST,
            dimension=wgpu.TextureDimension.d2,
            mip_level_count=1,
            sample_count=1,
        )
        device.queue.write_texture(
            {"texture": tex, "mip_level": 0, "origin": (0, 0, 0)},
            lut.tobytes(),
            {"bytes_per_row": 512 * 4, "rows_per_image": 512},
            (512, 512, 1),
        )
        return tex.create_view(
            format=wgpu.TextureFormat.rg16float,
            dimension=wgpu.TextureViewDimension.d2,
        )

    def _create_ibl_bind_group(self, device, irr_view, pf_view, lut_view):
        return device.create_bind_group(
            layout=self._ibl_bind_group_layout,
            entries=[
                {"binding": 0, "resource": self._ibl_sampler},
                {"binding": 1, "resource": irr_view},
                {"binding": 2, "resource": pf_view},
                {"binding": 3, "resource": lut_view},
            ],
        )

    def _upload_ibl_env(self, device, env):
        """Upload EnvironmentMap precomputed data to GPU textures, update active bind group."""
        env._precompute()
        irr = env._irradiance  # (6, 64, 64, 4) float16

        irr_tex = device.create_texture(
            size=(64, 64, 6),
            format=wgpu.TextureFormat.rgba16float,
            usage=wgpu.TextureUsage.TEXTURE_BINDING | wgpu.TextureUsage.COPY_DST,
            dimension=wgpu.TextureDimension.d2,
            mip_level_count=1,
            sample_count=1,
        )
        device.queue.write_texture(
            {"texture": irr_tex, "mip_level": 0, "origin": (0, 0, 0)},
            irr.tobytes(),
            {"bytes_per_row": 64 * 8, "rows_per_image": 64},
            (64, 64, 6),
        )
        irr_view = irr_tex.create_view(
            format=wgpu.TextureFormat.rgba16float,
            dimension=wgpu.TextureViewDimension.cube,
            base_mip_level=0,
            mip_level_count=1,
            base_array_layer=0,
            array_layer_count=6,
        )

        pf_tex = device.create_texture(
            size=(128, 128, 6),
            format=wgpu.TextureFormat.rgba16float,
            usage=wgpu.TextureUsage.TEXTURE_BINDING | wgpu.TextureUsage.COPY_DST,
            dimension=wgpu.TextureDimension.d2,
            mip_level_count=8,
            sample_count=1,
        )
        for mip, pf in enumerate(env._prefiltered):
            s = max(1, 128 >> mip)
            device.queue.write_texture(
                {"texture": pf_tex, "mip_level": mip, "origin": (0, 0, 0)},
                pf.astype(np.float16).tobytes(),
                {"bytes_per_row": s * 8, "rows_per_image": s},
                (s, s, 6),
            )
        pf_view = pf_tex.create_view(
            format=wgpu.TextureFormat.rgba16float,
            dimension=wgpu.TextureViewDimension.cube,
            base_mip_level=0,
            mip_level_count=8,
            base_array_layer=0,
            array_layer_count=6,
        )

        self._ibl_active_bind_group = self._create_ibl_bind_group(
            device, irr_view, pf_view, self._ibl_neutral_lut_view
        )
        self._ibl_env_id = id(env)

    def _get_or_create_pipeline(
        self,
        device,
        texture_format,
        geometry_id,
        material,
        registry,
        sprite=False,
        label=False,
        line=False,
        geometry_buffers=None,
    ):
        """Get or create a material-type specific pipeline.

        Pipeline cache key:
            mesh:   (vertex layout, material_type, material_subtype)
            sprite: (geometry_id, material_type, material_subtype, sprite)
            label:  (geometry_id, material_type, material_subtype, "label")
            line:   (geometry_id, material_type, material_subtype, "line")
        """
        self._ensure_pipeline(device, texture_format)
        material_type = type(material).__name__
        material_subtype = getattr(material, "pipeline_subtype", None)

        if line:
            key = (geometry_id, material_type, material_subtype, "line")
        elif label:
            key = (geometry_id, material_type, material_subtype, "label")
        elif sprite:
            key = (geometry_id, material_type, material_subtype, True)
        else:
            # A mesh pipeline depends on the geometry only through its vertex
            # layout, so geometries that share one share the pipeline: a stream
            # of new meshes compiles nothing new.
            gb = geometry_buffers or {}
            layout = ("layout", gb.get("stride", 6 * 4), gb.get("has_uvs", False),
                      gb.get("has_colors", False), gb.get("joints_offset"))
            key = (layout, material_type, material_subtype, getattr(material, "pipeline_state", None))

        if key in self._pipelines:
            return self._pipelines[key], self._bind_group_layouts.get(key)

        if line:
            # 3 bindings: globals, transforms, material uniform.
            # Native LineList topology — 1px lines on Vulkan/Metal/D3D12;
            # thickness honoring is deferred (Plan 3 Task 6 docstring).
            bind_group_entries = [
                {
                    "binding": 0,
                    "visibility": wgpu.ShaderStage.VERTEX | wgpu.ShaderStage.FRAGMENT,
                    "buffer": {"type": wgpu.BufferBindingType.uniform},
                },
                {
                    "binding": 1,
                    "visibility": wgpu.ShaderStage.VERTEX,
                    "buffer": {"type": wgpu.BufferBindingType.read_only_storage},
                },
                {
                    "binding": 2,
                    "visibility": wgpu.ShaderStage.VERTEX | wgpu.ShaderStage.FRAGMENT,
                    "buffer": {"type": wgpu.BufferBindingType.uniform},
                },
            ]

            bind_group_layout = device.create_bind_group_layout(entries=bind_group_entries)
            self._bind_group_layouts[key] = bind_group_layout

            pipeline_layout = device.create_pipeline_layout(bind_group_layouts=[bind_group_layout])
            self._pipeline_layouts[key] = pipeline_layout

            shader_module = device.create_shader_module(code=material._compile())

            pipeline = device.create_render_pipeline(
                layout=pipeline_layout,
                vertex={
                    "module": shader_module,
                    "entry_point": "vs_main",
                    "buffers": [
                        {
                            "array_stride": 3 * 4,  # AXIS_LINE_*: position only
                            "step_mode": wgpu.VertexStepMode.vertex,
                            "attributes": [
                                {
                                    "format": wgpu.VertexFormat.float32x3,
                                    "offset": 0,
                                    "shader_location": 0,
                                },
                            ],
                        }
                    ],
                },
                primitive={
                    "topology": wgpu.PrimitiveTopology.line_list,
                    "front_face": wgpu.FrontFace.ccw,
                    "cull_mode": wgpu.CullMode.none,
                },
                # Standard depth: lines occlude correctly with the 3D scene.
                depth_stencil={
                    "format": wgpu.TextureFormat.depth32float,
                    "depth_write_enabled": True,
                    "depth_compare": wgpu.CompareFunction.greater,
                },
                fragment={
                    "module": shader_module,
                    "entry_point": "fs_main",
                    "targets": [{"format": texture_format}],
                },
            )

            self._pipelines[key] = pipeline

            # AxisMaterial uniform: rgba(16) + anchor_mode + 3 pad = 32 bytes.
            material_buffer = device.create_buffer(
                size=32,
                usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST,
            )
            self._material_buffers[key] = material_buffer

            return pipeline, bind_group_layout

        if label:
            # 6 bindings: globals, transforms, material uniform, label_indices,
            # atlas texture array, atlas sampler.
            bind_group_entries = [
                {
                    "binding": 0,
                    "visibility": wgpu.ShaderStage.VERTEX | wgpu.ShaderStage.FRAGMENT,
                    "buffer": {"type": wgpu.BufferBindingType.uniform},
                },
                {
                    "binding": 1,
                    "visibility": wgpu.ShaderStage.VERTEX,
                    "buffer": {"type": wgpu.BufferBindingType.read_only_storage},
                },
                {
                    "binding": 2,
                    "visibility": wgpu.ShaderStage.VERTEX | wgpu.ShaderStage.FRAGMENT,
                    "buffer": {"type": wgpu.BufferBindingType.uniform},
                },
                {
                    "binding": 3,
                    "visibility": wgpu.ShaderStage.VERTEX,
                    "buffer": {"type": wgpu.BufferBindingType.read_only_storage},
                },
                {
                    "binding": 4,
                    "visibility": wgpu.ShaderStage.FRAGMENT,
                    "texture": {
                        "sample_type": wgpu.TextureSampleType.float,
                        "view_dimension": wgpu.TextureViewDimension.d2_array,
                    },
                },
                {
                    "binding": 5,
                    "visibility": wgpu.ShaderStage.FRAGMENT,
                    "sampler": {"type": wgpu.SamplerBindingType.filtering},
                },
            ]

            bind_group_layout = device.create_bind_group_layout(entries=bind_group_entries)
            self._bind_group_layouts[key] = bind_group_layout

            pipeline_layout = device.create_pipeline_layout(bind_group_layouts=[bind_group_layout])
            self._pipeline_layouts[key] = pipeline_layout

            shader_module = device.create_shader_module(code=material._compile())

            pipeline = device.create_render_pipeline(
                layout=pipeline_layout,
                vertex={
                    "module": shader_module,
                    "entry_point": "vs_main",
                    "buffers": [
                        {
                            "array_stride": 3 * 4,  # SPRITE_QUAD: position only
                            "step_mode": wgpu.VertexStepMode.vertex,
                            "attributes": [
                                {
                                    "format": wgpu.VertexFormat.float32x3,
                                    "offset": 0,
                                    "shader_location": 0,
                                },
                            ],
                        }
                    ],
                },
                primitive={
                    "topology": wgpu.PrimitiveTopology.triangle_list,
                    "front_face": wgpu.FrontFace.ccw,
                    # Camera-facing billboard always points at the camera
                    # by construction; back-face culling would just be a
                    # hidden footgun.
                    "cull_mode": wgpu.CullMode.none,
                },
                # Depth-test on (labels behind opaque geometry are occluded)
                # but depth-write off (overlapping labels alpha-blend cleanly).
                depth_stencil={
                    "format": wgpu.TextureFormat.depth32float,
                    "depth_write_enabled": False,
                    "depth_compare": wgpu.CompareFunction.greater_equal,
                },
                fragment={
                    "module": shader_module,
                    "entry_point": "fs_main",
                    "targets": [
                        {
                            "format": texture_format,
                            "blend": {
                                "color": {
                                    "src_factor": wgpu.BlendFactor.src_alpha,
                                    "dst_factor": wgpu.BlendFactor.one_minus_src_alpha,
                                    "operation": wgpu.BlendOperation.add,
                                },
                                "alpha": {
                                    "src_factor": wgpu.BlendFactor.one,
                                    "dst_factor": wgpu.BlendFactor.one_minus_src_alpha,
                                    "operation": wgpu.BlendOperation.add,
                                },
                            },
                        }
                    ],
                },
            )

            self._pipelines[key] = pipeline

            material_buffer = device.create_buffer(
                size=16,
                usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST,
            )
            self._material_buffers[key] = material_buffer

            return pipeline, bind_group_layout

        # StandardMaterial._compile accepts an optional textured= kwarg;
        # other materials' _compile() takes no args. Detect and forward.
        is_textured = (
            material_subtype == "textured" and getattr(material, "albedo_map", None) is not None
        )
        if is_textured:
            shader_source = material._compile(textured=True)
        elif material_subtype == "vcolor":
            shader_source = material._compile(vertex_colors=True)
        else:
            shader_source = material._compile()
        # Skinned geometry gets its own pipeline (the key holds geometry_id) whose
        # vertex shader poses position and normal by the entity's joint block.
        skinned = not sprite and (geometry_buffers or {}).get("has_joints", False)
        if skinned:
            shader_source = skin_shader(shader_source)
        if not sprite:
            self._shader_sources[key] = shader_source
        shader_module = device.create_shader_module(code=shader_source)

        if sprite:
            # Sprite pipeline: 7 bindings (0-6) with texture+sampbler
            bind_group_entries = [
                {
                    "binding": 0,
                    "visibility": wgpu.ShaderStage.VERTEX | wgpu.ShaderStage.FRAGMENT,
                    "buffer": {"type": wgpu.BufferBindingType.uniform},
                },
                {
                    "binding": 1,
                    "visibility": wgpu.ShaderStage.VERTEX,
                    "buffer": {"type": wgpu.BufferBindingType.read_only_storage},
                },
                {
                    "binding": 2,
                    "visibility": wgpu.ShaderStage.VERTEX | wgpu.ShaderStage.FRAGMENT,
                    "buffer": {"type": wgpu.BufferBindingType.uniform},
                },
                {
                    "binding": 3,
                    "visibility": wgpu.ShaderStage.VERTEX,
                    "buffer": {"type": wgpu.BufferBindingType.read_only_storage},
                },
                {
                    "binding": 4,
                    "visibility": wgpu.ShaderStage.VERTEX,
                    "buffer": {"type": wgpu.BufferBindingType.read_only_storage},
                },
                {
                    "binding": 5,
                    "visibility": wgpu.ShaderStage.FRAGMENT,
                    "texture": {
                        "sample_type": wgpu.TextureSampleType.float,
                        "view_dimension": wgpu.TextureViewDimension.d1,
                    },
                },
                {
                    "binding": 6,
                    "visibility": wgpu.ShaderStage.FRAGMENT,
                    "sampler": {"type": wgpu.SamplerBindingType.filtering},
                },
            ]

            bind_group_layout = device.create_bind_group_layout(entries=bind_group_entries)
            self._bind_group_layouts[key] = bind_group_layout

            pipeline_layout = device.create_pipeline_layout(bind_group_layouts=[bind_group_layout])
            self._pipeline_layouts[key] = pipeline_layout

            # Sprite vertex layout: position only (3 floats, stride=12)
            pipeline = device.create_render_pipeline(
                layout=pipeline_layout,
                vertex={
                    "module": shader_module,
                    "entry_point": "vs_main",
                    "buffers": [
                        {
                            "array_stride": 3 * 4,  # position only
                            "step_mode": wgpu.VertexStepMode.vertex,
                            "attributes": [
                                {
                                    "format": wgpu.VertexFormat.float32x3,
                                    "offset": 0,
                                    "shader_location": 0,
                                },
                            ],
                        }
                    ],
                },
                primitive={
                    "topology": wgpu.PrimitiveTopology.triangle_list,
                    "front_face": wgpu.FrontFace.ccw,
                    "cull_mode": wgpu.CullMode.back,
                },
                depth_stencil={
                    "format": wgpu.TextureFormat.depth32float,
                    "depth_write_enabled": True,
                    "depth_compare": wgpu.CompareFunction.greater,
                },
                fragment={
                    "module": shader_module,
                    "entry_point": "fs_main",
                    "targets": [{"format": texture_format}],
                },
            )

            self._pipelines[key] = pipeline

            # Create material uniform buffer for this sprite key
            buffer_size = 16  # 4 floats: vmin, vmax, lit_flag, _pad
            material_buffer = device.create_buffer(
                size=buffer_size,
                usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST,
            )
            self._material_buffers[key] = material_buffer

        else:
            # Existing mesh pipeline logic
            # Determine bind group layout based on shader bindings
            needs_lights = "@binding(3)" in shader_source
            if not needs_lights:
                bind_group_entries = [
                    {
                        "binding": 0,
                        "visibility": wgpu.ShaderStage.VERTEX | wgpu.ShaderStage.FRAGMENT,
                        "buffer": {"type": wgpu.BufferBindingType.uniform},
                    },
                    {
                        "binding": 1,
                        "visibility": wgpu.ShaderStage.VERTEX,
                        "buffer": {"type": wgpu.BufferBindingType.read_only_storage},
                    },
                    {
                        "binding": 2,
                        "visibility": wgpu.ShaderStage.FRAGMENT,
                        "buffer": {"type": wgpu.BufferBindingType.uniform},
                    },
                ]
            else:
                # StandardMaterial: 4 bindings (globals, transforms, material, lights);
                # +2 more (sampler, texture) when subtype == "textured".
                bind_group_entries = [
                    {
                        "binding": 0,
                        "visibility": wgpu.ShaderStage.VERTEX | wgpu.ShaderStage.FRAGMENT,
                        "buffer": {"type": wgpu.BufferBindingType.uniform},
                    },
                    {
                        "binding": 1,
                        "visibility": wgpu.ShaderStage.VERTEX,
                        "buffer": {"type": wgpu.BufferBindingType.read_only_storage},
                    },
                    {
                        "binding": 2,
                        "visibility": wgpu.ShaderStage.VERTEX | wgpu.ShaderStage.FRAGMENT,
                        "buffer": {"type": wgpu.BufferBindingType.uniform},
                    },
                    {
                        "binding": 3,
                        "visibility": wgpu.ShaderStage.FRAGMENT,
                        "buffer": {"type": wgpu.BufferBindingType.uniform},
                    },
                ]
                if material_subtype == "textured":
                    if not (geometry_buffers or {}).get("has_uvs", False):
                        raise MaterialGeometryMismatchError(
                            f"StandardMaterial(albedo_map=...) requires geometry "
                            f"with UVs; geometry id={geometry_id} has none"
                        )
                    bind_group_entries.extend(
                        [
                            {
                                "binding": 4,
                                "visibility": wgpu.ShaderStage.FRAGMENT,
                                "sampler": {"type": wgpu.SamplerBindingType.filtering},
                            },
                            {
                                "binding": 5,
                                "visibility": wgpu.ShaderStage.FRAGMENT,
                                "texture": {
                                    "sample_type": wgpu.TextureSampleType.float,
                                    "view_dimension": wgpu.TextureViewDimension.d2,
                                },
                            },
                        ]
                    )
            if skinned:
                bind_group_entries.extend(
                    {
                        "binding": binding,
                        "visibility": wgpu.ShaderStage.VERTEX,
                        "buffer": {"type": wgpu.BufferBindingType.read_only_storage},
                    }
                    for binding in (JOINTS_BINDING, BLOCKS_BINDING)
                )

            bind_group_layout = device.create_bind_group_layout(entries=bind_group_entries)
            self._bind_group_layouts[key] = bind_group_layout

            if needs_lights:
                # StandardMaterial: @group(1) = IBL textures, @group(2) = shadow map.
                all_layouts = [
                    bind_group_layout,
                    self._ibl_bind_group_layout,
                    self._shadow_bind_group_layout,
                ]
            else:
                all_layouts = [bind_group_layout]
            pipeline_layout = device.create_pipeline_layout(bind_group_layouts=all_layouts)
            self._pipeline_layouts[key] = pipeline_layout

            # Use the geometry's actual buffer stride so the pipeline
            # advances correctly over UV bytes the scalar shader doesn't read.
            geom_stride = (geometry_buffers or {}).get("stride", 6 * 4)
            _glow = material_subtype == "glow"        # alpha-blended rim-glow
            _scatter = material_subtype == "scatter"  # additive single-scattering atmosphere
            _cloud = material_subtype == "cloud"      # alpha-blended procedural cloud shell
            # A material's pipeline_state: (translucent, mirrored).
            _translucent, _mirrored = getattr(material, "pipeline_state", None) or (False, False)
            _blended = _glow or _scatter or _cloud or _translucent
            vertex_attributes = [
                {
                    "format": wgpu.VertexFormat.float32x3,
                    "offset": 0,
                    "shader_location": 0,
                },
                {
                    "format": wgpu.VertexFormat.float32x3,
                    "offset": 3 * 4,
                    "shader_location": 1,
                },
            ]
            if material_subtype == "textured":
                vertex_attributes.append(
                    {
                        "format": wgpu.VertexFormat.float32x2,
                        "offset": 6 * 4,
                        "shader_location": 2,
                    }
                )
            elif material_subtype == "vcolor":
                vertex_attributes.append(
                    {
                        "format": wgpu.VertexFormat.float32x4,  # colour + surface id
                        "offset": 6 * 4,
                        "shader_location": 2,
                    }
                )
            if skinned:
                vertex_attributes.append(
                    {
                        "format": wgpu.VertexFormat.uint32,
                        "offset": geometry_buffers["joints_offset"],
                        "shader_location": JOINT_LOCATION,
                    }
                )
            pipeline = device.create_render_pipeline(
                layout=pipeline_layout,
                vertex={
                    "module": shader_module,
                    "entry_point": "vs_main",
                    "buffers": [
                        {
                            "array_stride": geom_stride,
                            "step_mode": wgpu.VertexStepMode.vertex,
                            "attributes": vertex_attributes,
                        }
                    ],
                },
                primitive={
                    "topology": wgpu.PrimitiveTopology.triangle_list,
                    "front_face": wgpu.FrontFace.ccw,
                    # A reflection turns its triangles' winding around.
                    "cull_mode": (wgpu.CullMode.none if _blended
                                  else wgpu.CullMode.front if _mirrored else wgpu.CullMode.back),
                },
                depth_stencil={
                    "format": wgpu.TextureFormat.depth32float,
                    "depth_write_enabled": not _blended,
                    "depth_compare": wgpu.CompareFunction.greater,
                },
                fragment={
                    "module": shader_module,
                    "entry_point": "fs_main",
                    "targets": [
                        _scatter_target(texture_format) if _scatter
                        else _glow_target(texture_format) if (_glow or _cloud or _translucent)
                        else {"format": texture_format}
                    ],
                },
            )

            self._pipelines[key] = pipeline

            # Create material uniform buffer for this key
            if not needs_lights:
                buffer_size = 16  # vec4 color
            else:
                buffer_size = (
                    32  # 8 floats: albedo(3) + roughness(1) + metallic(1) + ao(1) + pad(2)
                )

            material_buffer = device.create_buffer(
                size=buffer_size,
                usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST,
            )
            self._material_buffers[key] = material_buffer

        return pipeline, bind_group_layout

    def _scene_bounds(self, engine):
        """World-space bounding sphere (center, radius) of all mesh entities.

        Transforms each geometry's local AABB corners by every instance's model
        matrix and accumulates a global AABB, then returns its bounding sphere.
        Local AABBs are cached per geometry id; skinned geometry keeps one per
        joint, moved by the entity's joint matrix. Returns None for an empty scene.
        """
        if self._in_frame and self._frame_bounds is not None:
            return self._frame_bounds[0]
        bounds = self._compute_scene_bounds(engine)
        if self._in_frame:
            self._frame_bounds = (bounds,)
        return bounds

    def _compute_scene_bounds(self, engine):
        instances = self._collect_mesh_instances(engine, with_entities=True)
        if not instances:
            return None
        cache = getattr(self, "_geom_local_bounds", None)
        if cache is None:
            cache = {}
            self._geom_local_bounds = cache
        gmin = np.full(3, np.inf, dtype=np.float32)
        gmax = np.full(3, -np.inf, dtype=np.float32)
        for geom_id, (mats, entities) in instances.items():
            boxes = cache.get(geom_id)
            if boxes is None:
                geom = engine._geometry_registry.get(geom_id)
                if geom is None or "positions" not in geom:
                    continue
                pos = np.asarray(geom["positions"], dtype=np.float32).reshape(-1, 3)
                if "joints" in geom:
                    joints = np.asarray(geom["joints"])
                    js = np.unique(joints)
                    boxes = (js, np.stack([_box_corners(pos[joints == j]) for j in js]))
                else:
                    boxes = (None, _box_corners(pos)[None])
                cache[geom_id] = boxes
            js, corners = boxes  # corners: (K, 8, 4), one box per joint (K = 1 unskinned)
            if js is None:
                # The box of a box under an affine map is centre' = M c + t, half' = |M| h:
                # per axis from the matrix columns, no corner products.
                lo, hi = corners[0, 0, :3], corners[0, 7, :3]
                c, h = (lo + hi) * 0.5, (hi - lo) * 0.5
                for i in range(3):
                    row = mats[:, 4 * i : 4 * i + 4]
                    centre = row[:, 0] * c[0] + row[:, 1] * c[1] + row[:, 2] * c[2] + row[:, 3]
                    half = np.abs(row[:, 0]) * h[0] + np.abs(row[:, 1]) * h[1] + np.abs(row[:, 2]) * h[2]
                    gmin[i] = min(gmin[i], (centre - half).min())
                    gmax[i] = max(gmax[i], (centre + half).max())
                continue
            full = mats.reshape(-1, 1, 4, 4)  # (N, 1, 4, 4) row-major model matrices
            if js is not None:
                # The mirror holds joint matrices transposed (column-major).
                joint_mats = self._joints.mirror[self._joints.block_of[entities]][:, js]
                full = full @ joint_mats.transpose(0, 1, 3, 2)  # (N, K, 4, 4)
            world = corners @ full.transpose(0, 1, 3, 2)  # (N, K, 8, 4): point @ M.T per box
            w = (world[..., :3] / world[..., 3:4]).reshape(-1, 3)
            gmin = np.minimum(gmin, w.min(axis=0))
            gmax = np.maximum(gmax, w.max(axis=0))
        if not np.all(np.isfinite(gmin)):
            return None
        center = (gmin + gmax) * 0.5
        radius = float(np.linalg.norm(gmax - gmin) * 0.5)
        return center.astype(np.float32), max(radius, 1e-3)

    def _caster_light_view_proj(self, engine, caster):
        """Light-space view-projection for the active shadow caster.

        caster: 0 = none (identity), 1 = sun (ortho auto-fit), 2 = spot (perspective).
        """
        if caster == 2:
            from manifoldx.shadow import compute_spot_light_view_proj

            spot = engine._spot
            cfg = engine._shadow_config
            rng = float(spot.distance) if getattr(spot, "distance", 0) else 30.0
            return compute_spot_light_view_proj(
                position=np.asarray(spot.position, dtype=np.float32),
                direction=np.asarray(spot.direction, dtype=np.float32),
                outer_angle=float(spot.outer_angle),
                near=cfg["near"],
                far=rng,
            )
        if caster == 1:
            return self._sun_light_view_proj(engine)
        return np.eye(4, dtype=np.float32)

    def _sun_light_view_proj(self, engine):
        """Light-space view-projection for the sun (identity until shadows are on)."""
        cfg = getattr(engine, "_shadow_config", None)
        sun = getattr(engine, "_sun", None)
        if cfg is None or sun is None:
            return np.eye(4, dtype=np.float32)
        from manifoldx.shadow import compute_light_view_proj

        direction = np.asarray(sun.direction, dtype=np.float32)

        if cfg.get("auto_fit", True):
            bounds = self._scene_bounds(engine)
            if bounds is not None:
                center, radius = bounds
                pad = max(radius * 0.05, 0.1)
                r = radius + pad
                return compute_light_view_proj(
                    direction=direction,
                    target=center,
                    extent=r,
                    near=0.01,
                    far=2.0 * r + pad,
                    back_distance=r + pad,
                )

        return compute_light_view_proj(
            direction=direction,
            target=np.asarray(cfg["target"], dtype=np.float32),
            extent=cfg["extent"],
            near=cfg["near"],
            far=cfg["far"],
        )

    def _bounds(self, engine, geom_id):
        """A sphere around a geometry's vertices, padded for a skinned one whose joints move them."""
        b = self._geom_bounds.get(geom_id)
        if b is None:
            geo = engine._geometry_registry.get(geom_id) or {}
            pos = geo.get("positions", geo.get("vertices")) if isinstance(geo, dict) else None
            if isinstance(geo, dict) and "bounds" in geo:
                b = geo["bounds"]  # computed where the geometry was built, off the frame
            elif pos is None or not len(pos):
                b = (np.zeros(3), np.inf)
            else:
                b = vertex_bounds(pos, skinned="joints" in geo)
            self._geom_bounds[geom_id] = b
        return b

    def _in_frustum(self, engine, geom_ids, models, mat_ids, clip):
        """Bool per instance: its bounding sphere touches the frustum of `clip`, the matrix taking
        world points to clip space (column vectors). A mirrored instance is tested where its
        reflection shows. `models` are (N, 16) row-major model matrices."""
        n = len(geom_ids)
        if n == 0:
            return np.zeros(0, dtype=bool)
        c = np.empty((n, 3))
        r = np.empty(n)
        geom_ids = np.asarray(geom_ids).astype(np.int64)
        for g in np.unique(geom_ids):  # one lookup per geometry, not per instance
            sel = geom_ids == g
            c[sel], r[sel] = self._bounds(engine, int(g))
        m = np.asarray(models, dtype=np.float64).reshape(n, 4, 4)
        centre = np.einsum("nij,nj->ni", m[:, :3, :3], c) + m[:, :3, 3]
        a = m[:, :3, :3]  # largest column norm: the scale; summing rows beats norm on (N,3,3)
        radius = r * np.sqrt((a[:, 0] ** 2 + a[:, 1] ** 2 + a[:, 2] ** 2).max(axis=1))
        mat_ids = np.asarray(mat_ids).astype(np.int64)
        for mid in np.unique(mat_ids):
            mirror = getattr(engine._material_registry.get(int(mid)), "mirror", None)
            if mirror is not None:
                sel = mat_ids == mid
                centre[sel, 1] = 2.0 * mirror - centre[sel, 1]
        k = np.asarray(clip, dtype=np.float64)
        planes = np.stack([k[3] + k[0], k[3] - k[0], k[3] + k[1], k[3] - k[1], k[2], k[3] - k[2]])
        planes /= np.linalg.norm(planes[:, :3], axis=1, keepdims=True)
        dist = centre @ planes[:, :3].T + planes[:, 3]
        return (dist >= -radius[:, None]).all(axis=1) | ~np.isfinite(radius)

    def _collect_mesh_instances(self, engine, with_entities=False, cull=False):
        """geom_id -> (N,16) row-major model matrices, mesh entities only.

        With with_entities, geom_id -> (matrices, entity indices), the entity
        behind each matrix (the shadow pass reads skinned entities' joint blocks
        through it).

        The shadow caster only needs geometry + transforms (material-agnostic).
        Sprites/labels/axis/volume have geom_id 0 or are non-mesh and are
        excluded here; this is sufficient for v1 (shadows cast by real meshes).
        """
        store = self._store
        alive = np.where(store._alive)[0]
        if len(alive) == 0:
            return {}
        models = self._model_matrices(alive)  # (N,16) row-major
        if "Mesh" not in store._components:
            return {}
        mesh_data = store.get_component_data("Mesh", alive)
        if mesh_data is None:
            return {}
        # Vectorized: a per-entity Python loop and a frustum test per geometry cost a game ~8 ms a frame.
        geoms = mesh_data[:, 0].astype(np.int64)
        keep = geoms != 0
        mat_data = store.get_component_data("Material", alive) if "Material" in store._components else None
        mats = mat_data[:, 0].astype(np.int64) if mat_data is not None else np.zeros(len(alive), np.int64)
        if mat_data is not None:
            for mid in np.unique(mats[keep]):  # one registry lookup per material, not per entity
                if not getattr(engine._material_registry.get(int(mid)), "casts_shadow", True):
                    keep &= mats != mid
        cfg = getattr(engine, "_shadow_config", None)
        if cull and keep.any() and cfg is not None and not cfg.get("auto_fit", True):
            # Only what the shadow caster's box holds casts; its box is fixed, not fitted to these.
            spot = getattr(engine, "_spot", None)
            clip = self._caster_light_view_proj(engine, 2 if spot is not None else 1)
            idx = np.nonzero(keep)[0]
            keep[idx] = self._in_frustum(engine, geoms[idx], models[idx], mats[idx], clip)
        idx = np.nonzero(keep)[0]
        order = idx[np.argsort(geoms[idx], kind="stable")]  # by geometry, entity order kept within each
        bounds = np.flatnonzero(np.diff(geoms[order])) + 1
        out = {int(geoms[v[0]]): v for v in np.split(order, bounds) if len(v)}
        if with_entities:
            return {g: (models[v].astype(np.float32), alive[v]) for g, v in out.items()}
        return {g: models[v].astype(np.float32) for g, v in out.items()}

    def render_shadow_map(self, engine, command_encoder):
        """Depth-only shadow pass (no-op unless a sun + enable_shadows() are set)."""
        from manifoldx.render.passes import shadow as _shadow_pass

        _shadow_pass.render_shadow_map(self, engine, command_encoder)

    def render(self, engine, render_pass):
        """Issue instanced draw calls into an active render pass."""
        if not self._initialized or self._device is None:
            return

        # GUI pass runs unconditionally (even with no scene entities).
        # All other passes require alive entities and known components.
        self._render_scene_passes(engine, render_pass)

        # ---------------------------------------------------------------
        # Draw GUI pass — last in render order. Reads engine.gui directly;
        # no batches needed. No-op when engine.gui is empty.
        # ---------------------------------------------------------------
        from manifoldx.render.passes import gui as _gui_pass

        _gui_pass.render_gui_pass(self, engine, render_pass)

    def _render_scene_passes(self, engine, render_pass):
        """Issue draw calls for scene entities (mesh / sprite / volume / label / axis)."""
        # Get all alive entities on the camera's layers
        alive_indices = np.where(self._store._alive)[0]
        layers = getattr(engine._camera, "layers", None)
        if layers is not None:
            if "Layer" in self._store._components:
                on = self._store._components["Layer"][alive_indices, 0]
            else:
                on = np.zeros(len(alive_indices))
            alive_indices = alive_indices[np.isin(on, list(layers))]
        if len(alive_indices) == 0:
            return

        # Check required components exist
        if "Transform" not in self._store._components:
            return

        has_mesh = "Mesh" in self._store._components
        has_point_cloud = "PointCloud" in self._store._components
        has_text_label = "TextLabel" in self._store._components
        has_axis_frame = "AxisFrame" in self._store._components
        has_volume = "Volume" in self._store._components
        if (
            not has_mesh
            and not has_point_cloud
            and not has_text_label
            and not has_axis_frame
            and not has_volume
        ):
            return

        # Compute view-projection matrix from camera
        camera = engine._camera
        aspect = engine.w / engine.h
        vp = camera.get_view_projection_matrix(aspect)

        # Compute view matrix from camera (column-major for WGSL)
        view_mat = camera.get_view_matrix().T.astype(np.float32)

        # Get transform data
        model_matrices = self._model_matrices(alive_indices)

        # Get mesh and material IDs
        mesh_data = None
        if has_mesh:
            mesh_data = self._store.get_component_data("Mesh", alive_indices)
        material_data = None
        if "Material" in self._store._components:
            material_data = self._store.get_component_data("Material", alive_indices)
        volume_data = None
        if has_volume:
            volume_data = self._store.get_component_data("Volume", alive_indices)

        # Per-entity routing. Priority: axis > label > volume > sprite > mesh. It depends
        # only on the material, so each distinct material is classified once and entities
        # are grouped by sorting: a Python loop over entities cost ~100 ms at 100k.
        # An entity is a sprite when PointCloud is registered AND it has no Mesh
        # geometry (geom_id == 0). Axis frames carry both AxisFrame *and* an
        # AXIS_LINE_* mesh, so they need to win over the mesh path.
        from manifoldx.viz import AxisMaterial, LabelMaterial, VolumeMaterial

        n = len(alive_indices)
        mats = material_data[:, 0].astype(np.int64) if material_data is not None else np.zeros(n, np.int64)
        geoms = mesh_data[:, 0].astype(np.int64) if mesh_data is not None else np.zeros(n, np.int64)
        uniq, inv = np.unique(mats, return_inverse=True)
        routes = [(has_axis_frame, AxisMaterial), (has_text_label, LabelMaterial), (has_volume, VolumeMaterial)]
        kind = np.zeros(len(uniq), np.int8)  # 0 sprite or mesh, 1 axis, 2 label, 3 volume
        for u, mid in enumerate(uniq):
            mat_obj = engine._material_registry.get(int(mid)) if mid > 0 else None
            kind[u] = next((r for r, (on, cls) in enumerate(routes, 1) if on and isinstance(mat_obj, cls)), 0)
        kind = kind[inv]

        axis_batches = {k: v for k, v in _groups(np.flatnonzero(kind == 1), geoms, mats)}
        label_batches = {k[0]: v for k, v in _groups(np.flatnonzero(kind == 2), mats)}
        volume_batches = []  # list of (entity_local_idx, mat_id, vol_id)
        if volume_data is not None:
            vols = volume_data[:, 0].astype(np.int64)
            volume_batches = [(int(i), int(mats[i]), int(vols[i])) for i in np.flatnonzero((kind == 3) & (vols > 0))]
        sprite_batches = {}
        if has_point_cloud:
            sprite_batches = {k[0]: v for k, v in _groups(np.flatnonzero((kind == 0) & (geoms == 0)), mats)}

        # Frustum culling: a mesh whose bounding sphere is out of view is not drawn. Only
        # entities with geometry are tested: sprites and labels would never be culled.
        mesh_batches = {}  # (geom_id, mat_type, mat_subtype, mat_id) -> local indices
        if has_mesh:
            idx = np.flatnonzero((kind == 0) & (geoms > 0))
            idx = idx[self._in_frustum(engine, geoms[idx], model_matrices[idx], mats[idx], vp.T)]
            for (geom_id, mat_id), v in _groups(idx, geoms, mats):
                mat_obj = engine._material_registry.get(mat_id) if mat_id > 0 else None
                mat_type = type(mat_obj).__name__ if mat_obj else "BasicMaterial"
                mat_subtype = getattr(mat_obj, "pipeline_subtype", None) if mat_obj else None
                # mat_id is part of the key so two material *instances* of the
                # same class never share a uniform buffer.
                mesh_batches[(geom_id, mat_type, mat_subtype, mat_id)] = v

        # Get camera position for globals
        camera_pos = np.array([0.0, 0.0, 5.0], dtype=np.float32)
        if hasattr(camera, "position"):
            camera_pos = np.array(camera.position, dtype=np.float32)

        # Upload globals: vp(64) + view(64) + proj(64) + camera_pos(12) +
        # pad(4) + viewport_size(8) + pad(8) + ibl_intensity(4) + ibl_enabled(4) + pad(8) = 240 bytes,
        # then the shadow block: light_view_proj(64) + sun_direction(12)+pad(4)
        # + sun_color(12)+sun_intensity(4) + shadow_enabled(4)+bias(4)+map_size(4)+pad(4) = 112 bytes,
        # total 352 bytes.
        globals_data = np.zeros(560, dtype=np.uint8)
        globals_data[0:64] = np.frombuffer(vp.astype(np.float32).tobytes(), dtype=np.uint8)
        globals_data[64:128] = np.frombuffer(view_mat.tobytes(), dtype=np.uint8)
        proj_mat = camera.get_projection_matrix(aspect, near=camera.near, far=camera.far).T.astype(
            np.float32
        )
        globals_data[128:192] = np.frombuffer(proj_mat.tobytes(), dtype=np.uint8)
        globals_data[192:204] = np.frombuffer(
            camera_pos.astype(np.float32).tobytes(), dtype=np.uint8
        )
        # bytes 204-207: the clock, wrapped to the 64 s every swaying wave repeats over
        globals_data[204:208] = np.frombuffer(
            np.float32(getattr(engine, "elapsed", 0.0) % 64.0).tobytes(), dtype=np.uint8
        )
        viewport_size = np.array([float(engine.w), float(engine.h)], dtype=np.float32)
        globals_data[208:216] = np.frombuffer(viewport_size.tobytes(), dtype=np.uint8)
        # bytes 216-223: the floating origin's x, z mod 256, so world-placed waves survive a rebase
        ox, oz = getattr(engine, "_origin", (0.0, 0.0))
        globals_data[216:224] = np.frombuffer(
            np.array([ox % 256.0, oz % 256.0], dtype=np.float32).tobytes(), dtype=np.uint8
        )
        ibl_env = engine._ready_environment()
        if ibl_env is not None:
            globals_data[224:228] = np.frombuffer(
                np.float32(ibl_env.intensity).tobytes(), dtype=np.uint8
            )
            globals_data[228:232] = np.frombuffer(np.uint32(1).tobytes(), dtype=np.uint8)
        # bytes 232-239: padding

        # --- Shadow / lighting block (offset 240+) ---
        cfg = getattr(engine, "_shadow_config", None)
        sun = getattr(engine, "_sun", None)
        spot = getattr(engine, "_spot", None)
        # Caster priority: a spot (flashlight) casts when present, else the sun.
        caster = 0
        if cfg is not None:
            if spot is not None:
                caster = 2
            elif sun is not None:
                caster = 1

        # light_view_proj @240 (stored transposed / column-major, like proj/view).
        lvp = self._caster_light_view_proj(engine, caster)
        globals_data[240:304] = np.frombuffer(lvp.T.astype(np.float32).tobytes(), dtype=np.uint8)

        # sun block @304 — 32 bytes from DirectionalLight.get_data().
        if sun is not None:
            globals_data[304:336] = np.frombuffer(
                sun.get_data().astype(np.float32).tobytes(), dtype=np.uint8
            )

        # shadow params @336 — enabled only when there is a caster.
        if caster != 0:
            globals_data[336:340] = np.frombuffer(np.uint32(1).tobytes(), dtype=np.uint8)
            globals_data[340:344] = np.frombuffer(np.float32(cfg["bias"]).tobytes(), dtype=np.uint8)
            globals_data[344:348] = np.frombuffer(
                np.float32(cfg["resolution"]).tobytes(), dtype=np.uint8
            )
            globals_data[348:352] = np.frombuffer(
                np.uint32(cfg.get("pcf_radius", 1)).tobytes(), dtype=np.uint8
            )

        # spot block @352 — flashlight params (position/range, dir/cos_inner,
        # color/intensity, cos_outer). Zero (skipped by the shader) when unset.
        if spot is not None:
            direction = np.asarray(spot.direction, dtype=np.float32)
            direction = direction / (np.linalg.norm(direction) + 1e-9)
            rng = float(spot.distance) if getattr(spot, "distance", 0) else 30.0
            hexs = spot.color.lstrip("#")
            color = (
                np.array(
                    [int(hexs[0:2], 16), int(hexs[2:4], 16), int(hexs[4:6], 16)], dtype=np.float32
                )
                / 255.0
            )
            globals_data[352:364] = np.frombuffer(
                np.asarray(spot.position, dtype=np.float32).tobytes(), dtype=np.uint8
            )
            globals_data[364:368] = np.frombuffer(np.float32(rng).tobytes(), dtype=np.uint8)
            globals_data[368:380] = np.frombuffer(direction.tobytes(), dtype=np.uint8)
            globals_data[380:384] = np.frombuffer(
                np.float32(np.cos(spot.inner_angle)).tobytes(), dtype=np.uint8
            )
            globals_data[384:396] = np.frombuffer(color.tobytes(), dtype=np.uint8)
            globals_data[396:400] = np.frombuffer(
                np.float32(spot.intensity).tobytes(), dtype=np.uint8
            )
            globals_data[400:404] = np.frombuffer(
                np.float32(np.cos(spot.outer_angle)).tobytes(), dtype=np.uint8
            )

        # shadow_caster @404
        globals_data[404:408] = np.frombuffer(np.uint32(caster).tobytes(), dtype=np.uint8)

        # fog block @408 — start/end/color/enabled (zero = off, skipped by the shader)
        if getattr(engine, "fog_enabled", False):
            globals_data[408:412] = np.frombuffer(np.float32(engine.fog_start).tobytes(), dtype=np.uint8)
            globals_data[412:416] = np.frombuffer(np.float32(engine.fog_end).tobytes(), dtype=np.uint8)
            globals_data[416:428] = np.frombuffer(
                np.asarray(engine.fog_color, dtype=np.float32).tobytes(), dtype=np.uint8
            )
            globals_data[428:432] = np.frombuffer(np.uint32(1).tobytes(), dtype=np.uint8)

        # benders @432 — 8 vec4 (x, z, radius, strength) in the render frame; radius 0 is unused
        benders = np.zeros((MAX_BENDERS, 4), dtype=np.float32)
        for bi, (bx, bz, radius, strength) in enumerate(list(getattr(engine, "benders", ()))[:MAX_BENDERS]):
            benders[bi] = (bx - ox, bz - oz, radius, strength)
        globals_data[432:560] = np.frombuffer(benders.tobytes(), dtype=np.uint8)

        self._device.queue.write_buffer(self._globals_buffer, 0, globals_data.tobytes())

        # Upload lights once (shared across all PBR draws)
        lights = getattr(engine, "_lights", [])
        lights_data = np.zeros(32, dtype=np.float32)  # 32 floats = 128 bytes
        for li, light in enumerate(lights[:4]):
            light_arr = light.get_data()
            offset = li * 8  # 8 floats per light
            lights_data[offset : offset + len(light_arr)] = light_arr
        self._device.queue.write_buffer(self._lights_buffer, 0, lights_data.tobytes())

        # ---------------------------------------------------------------
        # Skybox (renders at far plane — before mesh geometry)
        # ---------------------------------------------------------------
        env = engine._ready_environment()
        if env is not None and env.show_skybox:
            _skybox_pass.render_skybox(self, engine, render_pass)

        # ---------------------------------------------------------------
        # Draw mesh batches (if any)
        # ---------------------------------------------------------------
        self.drawn = sum(len(v) for v in mesh_batches.values())
        if mesh_batches:
            # Opaque meshes first, then mirrored ones, which the depth test then rejects cheaply
            # wherever the ground hides them, then translucent ones over everything.
            def _order(key):
                state = getattr(engine._material_registry.get(key[3]), "pipeline_state", None)
                return (bool(state and state[0]), bool(state and state[1]))

            mesh_batches = dict(sorted(mesh_batches.items(), key=lambda kv: _order(kv[0])))
            _mesh_pass.render_mesh_batches(
                self, engine, render_pass, mesh_batches, model_matrices, material_data, alive_indices
            )

        # ---------------------------------------------------------------
        # Draw sprite batches (if any)
        # ---------------------------------------------------------------
        if sprite_batches:
            _sprite_pass.render_sprite_batches(
                self, engine, render_pass, sprite_batches, model_matrices, material_data
            )

        # ---------------------------------------------------------------
        # Draw volume batches (depth-test on, depth-write off, alpha-blend on)
        # ---------------------------------------------------------------
        if volume_batches:
            _volume_pass.render_volume_pass(self, engine, render_pass, volume_batches)

        # ---------------------------------------------------------------
        # Draw label batches (depth-write off, alpha-blend on)
        # ---------------------------------------------------------------
        if label_batches:
            _label_pass.render_label_pass(
                self, engine, render_pass, label_batches, model_matrices, material_data
            )

        # ---------------------------------------------------------------
        # Draw axis batches (LineList topology, opaque)
        # ---------------------------------------------------------------
        if axis_batches:
            _axis_pass.render_axis_pass(
                self, engine, render_pass, axis_batches, model_matrices, material_data
            )

    def _model_matrices(self, indices):
        """(N,16) row-major model matrices: as of the frame's start in one, as the store is now outside one."""
        if not self._in_frame:
            self._transform_cache.refresh(self._store)
        return self._transform_cache.get_transforms(self._store, indices)

    def run(self, engine, dt: float):
        """Execute full render pipeline (CPU-side prep). Starts a frame, which end_frame() closes."""
        self._frame_bounds = None
        self._in_frame = True
        if "Transform" not in self._store._components:
            return
        self._transform_cache.refresh(self._store)

    def end_frame(self):
        """Close the frame run() opened: later reads see the store as it is then."""
        self._in_frame = False


__all__ = [
    "TransformCache",
    "RenderPipeline",
    "SHADER_SOURCE",
    "VOLUME_SHADER_SOURCE",
]
