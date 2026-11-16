"""Sci-viz materials.

ColormapMaterial: maps a per-instance scalar value through a 1D LUT.
Camera-facing point sprites with sphere-imposter normal reconstruction.
"""

import numpy as np
from typing import Dict

from manifoldx.resources import Material
from manifoldx.viz import colormaps


# WGSL shader source for ColormapMaterial.
#
# Bindings (group 0):
#   0: Globals uniform   { vp: mat4x4, view: mat4x4, proj: mat4x4, camera_pos: vec3, _pad: f32 }
#   1: transforms        storage<read> array<mat4x4<f32>>
#   2: material uniform  { vmin: f32, vmax: f32, lit_flag: f32, _pad: f32 }
#   3: scalar_values     storage<read> array<f32>
#   4: radii             storage<read> array<f32>
#   5: lut_texture       texture_1d<f32>
#   6: lut_sampler       sampler
#
# Vertex inputs:
#   @location(0) position: vec3<f32>   — quad-local in [-1, 1]^2 (z = 0)
#
# Vertex outputs / fragment inputs:
#   @location(0) quad_uv: vec2<f32>    — passes the quad-local xy for normal reconstruction
#   @location(1) scalar:  f32          — per-instance scalar value passed to fragment

_COLORMAP_SHADER = """
struct Globals {
    vp: mat4x4<f32>,
    view: mat4x4<f32>,
    proj: mat4x4<f32>,
    camera_pos: vec3<f32>,
    _pad: f32,
};

struct MaterialUniform {
    vmin: f32,
    vmax: f32,
    lit_flag: f32,
    _pad: f32,
};

@group(0) @binding(0) var<uniform> globals: Globals;
@group(0) @binding(1) var<storage, read> transforms: array<mat4x4<f32>>;
@group(0) @binding(2) var<uniform> material: MaterialUniform;
@group(0) @binding(3) var<storage, read> scalar_values: array<f32>;
@group(0) @binding(4) var<storage, read> radii: array<f32>;
@group(0) @binding(5) var lut_texture: texture_1d<f32>;
@group(0) @binding(6) var lut_sampler: sampler;

struct VSIn {
    @location(0) position: vec3<f32>,
};

struct VSOut {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) quad_uv: vec2<f32>,
    @location(1) scalar: f32,
};

@vertex
fn vs_main(in: VSIn, @builtin(instance_index) iidx: u32) -> VSOut {
    let model = transforms[iidx];
    let radius = radii[iidx];
    let scalar = scalar_values[iidx];

    // Center of the sprite in world space
    let world_center = (model * vec4<f32>(0.0, 0.0, 0.0, 1.0)).xyz;

    // Transform center to view space, then offset in view-space XY by quad-local
    // position scaled by radius. This makes the quad always face the camera.
    let view_center = (globals.view * vec4<f32>(world_center, 1.0)).xyz;
    let offset = vec2<f32>(in.position.x, in.position.y) * radius;
    let view_pos = vec4<f32>(view_center.x + offset.x, view_center.y + offset.y, view_center.z, 1.0);

    // Project view-space position directly to clip space.
    let clip = globals.proj * view_pos;

    var out: VSOut;
    out.clip_position = clip;
    out.quad_uv = in.position.xy;  // [-1, 1]^2
    out.scalar = scalar;
    return out;
}

@fragment
fn fs_main(in: VSOut) -> @location(0) vec4<f32> {
    let r2 = dot(in.quad_uv, in.quad_uv);
    if (r2 > 1.0) {
        discard;
    }
    let n_view = vec3<f32>(in.quad_uv.x, in.quad_uv.y, sqrt(1.0 - r2));
    let denom = max(material.vmax - material.vmin, 1e-6);
    let t = clamp((in.scalar - material.vmin) / denom, 0.0, 1.0);
    let base_color = textureSample(lut_texture, lut_sampler, t);
    let light_dir = normalize(vec3<f32>(0.5, 0.5, 1.0));
    let lambert = max(dot(n_view, light_dir), 0.0);
    let lit_factor = mix(1.0, 0.2 + 0.8 * lambert, material.lit_flag);
    return vec4<f32>(base_color.rgb * lit_factor, base_color.a);
}
""".strip()


# WGSL shader source for LabelMaterial.
#
# Bindings (group 0):
#   0: Globals uniform   { vp: mat4x4, view: mat4x4, proj: mat4x4,
#                          camera_pos: vec3, _pad0: f32,
#                          viewport_size: vec2, _pad1: vec2 }
#   1: transforms        storage<read> array<mat4x4<f32>>
#   2: material uniform  { pixel_width: f32, pixel_height: f32, anchor_mode: f32, _pad: f32 }
#   3: label_indices     storage<read> array<f32>     # cast to u32 in shader
#   4: atlas_texture     texture_2d_array<f32>
#   5: atlas_sampler     sampler
#
# Vertex inputs:
#   @location(0) position: vec3<f32>   — quad-local in [-1, 1]^2 (z = 0)
#
# Vertex outputs:
#   @location(0) uv:    vec2<f32>      — texture UV in [0, 1]^2
#   @location(1) slice: f32            — label slice index (f32-encoded u32)

_LABEL_SHADER = """
struct Globals {
    vp: mat4x4<f32>,
    view: mat4x4<f32>,
    proj: mat4x4<f32>,
    camera_pos: vec3<f32>,
    _pad0: f32,
    viewport_size: vec2<f32>,
    _pad1: vec2<f32>,
};

struct MaterialUniform {
    pixel_width: f32,
    pixel_height: f32,
    anchor_mode: f32,
    _pad: f32,
};

@group(0) @binding(0) var<uniform> globals: Globals;
@group(0) @binding(1) var<storage, read> transforms: array<mat4x4<f32>>;
@group(0) @binding(2) var<uniform> material: MaterialUniform;
@group(0) @binding(3) var<storage, read> label_indices: array<f32>;
@group(0) @binding(4) var atlas_texture: texture_2d_array<f32>;
@group(0) @binding(5) var atlas_sampler: sampler;

struct VSIn {
    @location(0) position: vec3<f32>,
};

struct VSOut {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) slice: f32,
};

@vertex
fn vs_main(in: VSIn, @builtin(instance_index) iidx: u32) -> VSOut {
    let model = transforms[iidx];
    let world_center = (model * vec4<f32>(0.0, 0.0, 0.0, 1.0)).xyz;

    var clip: vec4<f32>;

    if (material.anchor_mode < 0.5) {
        // World-anchored: project the entity's world position through the
        // camera, then add a billboard offset in view space scaled so the
        // label takes up `pixel_width` × `pixel_height` pixels on screen.
        let view_center = (globals.view * vec4<f32>(world_center, 1.0)).xyz;
        let half_w_pixels = material.pixel_width * 0.5;
        let half_h_pixels = material.pixel_height * 0.5;
        let view_z = max(-view_center.z, 1e-3);
        let view_dx = in.position.x * half_w_pixels * 2.0 * view_z / globals.proj[0][0] / globals.viewport_size.x;
        let view_dy = in.position.y * half_h_pixels * 2.0 * view_z / globals.proj[1][1] / globals.viewport_size.y;
        let view_pos = vec4<f32>(
            view_center.x + view_dx,
            view_center.y + view_dy,
            view_center.z,
            1.0,
        );
        clip = globals.proj * view_pos;
    } else {
        // Screen-anchored: bypass view/proj entirely. Treat
        // Transform.pos.xy as an NDC anchor in [-1, 1] and add the
        // quad-local offset converted from pixels to NDC. NDC width
        // is 2 units across the full viewport, so 1 pixel = 2 / viewport
        // units; a quad-local x in [-1, 1] spans the full pixel_width
        // → NDC offset = quad_local.x * pixel_width / viewport_size.x.
        let ndc_dx = in.position.x * material.pixel_width / globals.viewport_size.x;
        let ndc_dy = in.position.y * material.pixel_height / globals.viewport_size.y;
        clip = vec4<f32>(world_center.x + ndc_dx, world_center.y + ndc_dy, 1.0, 1.0);  // near plane (reversed depth)
    }

    var out: VSOut;
    out.clip_position = clip;
    // Map quad position [-1, 1]^2 to UV [0, 1]^2 with V flipped so PIL's
    // top-left origin lands at the top of the rendered quad.
    out.uv = vec2<f32>(in.position.x * 0.5 + 0.5, 0.5 - in.position.y * 0.5);
    out.slice = label_indices[iidx];
    return out;
}

@fragment
fn fs_main(in: VSOut) -> @location(0) vec4<f32> {
    let layer = i32(in.slice + 0.5);
    let texel = textureSample(atlas_texture, atlas_sampler, in.uv, layer);
    return texel;
}
""".strip()


class ColormapMaterial(Material):
    """Point-sprite material that colormaps a per-instance scalar.

    Per-batch uniform (4 floats):
        vmin, vmax, lit_flag, _pad

    Per-instance storage buffers (read in shader by instance_index):
        transforms (mat4x4)         — existing
        scalar_values (float)       — new, sourced from ScalarValue
        radii (float)               — new, sourced from Radius

    Per-batch texture: 256x1 RGBA8 1D LUT, sampled in fragment shader.
    """

    binding_slot = 2

    def __init__(self, cmap: str, vmin: float, vmax: float, lit: bool = False):
        # Validate cmap exists
        colormaps.get_colormap(cmap)
        self.cmap = cmap
        self.vmin = float(vmin)
        self.vmax = float(vmax)
        self.lit = bool(lit)

    @classmethod
    def _compile(cls) -> str:
        return _COLORMAP_SHADER

    @classmethod
    def uniform_type(cls) -> Dict[str, str]:
        return {"vmin": "f32", "vmax": "f32", "lit_flag": "f32", "_pad": "f32"}

    @property
    def pipeline_subtype(self) -> str:
        """Used by RenderPipeline cache to share pipelines across cmaps."""
        return self.cmap

    def get_data(self, n: int, registry=None) -> np.ndarray:
        """Per-batch material uniform data, broadcast to n rows.

        The renderer reads row 0 as the uniform; n rows are produced for
        compatibility with the existing material registry's per-instance
        data convention.
        """
        row = np.array(
            [self.vmin, self.vmax, 1.0 if self.lit else 0.0, 0.0],
            dtype=np.float32,
        )
        return np.broadcast_to(row, (n, 4)).copy()

    def get_lut(self) -> np.ndarray:
        """Return the (256, 4) uint8 LUT for this material's colormap."""
        return colormaps.get_colormap(self.cmap)


_VALID_ANCHOR_MODES = ("world", "screen")


class LabelMaterial(Material):
    """Camera-facing billboard material textured with a label-atlas slice.

    Per-batch uniform (4 floats):
        pixel_width, pixel_height, anchor_mode, _pad
        (anchor_mode: 0.0 = world, 1.0 = screen — Plan 2 ships world only)

    Per-instance storage buffer:
        transforms (mat4x4)  — existing
        label_indices (f32)  — atlas slice index, cast to u32 in shader

    Texture binding: 2D texture array (TILE_WIDTH x TILE_HEIGHT x MAX_LABELS).

    Pipeline cache key: includes pipeline_subtype = anchor_mode so the
    world-anchored and screen-anchored pipelines stay separate even though
    they share a shader. (Screen-anchored is reserved for Plan 3.)
    """

    binding_slot = 3

    def __init__(
        self,
        *,
        pixel_width: float = 256.0,
        pixel_height: float = 64.0,
        anchor_mode: str = "world",
    ):
        if anchor_mode not in _VALID_ANCHOR_MODES:
            raise ValueError(
                f"LabelMaterial.anchor_mode must be one of {_VALID_ANCHOR_MODES}, got {anchor_mode!r}"
            )
        self.pixel_width = float(pixel_width)
        self.pixel_height = float(pixel_height)
        self.anchor_mode = anchor_mode

    @classmethod
    def _compile(cls) -> str:
        return _LABEL_SHADER

    @classmethod
    def uniform_type(cls) -> Dict[str, str]:
        return {
            "pixel_width": "f32",
            "pixel_height": "f32",
            "anchor_mode": "f32",
            "_pad": "f32",
        }

    @property
    def pipeline_subtype(self) -> str:
        return self.anchor_mode

    def get_data(self, n: int, registry=None) -> np.ndarray:
        anchor_f = 0.0 if self.anchor_mode == "world" else 1.0
        row = np.array(
            [self.pixel_width, self.pixel_height, anchor_f, 0.0],
            dtype=np.float32,
        )
        return np.broadcast_to(row, (n, 4)).copy()


# WGSL shader source for AxisMaterial.
#
# Bindings (group 0):
#   0: Globals uniform   { vp, view, proj, camera_pos, _pad0,
#                          viewport_size, _pad1 }
#   1: transforms        storage<read> array<mat4x4<f32>>
#   2: material uniform  { r: f32, g: f32, b: f32, a: f32 }
#
# Vertex inputs:
#   @location(0) position: vec3<f32>   — line endpoint in unit space
#                                        (e.g. AXIS_LINE_X has vertices at
#                                        (-1, 0, 0) and (+1, 0, 0); the entity's
#                                        Transform.scale = (extent, 1, 1) puts
#                                        them at the right world distance)

_AXIS_SHADER = """
struct Globals {
    vp: mat4x4<f32>,
    view: mat4x4<f32>,
    proj: mat4x4<f32>,
    camera_pos: vec3<f32>,
    _pad0: f32,
    viewport_size: vec2<f32>,
    _pad1: vec2<f32>,
};

struct MaterialUniform {
    r: f32,
    g: f32,
    b: f32,
    a: f32,
    anchor_mode: f32,
    _pad0: f32,
    _pad1: f32,
    _pad2: f32,
};

@group(0) @binding(0) var<uniform> globals: Globals;
@group(0) @binding(1) var<storage, read> transforms: array<mat4x4<f32>>;
@group(0) @binding(2) var<uniform> material: MaterialUniform;

struct VSIn {
    @location(0) position: vec3<f32>,
};

struct VSOut {
    @builtin(position) clip_position: vec4<f32>,
};

@vertex
fn vs_main(in: VSIn, @builtin(instance_index) iidx: u32) -> VSOut {
    let model = transforms[iidx];
    var clip: vec4<f32>;
    if (material.anchor_mode < 0.5) {
        // World-anchored: project through camera as usual.
        let world_pos = (model * vec4<f32>(in.position, 1.0)).xyz;
        clip = globals.vp * vec4<f32>(world_pos, 1.0);
    } else {
        // Screen-anchored: model matrix gives the NDC position directly.
        // The entity's Transform.pos.xy is the NDC anchor in [-1, 1] and
        // Transform.scale.{x,y,z} sets the half-length of the line in NDC
        // units along whichever axis the geometry runs along.
        let ndc_pos = (model * vec4<f32>(in.position, 1.0)).xyz;
        clip = vec4<f32>(ndc_pos.x, ndc_pos.y, 1.0, 1.0);  // near plane (reversed depth)
    }
    var out: VSOut;
    out.clip_position = clip;
    return out;
}

@fragment
fn fs_main(in: VSOut) -> @location(0) vec4<f32> {
    return vec4<f32>(material.r, material.g, material.b, material.a);
}
""".strip()


_VALID_AXIS_ANCHOR_MODES = ("world", "screen")


class AxisMaterial(Material):
    """Unlit colored line material for axis frames.

    Per-batch uniform (8 floats): r, g, b, a, anchor_mode, _pad0, _pad1, _pad2.
        anchor_mode: 0.0 = world (project through camera), 1.0 = screen
        (model output's xy IS the NDC clip position; bypass view/proj).
    Per-instance: none (each color = one batch).
    Pipeline: native wgpu LineList topology, 1px native lines.

    Three axes (X / Y / Z) are typically spawned as three entities, each
    with its own AxisMaterial instance (and thus its own batch + color).
    Default colors follow the sci-viz v1 spec: X=#e64545, Y=#5fbf5f,
    Z=#5588ff (mutable via the `color` keyword).

    For screen-anchored axes (e.g. a scale bar), pass
    `anchor_mode="screen"`; the entity's Transform.pos.xy is then the
    NDC anchor in [-1, 1] and Transform.scale.{x,y,z} sets the
    half-length of the line in NDC units.
    """

    binding_slot = 3

    def __init__(self, *, color: str = "#ffffff", anchor_mode: str = "world"):
        if anchor_mode not in _VALID_AXIS_ANCHOR_MODES:
            raise ValueError(
                f"AxisMaterial.anchor_mode must be one of "
                f"{_VALID_AXIS_ANCHOR_MODES}, got {anchor_mode!r}"
            )
        self.color = color
        self.anchor_mode = anchor_mode

    @classmethod
    def _compile(cls) -> str:
        return _AXIS_SHADER

    @classmethod
    def uniform_type(cls) -> Dict[str, str]:
        return {
            "r": "f32",
            "g": "f32",
            "b": "f32",
            "a": "f32",
            "anchor_mode": "f32",
            "_pad0": "f32",
            "_pad1": "f32",
            "_pad2": "f32",
        }

    @property
    def pipeline_subtype(self):
        # World and screen modes need separate pipelines because the vertex
        # shader's clip-space derivation differs in observable ways (depth
        # write, occlusion). Color difference still rides through the
        # per-batch uniform; only the anchor mode forks the cache key.
        return self.anchor_mode

    def get_data(self, n: int, registry=None) -> np.ndarray:
        from manifoldx.types import Color
        c = Color(self.color).to_linear()
        anchor_f = 0.0 if self.anchor_mode == "world" else 1.0
        row = np.array(
            [c.r, c.g, c.b, c.a, anchor_f, 0.0, 0.0, 0.0],
            dtype=np.float32,
        )
        return np.broadcast_to(row, (n, 8)).copy()


class VolumeMaterial(Material):
    """Direct volume rendering material — colormap LUT + opacity LUT.

    See `.knowledge/analysis/2026-05-08-volume-rendering-v1-design.md`
    for the full transfer-function semantics.
    """

    pipeline_subtype: str | None = "volume"
    binding_slot = 4

    def __init__(
        self,
        cmap: str = "viridis",
        *,
        vmin: float = 0.0,
        vmax: float = 1.0,
        opacity_stops=None,
        density_scale: float = 1.0,
        step_size: float | None = None,
        max_steps: int = 256,
    ):
        from manifoldx.viz.colormaps import _LUTS

        if cmap not in _LUTS:
            raise ValueError(
                f"unknown cmap: {cmap!r}; available: {sorted(_LUTS)}"
            )
        if not (vmin < vmax):
            raise ValueError(f"vmin must be < vmax; got vmin={vmin}, vmax={vmax}")
        if step_size is not None and step_size <= 0.0:
            raise ValueError(f"step_size must be > 0; got {step_size}")
        if max_steps <= 0:
            raise ValueError(f"max_steps must be > 0; got {max_steps}")

        self.cmap = cmap
        self.vmin = float(vmin)
        self.vmax = float(vmax)
        self.density_scale = float(density_scale)
        self.step_size = step_size
        self.max_steps = int(max_steps)
        self.opacity_lut = self._bake_opacity(opacity_stops)
        super().__init__()

    @staticmethod
    def _bake_opacity(stops) -> np.ndarray:
        """Produce a (256,) float32 array of alpha values in [0, 1]."""
        if stops is None:
            return np.linspace(0.0, 1.0, 256, dtype=np.float32)

        if isinstance(stops, np.ndarray):
            if stops.shape != (256,):
                raise ValueError(
                    f"opacity_stops array must have shape (256,); got {stops.shape}"
                )
            return stops.astype(np.float32, copy=False)

        xs = [float(s) for s, _ in stops]
        ys = [float(a) for _, a in stops]
        for i in range(1, len(xs)):
            if xs[i] < xs[i - 1]:
                raise ValueError(
                    "opacity_stops scalars must be in ascending order; "
                    f"got {xs}"
                )
        sample_xs = np.linspace(0.0, 1.0, 256, dtype=np.float32)
        return np.interp(sample_xs, xs, ys).astype(np.float32)

    @classmethod
    def _compile(cls) -> str:
        # Real shader source lives in renderer.py for v1; this class
        # carries only CPU-side parameters and the LUT bake.
        return ""

    @classmethod
    def uniform_type(cls) -> Dict[str, str]:
        return {
            "vmin": "f32",
            "vmax": "f32",
            "density_scale": "f32",
            "step_size": "f32",
            "max_steps": "u32",
            "_pad0": "f32",
            "_pad1": "f32",
            "_pad2": "f32",
        }
