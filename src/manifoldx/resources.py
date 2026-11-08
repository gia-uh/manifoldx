"""GPU resource management: Geometry, Material registries and factories."""

import numpy as np
import wgpu
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

# _SPRITE_QUAD is imported lazily inside GeometryRegistry.__init__ to avoid
# a circular import: resources -> viz.geometry -> viz -> viz.materials -> resources.


# =============================================================================
# Material Base Class
# =============================================================================


class Material(ABC):
    """Abstract base class for PBR materials."""

    binding_slot: int = -1

    @classmethod
    @abstractmethod
    def _compile(cls) -> str:
        """Return WGSL shader source code."""
        pass

    @classmethod
    @abstractmethod
    def uniform_type(cls) -> Dict[str, str]:
        """Return {field_name: wgsl_type} for uniform buffer fields."""
        pass

    def get_texture_bindings(self) -> Dict[int, Any]:
        """Return {binding_index: TextureHandle} for sampler+texture entries.

        Default: no textures. Override on materials that bind 2D textures.
        Sampler is attached at `binding_index`; texture view at `binding_index + 1`.
        """
        return {}


def srgb_to_linear(rgb) -> np.ndarray:
    """sRGB components in [0, 1] to linear light (the exact sRGB curve)."""
    c = np.asarray(rgb, dtype=np.float32)
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4).astype(np.float32)


def material_rgb(color) -> np.ndarray:
    """A material colour (hex string or float tuple, both sRGB) as linear RGB.

    Material colours mean the same as texture texels, which are uploaded as
    sRGB and decoded on sampling: `#cc2222` as `color=` and as an albedo map
    render alike.
    """
    if isinstance(color, str):
        h = color.lstrip("#")
        rgb = [int(h[0:2], 16) / 255.0, int(h[2:4], 16) / 255.0, int(h[4:6], 16) / 255.0]
    else:
        rgb = list(color[:3])
    return srgb_to_linear(rgb)


_BASICMATERIAL_SHADER = """
struct Globals {
    vp: mat4x4<f32>,
    view: mat4x4<f32>,
    camera_pos: vec3<f32>,
    _pad: f32,
};

struct Transforms {
    models: array<mat4x4<f32>>,
};

struct BasicMaterialUniforms {
    color: vec4<f32>,
};

@group(0) @binding(0) var<uniform> globals: Globals;
@group(0) @binding(1) var<storage, read> transforms: Transforms;
@group(0) @binding(2) var<uniform> material: BasicMaterialUniforms;

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @builtin(instance_index) instance: u32,
};

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) world_normal: vec3<f32>,
    @location(1) world_pos: vec3<f32>,
};

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    let model = transforms.models[in.instance];
    out.world_pos = (model * vec4<f32>(in.position, 1.0)).xyz;
    out.position = globals.vp * vec4<f32>(out.world_pos, 1.0);
    out.world_normal = normalize((model * vec4<f32>(in.normal, 0.0)).xyz);
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let normal = normalize(in.world_normal);
    let light_dir = normalize(vec3<f32>(0.5773, 0.5773, 0.5773));
    let diffuse = max(dot(normal, light_dir), 0.0);
    let brightness = 0.3 + 0.7 * diffuse;
    return vec4<f32>(material.color.rgb * brightness, material.color.a);
}
"""


class BasicMaterial(Material):
    """Flat colour with a fixed head-on light (0.3 ambient + 0.7 Lambert from (1, 1, 1))."""

    binding_slot = 0

    def __init__(self, color):
        self.color = color

    @classmethod
    def _compile(cls) -> str:
        return _BASICMATERIAL_SHADER

    @classmethod
    def uniform_type(cls) -> Dict[str, str]:
        return {"color": "vec4<f32>"}

    def get_data(self, n: int, registry) -> np.ndarray:
        """Return material data as numpy array (color as vec4)."""
        alpha = 1.0 if isinstance(self.color, str) or len(self.color) == 3 else float(self.color[3])
        color = np.append(material_rgb(self.color), alpha).astype(np.float32)
        return np.tile(color, (n, 1))


_FLATMATERIAL_SHADER = _BASICMATERIAL_SHADER.replace(
    """    let normal = normalize(in.world_normal);
    let light_dir = normalize(vec3<f32>(0.5773, 0.5773, 0.5773));
    let diffuse = max(dot(normal, light_dir), 0.0);
    let brightness = 0.3 + 0.7 * diffuse;
    return vec4<f32>(material.color.rgb * brightness, material.color.a);""",
    """    return material.color;""",
).replace("BasicMaterialUniforms", "FlatMaterialUniforms")


class FlatMaterial(BasicMaterial):
    """Outputs its colour unchanged: no lighting, no tone mapping, no gamma.

    Used for id passes, where every pixel must carry its object's exact colour.
    """

    @classmethod
    def _compile(cls) -> str:
        return _FLATMATERIAL_SHADER


_ATMOSPHERE_SHADER = """
struct Globals {
    vp: mat4x4<f32>, view: mat4x4<f32>, proj: mat4x4<f32>,
    camera_pos: vec3<f32>, _pad0: f32,
    viewport_size: vec2<f32>, _pad1: vec2<f32>,
    ibl_intensity: f32, ibl_enabled: u32, _pad_ibl: vec2<f32>,
    light_view_proj: mat4x4<f32>,
    sun_direction: vec3<f32>, _pad_sun0: f32,
    sun_color: vec3<f32>, sun_intensity: f32,
};
struct Transforms { models: array<mat4x4<f32>> };
struct GlowUniforms { params: vec4<f32> };   // rgb = colour, a = intensity

@group(0) @binding(0) var<uniform> globals: Globals;
@group(0) @binding(1) var<storage, read> transforms: Transforms;
@group(0) @binding(2) var<uniform> material: GlowUniforms;

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @builtin(instance_index) instance: u32,
};
struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) world_normal: vec3<f32>,
    @location(1) world_pos: vec3<f32>,
};

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    let model = transforms.models[in.instance];
    out.world_pos = (model * vec4<f32>(in.position, 1.0)).xyz;
    out.world_normal = normalize((model * vec4<f32>(in.normal, 0.0)).xyz);
    out.position = globals.vp * vec4<f32>(out.world_pos, 1.0);
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let N = normalize(in.world_normal);            // radial (outward)
    let V = normalize(globals.camera_pos - in.world_pos);
    let fresnel = pow(1.0 - max(dot(N, V), 0.0), 3.2);

    // Day/night: the shell glows where it faces the sun, fades on the night side.
    let sun = normalize(-globals.sun_direction);
    let face = dot(N, sun);
    let day = smoothstep(-0.25, 0.35, face);
    // Warm sunset tint near the terminator.
    let dusk = smoothstep(-0.35, 0.05, face) * (1.0 - smoothstep(0.05, 0.5, face));
    let col = mix(material.params.rgb, vec3<f32>(1.0, 0.5, 0.28), dusk * 0.7);

    let alpha = fresnel * material.params.a * clamp(day + dusk * 0.5, 0.0, 1.0);
    return vec4<f32>(col, alpha);
}
"""


_WATER_SHADER = """
struct Globals {
    vp: mat4x4<f32>, view: mat4x4<f32>, proj: mat4x4<f32>,
    camera_pos: vec3<f32>, _pad0: f32,
    viewport_size: vec2<f32>, _pad1: vec2<f32>,
    ibl_intensity: f32, ibl_enabled: u32, _pad_ibl: vec2<f32>,
    light_view_proj: mat4x4<f32>,
    sun_direction: vec3<f32>, _pad_sun0: f32,
    sun_color: vec3<f32>, sun_intensity: f32,
};
struct Transforms { models: array<mat4x4<f32>> };
struct WaterUniforms { params: vec4<f32> };   // rgb = deep colour, a = fresnel power

@group(0) @binding(0) var<uniform> globals: Globals;
@group(0) @binding(1) var<storage, read> transforms: Transforms;
@group(0) @binding(2) var<uniform> material: WaterUniforms;

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @builtin(instance_index) instance: u32,
};
struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) world_normal: vec3<f32>,
    @location(1) world_pos: vec3<f32>,
};

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    let model = transforms.models[in.instance];
    out.world_pos = (model * vec4<f32>(in.position, 1.0)).xyz;
    out.world_normal = normalize((model * vec4<f32>(in.normal, 0.0)).xyz);
    out.position = globals.vp * vec4<f32>(out.world_pos, 1.0);
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let N = normalize(in.world_normal);
    let V = normalize(globals.camera_pos - in.world_pos);
    let sun = normalize(-globals.sun_direction);
    let sc = globals.sun_color * max(globals.sun_intensity, 0.5);

    let ndl = max(dot(N, sun), 0.0);
    let day = smoothstep(-0.05, 0.30, dot(N, sun));            // 0 night, 1 day
    let deep = material.params.rgb;
    var col = deep * (0.12 + 0.88 * ndl);                      // saturated diffuse water (dark at night)

    // Fresnel reflection of the sky (sky-tinted at grazing only; keeps open water deep-blue).
    let fresnel = pow(1.0 - max(dot(N, V), 0.0), material.params.a);
    let sky = vec3<f32>(0.42, 0.60, 0.90) * day;
    col = mix(col, sky, fresnel * 0.30);

    // Sharp, small sun glint (day side only — no giant specular blob on the open sphere).
    let H = normalize(V + sun);
    let glint = pow(max(dot(N, H), 0.0), 500.0);
    col += glint * sc * 0.20 * day;

    // No desaturating reinhard — water is LDR now, so keep the deep colour saturated.
    col = clamp(col, vec3<f32>(0.0), vec3<f32>(1.0));  // the sRGB render target encodes gamma
    return vec4<f32>(col, 1.0);
}
"""


class WaterMaterial(Material):
    """Sun-aware water: day/night diffuse, fresnel sky reflection, sharp sun glint."""

    binding_slot = 0

    def __init__(self, color=(0.04, 0.16, 0.35), fresnel_power: float = 4.0):
        self.color = color
        self.fresnel_power = fresnel_power

    @classmethod
    def _compile(cls) -> str:
        return _WATER_SHADER

    @classmethod
    def uniform_type(cls) -> Dict[str, str]:
        return {"params": "vec4<f32>"}

    def get_data(self, n: int, registry) -> np.ndarray:
        rgb = material_rgb(self.color)
        return np.tile(np.array([*rgb, self.fresnel_power], dtype=np.float32), (n, 1))


_ATMO_SCATTER_SHADER = """
struct Globals {
    vp: mat4x4<f32>, view: mat4x4<f32>, proj: mat4x4<f32>,
    camera_pos: vec3<f32>, _pad0: f32,
    viewport_size: vec2<f32>, _pad1: vec2<f32>,
    ibl_intensity: f32, ibl_enabled: u32, _pad_ibl: vec2<f32>,
    light_view_proj: mat4x4<f32>,
    sun_direction: vec3<f32>, _pad_sun0: f32,
    sun_color: vec3<f32>, sun_intensity: f32,
};
struct Transforms { models: array<mat4x4<f32>> };
struct AtmoUniforms { params: vec4<f32> };   // x=Rg, y=Ra, z=intensity, w=exposure

@group(0) @binding(0) var<uniform> globals: Globals;
@group(0) @binding(1) var<storage, read> transforms: Transforms;
@group(0) @binding(2) var<uniform> material: AtmoUniforms;

const PI: f32 = 3.14159265;

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @builtin(instance_index) instance: u32,
};
struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) world_pos: vec3<f32>,
};

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    let model = transforms.models[in.instance];
    out.world_pos = (model * vec4<f32>(in.position, 1.0)).xyz;
    out.position = globals.vp * vec4<f32>(out.world_pos, 1.0);
    return out;
}

// (t_near, t_far) of ray o+t*d with a sphere of radius r at the origin; near>far on miss.
fn ray_sphere(o: vec3<f32>, d: vec3<f32>, r: f32) -> vec2<f32> {
    let b = dot(o, d);
    let c = dot(o, o) - r * r;
    let disc = b * b - c;
    if disc < 0.0 { return vec2<f32>(1.0, -1.0); }
    let s = sqrt(disc);
    return vec2<f32>(-b - s, -b + s);
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let Rg = material.params.x;
    let Ra = material.params.y;
    let cam = globals.camera_pos;
    let dir = normalize(in.world_pos - cam);
    let sun = normalize(-globals.sun_direction);

    let Hr = 0.06 * Rg;
    let Hm = 0.012 * Rg;
    let betaR = vec3<f32>(5.5, 13.0, 30.0) / Rg;
    let betaM = vec3<f32>(0.4, 0.4, 0.4) / Rg;
    let g = 0.62;

    let atm = ray_sphere(cam, dir, Ra);
    if atm.y < atm.x || atm.y < 0.0 { return vec4<f32>(0.0); }
    var t0 = max(atm.x, 0.0);
    var t1 = atm.y;
    let gnd = ray_sphere(cam, dir, Rg);
    let hits_ground = gnd.x > 0.0 && gnd.x < t1;
    if hits_ground { t1 = gnd.x; }
    if t1 <= t0 { return vec4<f32>(0.0); }
    // Rays that strike the planet (the disc) get their aerial haze knocked down so the
    // surface keeps its true colours; rays that miss (the limb from space, every upward
    // ray at the surface) keep the full sky/rim. View-dependent, so it holds at any altitude.
    let haze = select(1.0, 0.14, hits_ground);
    // Altitude fade: full sky near the surface; from space fade to a thin rim so the
    // planet disc keeps its true colours instead of a milky additive veil.
    let cam_alt = length(cam) - Rg;
    let atmo_h = Ra - Rg;
    let alt_gate = mix(1.0, 0.16, smoothstep(atmo_h, atmo_h * 3.0, cam_alt));

    let seg = (t1 - t0) / 16.0;
    var odR = 0.0;
    var odM = 0.0;
    var sumR = vec3<f32>(0.0);
    var sumM = vec3<f32>(0.0);
    let mu = dot(dir, sun);
    let phaseR = 3.0 / (16.0 * PI) * (1.0 + mu * mu);
    let g2 = g * g;
    let phaseM = 3.0 / (8.0 * PI) * ((1.0 - g2) * (1.0 + mu * mu))
                 / ((2.0 + g2) * pow(1.0 + g2 - 2.0 * g * mu, 1.5));

    for (var i: i32 = 0; i < 16; i = i + 1) {
        let p = cam + dir * (t0 + seg * (f32(i) + 0.5));
        let h = length(p) - Rg;
        let dR = exp(-h / Hr) * seg;
        let dM = exp(-h / Hm) * seg;
        odR = odR + dR;
        odM = odM + dM;

        var sodR = 1.0e9;
        var sodM = 1.0e9;
        let sground = ray_sphere(p, sun, Rg);
        if sground.x <= 0.0 {                       // sun above the local horizon
            let satm = ray_sphere(p, sun, Ra);
            let sseg = satm.y / 8.0;
            sodR = 0.0;
            sodM = 0.0;
            for (var j: i32 = 0; j < 8; j = j + 1) {
                let sp = p + sun * (sseg * (f32(j) + 0.5));
                let sh = length(sp) - Rg;
                sodR = sodR + exp(-sh / Hr) * sseg;
                sodM = sodM + exp(-sh / Hm) * sseg;
            }
        }
        let tau = betaR * (odR + sodR) + betaM * 1.1 * (odM + sodM);
        let attn = exp(-tau);
        sumR = sumR + attn * dR;
        sumM = sumM + attn * dM;
    }

    var col = material.params.z * haze * alt_gate * max(globals.sun_intensity, 0.5)
              * (sumR * betaR * phaseR + sumM * betaM * phaseM);

    // Art-directed sunset warmth: the physical shadow test cuts the sky to black the
    // instant the sun dips, so paint a warm glow near the horizon toward the sun,
    // peaking as the sun crosses it. Additive, gated like the rest of the sky.
    let up = normalize(cam);
    let sun_elev = dot(up, sun);
    let view_elev = dot(up, dir);
    let toward = max(mu, 0.0);
    let dusk = exp(-(sun_elev * sun_elev) / (2.0 * 0.12 * 0.12));       // peaks at the horizon
    let band = clamp(1.0 - max(view_elev, 0.0) / 0.40, 0.0, 1.0);       // strongest low in the sky
    let warm = vec3<f32>(0.9, 0.34, 0.12) * dusk * pow(toward, 3.0) * band;
    col = col + warm * material.params.z * haze * alt_gate;

    col = vec3<f32>(1.0) - exp(-col * material.params.w);   // exposure tonemap
    // gamma: the sRGB render target encodes it
    let lum = dot(col, vec3<f32>(0.2126, 0.7152, 0.0722));  // punch up saturation (art-directed)
    col = max(mix(vec3<f32>(lum), col, 1.4), vec3<f32>(0.0));
    return vec4<f32>(col, 1.0);
}
"""


class AtmosphereScatteringMaterial(Material):
    """Physically-based single-scattering atmosphere (Rayleigh + Mie), rendered on a shell."""

    binding_slot = 0

    def __init__(
        self, ground_radius, atmosphere_radius, intensity: float = 22.0, exposure: float = 1.0
    ):
        self.ground_radius = ground_radius
        self.atmosphere_radius = atmosphere_radius
        self.intensity = intensity
        self.exposure = exposure

    @property
    def pipeline_subtype(self):
        return "scatter"

    @classmethod
    def _compile(cls) -> str:
        return _ATMO_SCATTER_SHADER

    @classmethod
    def uniform_type(cls) -> Dict[str, str]:
        return {"params": "vec4<f32>"}

    def get_data(self, n: int, registry) -> np.ndarray:
        return np.tile(
            np.array(
                [self.ground_radius, self.atmosphere_radius, self.intensity, self.exposure],
                dtype=np.float32,
            ),
            (n, 1),
        )


_CLOUD_SHADER = """
struct Globals {
    vp: mat4x4<f32>, view: mat4x4<f32>, proj: mat4x4<f32>,
    camera_pos: vec3<f32>, _pad0: f32,
    viewport_size: vec2<f32>, _pad1: vec2<f32>,
    ibl_intensity: f32, ibl_enabled: u32, _pad_ibl: vec2<f32>,
    light_view_proj: mat4x4<f32>,
    sun_direction: vec3<f32>, _pad_sun0: f32,
    sun_color: vec3<f32>, sun_intensity: f32,
};
struct Transforms { models: array<mat4x4<f32>> };
struct CloudUniforms { params: vec4<f32> };   // x=coverage, y=softness, z=freq, w=opacity

@group(0) @binding(0) var<uniform> globals: Globals;
@group(0) @binding(1) var<storage, read> transforms: Transforms;
@group(0) @binding(2) var<uniform> material: CloudUniforms;

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @builtin(instance_index) instance: u32,
};
struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) world_normal: vec3<f32>,
    @location(1) local_pos: vec3<f32>,
};

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    let model = transforms.models[in.instance];
    out.local_pos = in.position;                                // object space: pattern rides the shell
    out.world_normal = normalize((model * vec4<f32>(in.normal, 0.0)).xyz);
    out.position = globals.vp * vec4<f32>((model * vec4<f32>(in.position, 1.0)).xyz, 1.0);
    return out;
}

fn hash(p: vec3<f32>) -> f32 {
    var p3 = fract(p * vec3<f32>(0.1031, 0.1030, 0.0973));
    p3 = p3 + dot(p3, p3.yxz + 33.33);
    return fract((p3.x + p3.y) * p3.z);
}
fn vnoise(p: vec3<f32>) -> f32 {
    let i = floor(p);
    let f = fract(p);
    let u = f * f * (3.0 - 2.0 * f);
    let n000 = hash(i + vec3<f32>(0.0, 0.0, 0.0));
    let n100 = hash(i + vec3<f32>(1.0, 0.0, 0.0));
    let n010 = hash(i + vec3<f32>(0.0, 1.0, 0.0));
    let n110 = hash(i + vec3<f32>(1.0, 1.0, 0.0));
    let n001 = hash(i + vec3<f32>(0.0, 0.0, 1.0));
    let n101 = hash(i + vec3<f32>(1.0, 0.0, 1.0));
    let n011 = hash(i + vec3<f32>(0.0, 1.0, 1.0));
    let n111 = hash(i + vec3<f32>(1.0, 1.0, 1.0));
    let x00 = mix(n000, n100, u.x);
    let x10 = mix(n010, n110, u.x);
    let x01 = mix(n001, n101, u.x);
    let x11 = mix(n011, n111, u.x);
    return mix(mix(x00, x10, u.y), mix(x01, x11, u.y), u.z);
}
fn fbm(p0: vec3<f32>) -> f32 {
    var s = 0.0;
    var a = 0.5;
    var q = p0;
    for (var k: i32 = 0; k < 5; k = k + 1) {
        s = s + a * vnoise(q);
        q = q * 2.02;
        a = a * 0.5;
    }
    return s;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let dir = normalize(in.local_pos);
    let freq = material.params.z;
    // Two-octave-warped fbm for puffy, clumped coverage.
    let warp = fbm(dir * freq * 0.5) * 0.6;
    let d = fbm(dir * freq + vec3<f32>(warp));
    let cover = smoothstep(material.params.x, material.params.x + material.params.y, d);
    if cover < 0.02 { discard; }

    let N = normalize(in.world_normal);
    let sun = normalize(-globals.sun_direction);
    let day = smoothstep(-0.15, 0.30, dot(N, sun));            // 0 night, 1 day
    // Bright sunlit tops, cool shadowed undersides at the terminator, dark at night.
    let lit = mix(vec3<f32>(0.55, 0.60, 0.72), vec3<f32>(1.0, 0.99, 0.96), day);
    let alpha = cover * material.params.w * clamp(0.25 + 0.75 * day, 0.0, 1.0);
    return vec4<f32>(lit, alpha);
}
"""


class CloudMaterial(Material):
    """Procedural cloud shell: fbm coverage, sun-lit, alpha-blended. Rotate the shell to drift."""

    binding_slot = 0

    def __init__(
        self,
        coverage: float = 0.55,
        softness: float = 0.12,
        freq: float = 3.0,
        opacity: float = 0.9,
    ):
        self.coverage = coverage
        self.softness = softness
        self.freq = freq
        self.opacity = opacity

    @property
    def pipeline_subtype(self):
        return "cloud"

    @classmethod
    def _compile(cls) -> str:
        return _CLOUD_SHADER

    @classmethod
    def uniform_type(cls) -> Dict[str, str]:
        return {"params": "vec4<f32>"}

    def get_data(self, n: int, registry) -> np.ndarray:
        return np.tile(
            np.array([self.coverage, self.softness, self.freq, self.opacity], dtype=np.float32),
            (n, 1),
        )


class AtmosphereMaterial(Material):
    """Unlit fresnel rim-glow (blended) — an atmosphere halo shell."""

    binding_slot = 0

    def __init__(self, color, intensity: float = 1.0):
        self.color = color
        self.intensity = intensity

    @property
    def pipeline_subtype(self):
        return "glow"

    @classmethod
    def _compile(cls) -> str:
        return _ATMOSPHERE_SHADER

    @classmethod
    def uniform_type(cls) -> Dict[str, str]:
        return {"params": "vec4<f32>"}

    def get_data(self, n: int, registry) -> np.ndarray:
        rgb = material_rgb(self.color)
        return np.tile(np.array([*rgb, self.intensity], dtype=np.float32), (n, 1))


_STANDARDMATERIAL_SHADER = """
struct Globals {
    vp:              mat4x4<f32>,   // offset   0
    view:            mat4x4<f32>,   // offset  64
    proj:            mat4x4<f32>,   // offset 128
    camera_pos:      vec3<f32>,     // offset 192
    time:            f32,           // offset 204  (seconds, wraps every 64)
    viewport_size:   vec2<f32>,     // offset 208
    origin_mod:      vec2<f32>,     // offset 216  (the floating origin's x, z mod 256)
    ibl_intensity:   f32,           // offset 224
    ibl_enabled:     u32,           // offset 228
    _pad_ibl:        vec2<f32>,     // offset 232
    light_view_proj: mat4x4<f32>,   // offset 240
    sun_direction:   vec3<f32>,     // offset 304
    _pad_sun0:       f32,           // offset 316
    sun_color:       vec3<f32>,     // offset 320
    sun_intensity:   f32,           // offset 332
    shadow_enabled:  u32,           // offset 336
    shadow_bias:     f32,           // offset 340
    shadow_map_size: f32,           // offset 344
    shadow_pcf_radius: u32,         // offset 348
    spot_position:   vec3<f32>,     // offset 352
    spot_range:      f32,           // offset 364
    spot_direction:  vec3<f32>,     // offset 368
    spot_cos_inner:  f32,           // offset 380
    spot_color:      vec3<f32>,     // offset 384
    spot_intensity:  f32,           // offset 396
    spot_cos_outer:  f32,           // offset 400
    shadow_caster:   u32,           // offset 404  (0=none, 1=sun, 2=spot)
    fog_start:       f32,           // offset 408
    fog_end:         f32,           // offset 412
    fog_color:       vec3<f32>,     // offset 416
    fog_enabled:     u32,           // offset 428
    benders:         array<vec4<f32>, 8>,  // offset 432  (x, z, radius, strength) in the render frame
};

struct Transforms {
    models: array<mat4x4<f32>>,
};

struct PBRMaterialUniforms {
    albedo:    vec3<f32>,
    roughness: f32,
    metallic:  f32,
    ao:        f32,
    sway:      f32,
    sway_reach: f32,
    ripple:    f32,
    mirror:    f32,     // 1.0: drawn reflected in the plane y = mirror_y
    mirror_y:  f32,
    opacity:   f32,
    cut:       vec4<f32>,   // x0, z0, x1, z1: an xz rectangle where nothing draws (empty when x1 <= x0)
};

struct PointLightData {
    position:  vec3<f32>,
    _pad0:     f32,
    color:     vec3<f32>,
    intensity: f32,
};

struct LightData {
    lights: array<PointLightData, 4>,
};

@group(0) @binding(0) var<uniform> globals:    Globals;
@group(0) @binding(1) var<storage, read> transforms: Transforms;
@group(0) @binding(2) var<uniform> material:  PBRMaterialUniforms;
@group(0) @binding(3) var<uniform> light_data: LightData;

@group(1) @binding(0) var env_sampler:     sampler;
@group(1) @binding(1) var irradiance_map:  texture_cube<f32>;
@group(1) @binding(2) var prefiltered_map: texture_cube<f32>;
@group(1) @binding(3) var brdf_lut:        texture_2d<f32>;

@group(2) @binding(0) var shadow_map:     texture_depth_2d;
@group(2) @binding(1) var shadow_sampler: sampler_comparison;

const PI: f32 = 3.14159265359;

fn distributionGGX(N: vec3<f32>, H: vec3<f32>, roughness: f32) -> f32 {
    let a = roughness * roughness;
    let a2 = a * a;
    let NdotH = max(dot(N, H), 0.0);
    let NdotH2 = NdotH * NdotH;
    let denom_base = NdotH2 * (a2 - 1.0) + 1.0;
    return a2 / (PI * denom_base * denom_base);
}

fn geometrySchlickGGX(NdotV: f32, roughness: f32) -> f32 {
    let k = (roughness + 1.0) * (roughness + 1.0) / 8.0;
    return NdotV / (NdotV * (1.0 - k) + k);
}

fn geometrySmith(N: vec3<f32>, V: vec3<f32>, L: vec3<f32>, roughness: f32) -> f32 {
    return geometrySchlickGGX(max(dot(N, V), 0.0), roughness)
         * geometrySchlickGGX(max(dot(N, L), 0.0), roughness);
}

fn fresnelSchlick(cosTheta: f32, F0: vec3<f32>) -> vec3<f32> {
    return F0 + (vec3<f32>(1.0) - F0) * pow(clamp(1.0 - cosTheta, 0.0, 1.0), 5.0);
}

fn fresnelSchlickRoughness(cosTheta: f32, F0: vec3<f32>, roughness: f32) -> vec3<f32> {
    let r1 = vec3<f32>(1.0 - roughness);
    return F0 + (max(r1, F0) - F0) * pow(clamp(1.0 - cosTheta, 0.0, 1.0), 5.0);
}

fn calculatePointLight(N: vec3<f32>, V: vec3<f32>, worldPos: vec3<f32>,
                       F0: vec3<f32>, albedo: vec3<f32>, metallic: f32,
                       roughness: f32, light: PointLightData) -> vec3<f32> {
    let L = normalize(light.position - worldPos);
    let H = normalize(V + L);
    let dist = length(light.position - worldPos);
    let radiance = light.color * light.intensity / (dist * dist);
    let NDF = distributionGGX(N, H, roughness);
    let G   = geometrySmith(N, V, L, roughness);
    let F   = fresnelSchlick(max(dot(H, V), 0.0), F0);
    let kD  = (vec3<f32>(1.0) - F) * (1.0 - metallic);
    let specular = NDF * G * F / (4.0 * max(dot(N, V), 0.0) * max(dot(N, L), 0.0) + 0.0001);
    return (kD * albedo / PI + specular) * radiance * max(dot(N, L), 0.0);
}

fn calculateSun(N: vec3<f32>, V: vec3<f32>, F0: vec3<f32>, albedo: vec3<f32>,
                metallic: f32, roughness: f32) -> vec3<f32> {
    // Directional light: L points toward the light = -direction. No distance falloff.
    let L = normalize(-globals.sun_direction);
    let H = normalize(V + L);
    let radiance = globals.sun_color * globals.sun_intensity;
    let NDF = distributionGGX(N, H, roughness);
    let G   = geometrySmith(N, V, L, roughness);
    let F   = fresnelSchlick(max(dot(H, V), 0.0), F0);
    let kD  = (vec3<f32>(1.0) - F) * (1.0 - metallic);
    let specular = NDF * G * F / (4.0 * max(dot(N, V), 0.0) * max(dot(N, L), 0.0) + 0.0001);
    return (kD * albedo / PI + specular) * radiance * max(dot(N, L), 0.0);
}

fn calculateSpot(N: vec3<f32>, V: vec3<f32>, worldPos: vec3<f32>, F0: vec3<f32>,
                 albedo: vec3<f32>, metallic: f32, roughness: f32) -> vec3<f32> {
    let toL = globals.spot_position - worldPos;
    let dist = length(toL);
    let L = toL / dist;
    // Flashlight cone: smooth falloff between the inner and outer cone angles.
    let cos_angle = dot(-L, normalize(globals.spot_direction));
    let t = clamp((cos_angle - globals.spot_cos_outer)
                  / max(globals.spot_cos_inner - globals.spot_cos_outer, 0.0001), 0.0, 1.0);
    let cone = t * t;  // squared for a softer edge
    // Distance attenuation with a smooth range cutoff.
    let range_falloff = clamp(1.0 - pow(dist / globals.spot_range, 4.0), 0.0, 1.0);
    let atten = range_falloff * range_falloff / (dist * dist + 0.0001);
    let radiance = globals.spot_color * globals.spot_intensity * cone * atten;
    let H = normalize(V + L);
    let NDF = distributionGGX(N, H, roughness);
    let G   = geometrySmith(N, V, L, roughness);
    let F   = fresnelSchlick(max(dot(H, V), 0.0), F0);
    let kD  = (vec3<f32>(1.0) - F) * (1.0 - metallic);
    let specular = NDF * G * F / (4.0 * max(dot(N, V), 0.0) * max(dot(N, L), 0.0) + 0.0001);
    return (kD * albedo / PI + specular) * radiance * max(dot(N, L), 0.0);
}

fn shadowFactor(world_pos: vec3<f32>, N: vec3<f32>) -> f32 {
    if globals.shadow_enabled == 0u { return 1.0; }
    let lp = globals.light_view_proj * vec4<f32>(world_pos, 1.0);
    let ndc = lp.xyz / lp.w;
    let uv = vec2<f32>(ndc.x * 0.5 + 0.5, -ndc.y * 0.5 + 0.5);
    if uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0 || ndc.z > 1.0 || ndc.z < 0.0 {
        return 1.0;  // outside the caster's frustum — unshadowed
    }
    // Slope-scaled bias, using the direction toward the casting light so grazing
    // surfaces don't self-shadow (acne) without peter-panning on lit faces.
    var Lc = normalize(-globals.sun_direction);
    if globals.shadow_caster == 2u {
        Lc = normalize(globals.spot_position - world_pos);
    }
    let ndl = max(dot(N, Lc), 0.0);
    let bias = globals.shadow_bias * (1.0 + 4.0 * (1.0 - ndl));
    // PCF: average an (2r+1)x(2r+1) grid of comparisons for soft edges.
    // r == 0 collapses to a single hard-shadow tap. textureSampleCompareLevel
    // (explicit level, no derivatives) is legal in non-uniform control flow.
    let ref_depth = ndc.z - bias;
    let texel = 1.0 / globals.shadow_map_size;
    let r = i32(globals.shadow_pcf_radius);
    var sum = 0.0;
    var count = 0.0;
    for (var dy = -r; dy <= r; dy = dy + 1) {
        for (var dx = -r; dx <= r; dx = dx + 1) {
            let off = vec2<f32>(f32(dx), f32(dy)) * texel;
            sum += textureSampleCompareLevel(shadow_map, shadow_sampler, uv + off, ref_depth);
            count += 1.0;
        }
    }
    // Fade to unshadowed over the outer 15% of the map, so the edge of what it
    // covers never shows on screen as a line where the shadows stop.
    let edge = min(min(uv.x, 1.0 - uv.x), min(uv.y, 1.0 - uv.y));
    return mix(1.0, sum / count, smoothstep(0.0, 0.15, edge));
}

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) normal:   vec3<f32>,
    @builtin(instance_index) instance: u32,
};

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) world_normal: vec3<f32>,
    @location(1) world_pos:    vec3<f32>,
};

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    let model = transforms.models[in.instance];
    out.world_pos = (model * vec4<f32>(in.position, 1.0)).xyz;
    if (material.sway > 0.0) {
        // Wind: a vertex moves with its height above the model's origin, in proportion
        // near the ground and levelling off past sway_reach (reach * tanh(height / reach)),
        // so grass bends as much as a tree; the phase comes from where it stands in the
        // world, so neighbours sway apart.
        // Every wave repeats over 64 s and 256 world units, which time and origin_mod
        // wrap by, so neither the clock nor a moved origin ever makes it jump.
        let TAU = 6.28318530718;
        let p = out.world_pos.xz + globals.origin_mod;
        let t = globals.time;
        let a = TAU * (27.0 * t / 64.0 - (23.0 * p.x + 13.0 * p.y) / 256.0);
        let b = TAU * (61.0 * t / 64.0 - (-31.0 * p.x + 41.0 * p.y) / 256.0);
        let reach = material.sway_reach;
        let h = reach * tanh(max(in.position.y, 0.0) / reach) * material.sway;
        let base = out.world_pos.xz;
        out.world_pos.x += h * (sin(a) + 0.35 * sin(b));
        out.world_pos.z += h * 0.5 * (cos(a) + 0.35 * sin(b + 1.3));
        // Benders part what is low (grass, flowers; not a tree's crown) away from each point,
        // most at its centre and nothing at its radius, bowing it down a little as it leans.
        let low = reach * tanh(max(in.position.y, 0.0) / reach) * (1.0 - smoothstep(0.8, 1.2, in.position.y));
        for (var i = 0u; i < 8u; i = i + 1u) {
            let bender = globals.benders[i];
            let off = base - bender.xy;
            let d = length(off);
            if (bender.z <= 0.0 || d >= bender.z || d < 1e-4) {
                continue;
            }
            let f = 1.0 - d / bender.z;
            let lean = bender.w * f * f * low;
            out.world_pos.x += off.x / d * lean;
            out.world_pos.z += off.y / d * lean;
            out.world_pos.y -= 0.4 * lean;
        }
    }
    // A mirrored mesh is lit where it stands, but drawn reflected in the plane y = mirror_y.
    var seen = out.world_pos;
    if (material.mirror > 0.0) {
        seen.y = 2.0 * material.mirror_y - seen.y;
    }
    out.position  = globals.vp * vec4<f32>(seen, 1.0);
    out.world_normal = normalize((model * vec4<f32>(in.normal, 0.0)).xyz);
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    if (material.mirror > 0.0 && in.world_pos.y < material.mirror_y) {
        discard;   // below the mirror: its reflection would stand above it
    }
    if (all(in.world_pos.xz > material.cut.xy) && all(in.world_pos.xz < material.cut.zw)) {
        discard;
    }
    let N = normalize(in.world_normal);
    let V = normalize(globals.camera_pos - in.world_pos);

    let F0 = mix(vec3<f32>(0.04), material.albedo, material.metallic);

    var Lo = vec3<f32>(0.0);
    for (var i = 0u; i < 4u; i++) {
        let light = light_data.lights[i];
        if light.intensity > 0.0 {
            Lo += calculatePointLight(N, V, in.world_pos, F0,
                                      material.albedo, material.metallic,
                                      material.roughness, light);
        }
    }

    if globals.sun_intensity > 0.0 {
        var s = 1.0;
        if globals.shadow_caster == 1u { s = shadowFactor(in.world_pos, N); }
        Lo += calculateSun(N, V, F0, material.albedo, material.metallic, material.roughness) * s;
    }

    if globals.spot_intensity > 0.0 {
        var s = 1.0;
        if globals.shadow_caster == 2u { s = shadowFactor(in.world_pos, N); }
        Lo += calculateSpot(N, V, in.world_pos, F0, material.albedo,
                            material.metallic, material.roughness) * s;
    }

    var ambient = vec3<f32>(0.03) * material.albedo * material.ao;

    if globals.ibl_enabled != 0u {
        let NdotV = max(dot(N, V), 0.0001);
        let R     = reflect(-V, N);
        let F     = fresnelSchlickRoughness(NdotV, F0, material.roughness);
        let kD    = (1.0 - F) * (1.0 - material.metallic);

        let irradiance  = textureSample(irradiance_map, env_sampler, N).rgb;
        let diffuse_ibl = kD * irradiance * material.albedo;

        let MAX_MIPS: f32 = 7.0;
        let prefilt   = textureSampleLevel(prefiltered_map, env_sampler, R,
                                           material.roughness * MAX_MIPS).rgb;
        let brdf_samp = textureSample(brdf_lut, env_sampler,
                                      vec2<f32>(NdotV, material.roughness)).rg;
        let spec_ibl  = prefilt * (F * brdf_samp.r + brdf_samp.g);

        ambient = (diffuse_ibl + spec_ibl) * material.ao * globals.ibl_intensity;
    }

    var color = ambient + Lo;
    if (material.ripple > 0.0) {
        // Water: thin bands of light drift across the surface, one wave train bent by a
        // second so the crests wobble. As with the wind, every wave repeats over 64 s and
        // 256 world units, so neither the clock nor a moved origin makes them jump.
        let TAU = 6.28318530718;
        let p = in.world_pos.xz + globals.origin_mod;
        let t = globals.time;
        let bend = 1.5 * sin(TAU * ((96.0 * p.x - 24.0 * p.y) / 256.0 + 9.0 * t / 64.0));
        let w = sin(TAU * ((40.0 * p.x + 160.0 * p.y) / 256.0 - 16.0 * t / 64.0) + bend);
        // A third wave, running along the crests, breaks them into dashes.
        let gap = sin(TAU * ((112.0 * p.x + 8.0 * p.y) / 256.0 + 5.0 * t / 64.0));
        let band = smoothstep(0.86, 0.93, w) * smoothstep(-0.35, 0.05, gap);
        color += material.ripple * band * (0.25 * color + vec3<f32>(0.12, 0.16, 0.18));
    }
    color = color / (color + vec3<f32>(1.0));
    // gamma: the sRGB render target encodes it
    if globals.fog_enabled != 0u {
        let fd = clamp((distance(globals.camera_pos, in.world_pos) - globals.fog_start)
                       / (globals.fog_end - globals.fog_start), 0.0, 1.0);
        color = mix(color, globals.fog_color, fd);
    }
    return vec4<f32>(color, material.opacity);
}
"""


_STANDARDMATERIAL_TEXTURED_SHADER = (
    _STANDARDMATERIAL_SHADER.replace(
        "@group(0) @binding(3) var<uniform> light_data: LightData;",
        "@group(0) @binding(3) var<uniform> light_data: LightData;\n"
        "@group(0) @binding(4) var albedo_sampler: sampler;\n"
        "@group(0) @binding(5) var albedo_tex: texture_2d<f32>;",
    )
    .replace(
        "    @location(1) normal:   vec3<f32>,\n    @builtin(instance_index) instance: u32,",
        "    @location(1) normal:   vec3<f32>,\n"
        "    @location(2) uv:       vec2<f32>,\n"
        "    @builtin(instance_index) instance: u32,",
    )
    .replace(
        "    @location(1) world_pos:    vec3<f32>,\n};",
        "    @location(1) world_pos:    vec3<f32>,\n    @location(2) uv:           vec2<f32>,\n};",
    )
    .replace(
        "    out.world_normal = normalize((model * vec4<f32>(in.normal, 0.0)).xyz);\n"
        "    return out;",
        "    out.world_normal = normalize((model * vec4<f32>(in.normal, 0.0)).xyz);\n"
        "    out.uv = in.uv;\n"
        "    return out;",
    )
    .replace(
        "    let F0 = mix(vec3<f32>(0.04), material.albedo, material.metallic);",
        "    let albedo_texel = textureSample(albedo_tex, albedo_sampler, in.uv);\n"
        "    // Texels under half alpha are holes; translucent materials blend the rest.\n"
        "    if albedo_texel.a < 0.5 { discard; }\n"
        "    let sampled_albedo = albedo_texel.rgb;\n"
        "    let F0 = mix(vec3<f32>(0.04), sampled_albedo, material.metallic);",
    )
    .replace(
        "    return vec4<f32>(color, material.opacity);",
        "    return vec4<f32>(color, material.opacity * albedo_texel.a);",
    )
    .replace(
        "Lo += calculatePointLight(N, V, in.world_pos, F0,\n"
        "                                      material.albedo, material.metallic,\n"
        "                                      material.roughness, light);",
        "Lo += calculatePointLight(N, V, in.world_pos, F0,\n"
        "                                      sampled_albedo, material.metallic,\n"
        "                                      material.roughness, light);",
    )
    .replace(
        "Lo += calculateSun(N, V, F0, material.albedo, material.metallic, material.roughness) * s;",
        "Lo += calculateSun(N, V, F0, sampled_albedo, material.metallic, material.roughness) * s;",
    )
    .replace(
        "Lo += calculateSpot(N, V, in.world_pos, F0, material.albedo,\n"
        "                            material.metallic, material.roughness) * s;",
        "Lo += calculateSpot(N, V, in.world_pos, F0, sampled_albedo,\n"
        "                            material.metallic, material.roughness) * s;",
    )
    .replace(
        "        let diffuse_ibl = kD * irradiance * material.albedo;",
        "        let diffuse_ibl = kD * irradiance * sampled_albedo;",
    )
    .replace(
        "    var ambient = vec3<f32>(0.03) * material.albedo * material.ao;",
        "    var ambient = vec3<f32>(0.03) * sampled_albedo * material.ao;",
    )
)


_SURFACE_PATTERNS_WGSL = """
// ---- surface patterns: world-space comic textures, picked by a per-vertex id ----
const SP_INK = vec3<f32>(0.0080, 0.0063, 0.0112);  // #1c1a21 as pow(c, 2.2), as the reference

fn sp_mod(x: f32, y: f32) -> f32 { return x - y * floor(x / y); }
fn sp_h1(p: vec2<f32>) -> f32 { return fract(sin(dot(p, vec2<f32>(127.1, 311.7))) * 43758.5453); }
fn sp_h2(p: vec2<f32>) -> vec2<f32> {
    return fract(sin(vec2<f32>(dot(p, vec2<f32>(127.1, 311.7)), dot(p, vec2<f32>(269.5, 183.3)))) * 43758.5453);
}
fn sp_vnoise(p: vec2<f32>) -> f32 {
    let i = floor(p);
    var f = fract(p);
    f = f * f * (3.0 - 2.0 * f);
    return mix(mix(sp_h1(i), sp_h1(i + vec2<f32>(1.0, 0.0)), f.x),
               mix(sp_h1(i + vec2<f32>(0.0, 1.0)), sp_h1(i + vec2<f32>(1.0, 1.0)), f.x), f.y);
}
fn sp_fbm(p: vec2<f32>) -> f32 {
    return 0.55 * sp_vnoise(p) + 0.3 * sp_vnoise(p * 2.03 + 7.1) + 0.15 * sp_vnoise(p * 4.1 + 3.7);
}
fn sp_seg(p: vec2<f32>, a: vec2<f32>, b: vec2<f32>) -> f32 {
    let pa = p - a;
    let ba = b - a;
    return length(pa - ba * clamp(dot(pa, ba) / dot(ba, ba), 0.0, 1.0));
}
// Each pattern returns (tone 0..1, ink 0..1).
fn sp_grass(u: vec2<f32>, aa: f32, w: f32) -> vec2<f32> {
    let t = sp_fbm(u * 0.6);
    let c = floor(u * 1.6);
    let f = fract(u * 1.6);
    var ink = 0.0;
    if (sp_h1(c + 3.1) > 0.45) {
        let p = f - (0.25 + 0.5 * sp_h2(c));
        let d = min(min(sp_seg(p, vec2<f32>(0.0), vec2<f32>(-0.09, 0.16)),
                        sp_seg(p, vec2<f32>(0.0), vec2<f32>(0.0, 0.2))),
                    sp_seg(p, vec2<f32>(0.0), vec2<f32>(0.09, 0.16)));
        ink = 1.0 - smoothstep(w * 1.6, w * 1.6 + aa * 1.6, d);
    }
    return vec2<f32>(t, ink);
}
fn sp_cobbles(u: vec2<f32>, aa: f32, w: f32) -> vec2<f32> {
    let i = floor(u);
    let f = fract(u);
    var f1 = 9.0;
    var f2 = 9.0;
    var best = vec2<f32>(0.0);
    for (var y = -1; y <= 1; y++) {
        for (var x = -1; x <= 1; x++) {
            let g = vec2<f32>(f32(x), f32(y));
            let r = g + 0.15 + 0.7 * sp_h2(i + g) - f;
            let d = dot(r, r);
            if (d < f1) { f2 = f1; f1 = d; best = i + g; } else if (d < f2) { f2 = d; }
        }
    }
    return vec2<f32>(sp_h1(best), 1.0 - smoothstep(w, w + aa, sqrt(f2) - sqrt(f1)));
}
fn sp_veins(u: vec2<f32>, aa: f32, w: f32) -> vec2<f32> {
    let t = sp_fbm(u * 0.45);
    let v = abs(sp_vnoise(u * 0.7 + 3.0 * sp_fbm(u * 0.3)) - 0.5);
    return vec2<f32>(t, 1.0 - smoothstep(w * 0.5, w * 0.5 + aa, v));
}
fn sp_bricks(u: vec2<f32>, size: vec2<f32>, aa: f32, w: f32) -> vec2<f32> {
    var q = u / size;
    let row = floor(q.y);
    q.x += 0.5 * sp_mod(row, 2.0);
    let f = fract(q);
    let c = floor(q);
    let dx = min(f.x, 1.0 - f.x) * size.x;
    let dy = min(f.y, 1.0 - f.y) * size.y;
    return vec2<f32>(sp_h1(c + row * 7.0), 1.0 - smoothstep(w, w + aa, min(dx, dy)));
}
fn sp_planks(u: vec2<f32>, aa: f32, w: f32) -> vec2<f32> {
    let bw = 0.4;
    let row = floor(u.y / bw);
    let y = fract(u.y / bw) * bw;
    let len = 3.0;
    let x = u.x / len + sp_h1(vec2<f32>(row, 1.0));
    let fx = fract(x) * len;
    let d = min(min(y, bw - y), min(fx, len - fx));
    let grain = sp_vnoise(vec2<f32>(u.x * 0.6, u.y * 9.0) + row * 13.0);
    return vec2<f32>(0.35 * sp_h1(vec2<f32>(row, floor(x))) + 0.65 * grain,
                     1.0 - smoothstep(w, w + aa, d));
}
fn sp_tiles(u: vec2<f32>, aa: f32, w: f32) -> vec2<f32> {
    let size = vec2<f32>(0.6, 0.45);
    var q = u / size;
    let row = floor(q.y);
    q.x += 0.5 * sp_mod(row, 2.0);
    let f = fract(q);
    let c = floor(q);
    let e = 2.0 * f.x - 1.0;
    let edge_y = 0.28 * (1.0 - e * e);  // each tile's rounded lower edge
    let dy = abs(f.y - edge_y) * size.y;
    let dx = min(f.x, 1.0 - f.x) * size.x;
    let side = select(9.0, dx, f.y > edge_y);
    return vec2<f32>(0.5 * sp_h1(c + row * 5.0) + 0.5 * clamp(f.y, 0.0, 1.0),
                     1.0 - smoothstep(w, w + aa, min(dy, side)));
}
// The albedo of surface `id` (0 none, 1 grass .. 7 tiles) round the linear colour `base`,
// at pattern coordinate `proj` (world units) whose screen derivative length is `fw`.
fn surface_albedo(base: vec3<f32>, id: i32, proj: vec2<f32>, fw: f32) -> vec3<f32> {
    if (id <= 0 || id > 7) { return base; }
    let k0 = id - 1;
    let prm = material.patterns[k0];  // scale, contrast, tones, ink width
    let dark = material.darks[k0 / 4][k0 % 4];
    let u = proj / prm.x;
    let fwu = fw / prm.x;
    let aa = fwu * 0.7;
    var r = vec2<f32>(0.5, 0.0);
    switch k0 {
        case 0: { r = sp_grass(u, aa, prm.w); }
        case 1: { r = sp_cobbles(u, aa, prm.w); }
        case 2: { r = sp_veins(u, aa, prm.w); }
        case 3: { r = sp_bricks(u, vec2<f32>(0.5, 0.25), aa, prm.w); }
        case 4: { r = sp_bricks(u, vec2<f32>(1.0, 0.55), aa, prm.w); }
        case 5: { r = sp_planks(u, aa, prm.w); }
        default: { r = sp_tiles(u, aa, prm.w); }
    }
    let t = r.x;
    var k = select(-0.5, 0.5, t >= 0.5);
    if (prm.z > 2.5) { k = select(select(1.0, 0.0, t < 0.7), -1.0, t < 0.38); }
    var col = base * (1.0 + k * prm.y * 1.6);
    col = mix(col, mix(col, SP_INK, dark), r.y);
    // Fade to the flat colour once a pattern cell is only a few pixels on screen.
    let fade = smoothstep(4.0, 10.0, 1.0 / max(fwu, 1e-5));
    return mix(base, col, fade);
}
"""

_STANDARDMATERIAL_VCOLOR_SHADER = (
    _STANDARDMATERIAL_SHADER.replace(
        "    @location(1) normal:   vec3<f32>,\n    @builtin(instance_index) instance: u32,",
        "    @location(1) normal:   vec3<f32>,\n"
        "    @location(2) color:    vec4<f32>,   // rgb + surface pattern id\n"
        "    @builtin(instance_index) instance: u32,",
    )
    .replace(
        "    @location(1) world_pos:    vec3<f32>,\n};",
        "    @location(1) world_pos:    vec3<f32>,\n    @location(2) vcolor:       vec4<f32>,\n};",
    )
    .replace(
        "    cut:       vec4<f32>,   // x0, z0, x1, z1: an xz rectangle where nothing draws (empty when x1 <= x0)\n};",
        "    cut:       vec4<f32>,   // x0, z0, x1, z1: an xz rectangle where nothing draws (empty when x1 <= x0)\n"
        "    patterns:  array<vec4<f32>, 7>,  // per surface: scale, contrast, tones, ink width\n"
        "    darks:     array<vec4<f32>, 2>,  // per surface: ink darkness, four to a vec4\n};",
    )
    .replace(
        "@group(0) @binding(3) var<uniform> light_data: LightData;\n",
        "@group(0) @binding(3) var<uniform> light_data: LightData;\n" + _SURFACE_PATTERNS_WGSL,
        1,
    )
    .replace(
        "fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {\n    if (material.mirror",
        "fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {\n"
        "    // Pattern coordinate: world position on the plane the normal faces most, with the\n"
        "    // floating origin added back. Derivatives first, in uniform control flow.\n"
        "    let sp_n = abs(normalize(in.world_normal));\n"
        "    let sp_w = in.world_pos + vec3<f32>(globals.origin_mod.x, 0.0, globals.origin_mod.y);\n"
        "    let sp_proj = select(select(sp_w.xy, sp_w.zy, sp_n.x >= sp_n.z), sp_w.xz,\n"
        "                         sp_n.y >= sp_n.x && sp_n.y >= sp_n.z);\n"
        "    let sp_fw = length(fwidth(sp_proj));\n"
        "    if (material.mirror",
    )
    .replace(
        "    out.world_normal = normalize((model * vec4<f32>(in.normal, 0.0)).xyz);\n"
        "    return out;",
        "    out.world_normal = normalize((model * vec4<f32>(in.normal, 0.0)).xyz);\n"
        "    out.vcolor = in.color;\n"
        "    return out;",
    )
    .replace(
        "    let F0 = mix(vec3<f32>(0.04), material.albedo, material.metallic);",
        "    let vertex_albedo = surface_albedo(in.vcolor.xyz * material.albedo,\n"
        "                                       i32(in.vcolor.w + 0.5), sp_proj, sp_fw);\n"
        "    let F0 = mix(vec3<f32>(0.04), vertex_albedo, material.metallic);",
    )
    .replace(
        "Lo += calculatePointLight(N, V, in.world_pos, F0,\n"
        "                                      material.albedo, material.metallic,\n"
        "                                      material.roughness, light);",
        "Lo += calculatePointLight(N, V, in.world_pos, F0,\n"
        "                                      vertex_albedo, material.metallic,\n"
        "                                      material.roughness, light);",
    )
    .replace(
        "Lo += calculateSun(N, V, F0, material.albedo, material.metallic, material.roughness) * s;",
        "Lo += calculateSun(N, V, F0, vertex_albedo, material.metallic, material.roughness) * s;",
    )
    .replace(
        "Lo += calculateSpot(N, V, in.world_pos, F0, material.albedo,\n"
        "                            material.metallic, material.roughness) * s;",
        "Lo += calculateSpot(N, V, in.world_pos, F0, vertex_albedo,\n"
        "                            material.metallic, material.roughness) * s;",
    )
    .replace(
        "        let diffuse_ibl = kD * irradiance * material.albedo;",
        "        let diffuse_ibl = kD * irradiance * vertex_albedo;",
    )
    .replace(
        "    var ambient = vec3<f32>(0.03) * material.albedo * material.ao;",
        "    var ambient = vec3<f32>(0.03) * vertex_albedo * material.ao;",
    )
)


# Surface patterns (vertex_colors materials): id -> (scale, contrast, tones, ink width,
# ink darkness). Ids: 1 grass, 2 cobbles, 3 veins, 4 brick, 5 blocks, 6 planks, 7 tiles;
# 0 draws the flat vertex colour. Tuned values from a game's art direction; blocks
# and planks share brick's.
SURFACE_PATTERNS = {
    1: (1.0, 0.18, 3, 0.06, 0.55),
    2: (1.6, 0.16, 3, 0.06, 0.60),
    3: (1.0, 0.20, 3, 0.05, 0.60),
    4: (1.0, 0.10, 2, 0.05, 0.45),
    5: (1.0, 0.10, 2, 0.05, 0.45),
    6: (1.0, 0.10, 2, 0.05, 0.45),
    7: (1.0, 0.16, 3, 0.07, 0.60),
}


class StandardMaterial(Material):
    """PBR material with GGX BRDF. Optional 2D albedo map.

    With vertex_colors, a geometry's "surfaces" key (one int per vertex) picks a
    world-space pattern drawn round each vertex colour; `surface_patterns` overrides
    entries of SURFACE_PATTERNS for this material."""

    binding_slot = 1

    def __init__(
        self,
        color,
        roughness=0.5,
        metallic=0.0,
        ao=1.0,
        albedo_map=None,
        vertex_colors=False,
        sway=0.0,
        sway_reach=4.0,
        ripple=0.0,
        mirror=None,
        opacity=1.0,
        cut=None,
        surface_patterns=None,
    ):
        from manifoldx.textures import TextureHandle

        if albedo_map is not None and not isinstance(albedo_map, TextureHandle):
            raise TypeError(
                f"albedo_map expects a TextureHandle from load_texture(...); "
                f"got {type(albedo_map).__name__}. "
                f"Did you forget to call load_texture(engine, path) first?"
            )
        self.color = color
        self.roughness = roughness
        self.metallic = metallic
        self.ao = ao
        self.albedo_map = albedo_map
        self.vertex_colors = vertex_colors
        # Wind: how far a vertex sways per unit of height above its model's origin, up to
        # about sway_reach, where the sway levels off. Only the drawn mesh sways; its shadow stays put.
        self.sway = float(sway)
        self.sway_reach = float(sway_reach)
        # Water: how strongly bands of light drift across the surface; 0 leaves it still.
        self.ripple = float(ripple)
        # A cheap reflection: mirror=y draws the mesh reflected in the plane y (whatever lies
        # below the plane is dropped), and opacity < 1 blends a surface over what is behind
        # it, after every opaque mesh. Mirrored scenery under translucent water reflects.
        # Neither kind casts a shadow.
        self.mirror = None if mirror is None else float(mirror)
        self.opacity = float(opacity)
        # cut=(x0, z0, x1, z1): an xz rectangle, in render space, where nothing of the mesh
        # draws, such as far land under near water. Set it again any frame to move it.
        self.cut = cut
        self.surface_patterns = {**SURFACE_PATTERNS, **(surface_patterns or {})}

    @property
    def pipeline_subtype(self):
        if self.albedo_map is not None:
            return "textured"
        if self.vertex_colors:
            return "vcolor"
        return None

    @property
    def pipeline_state(self):
        """What the pipeline does besides the shader: (translucent, mirrored)."""
        return (self.opacity < 1.0, self.mirror is not None)

    @property
    def casts_shadow(self):
        return self.mirror is None and self.opacity >= 1.0

    def get_texture_bindings(self):
        if self.albedo_map is None:
            return {}
        return {4: self.albedo_map}

    @classmethod
    def _compile(cls, textured: bool = False, vertex_colors: bool = False) -> str:
        if vertex_colors:
            return _STANDARDMATERIAL_VCOLOR_SHADER
        return _STANDARDMATERIAL_TEXTURED_SHADER if textured else _STANDARDMATERIAL_SHADER

    @classmethod
    def uniform_type(cls) -> Dict[str, str]:
        return {
            "albedo": "vec3<f32>",
            "roughness": "f32",
            "metallic": "f32",
            "ao": "f32",
        }

    def get_data(self, n: int, registry) -> np.ndarray:
        """Return material data as numpy array (albedo, roughness, metallic, ao + padding)."""
        albedo = material_rgb(self.color)

        data = np.zeros((n, 16), dtype=np.float32)
        data[:, 0:3] = albedo
        data[:, 3] = self.roughness
        data[:, 4] = self.metallic
        data[:, 5] = self.ao
        data[:, 6] = self.sway
        data[:, 7] = self.sway_reach
        data[:, 8] = self.ripple
        data[:, 9] = 0.0 if self.mirror is None else 1.0
        data[:, 10] = 0.0 if self.mirror is None else self.mirror
        data[:, 11] = self.opacity
        if self.cut is not None:
            data[:, 12:16] = self.cut
        if self.vertex_colors:  # the pattern table the vcolor shader reads
            table = np.zeros((n, 44), dtype=np.float32)
            for sid in range(1, 8):
                scale, contrast, tones, ink, dark = self.surface_patterns[sid]
                table[:, (sid - 1) * 4:(sid - 1) * 4 + 4] = (scale, contrast, tones, ink)
                table[:, 28 + sid - 1] = dark
            data = np.concatenate([data, table], axis=1)
        return data


# =============================================================================
# Geometry Registry
# =============================================================================


def pack_vertices(geometry_obj: dict):
    """A geometry's vertices as the interleaved bytes its vertex buffer holds,
    and the layout: (data, info), with data None when it has no positions.
    Pure numpy, so it can run off the frame (a Streamer worker)."""
    buffers = {}
    data = None

    if "positions" in geometry_obj:
        positions = geometry_obj["positions"].astype(np.float32)
    elif "vertices" in geometry_obj:
        positions = geometry_obj["vertices"].astype(np.float32)
    else:
        positions = None

    if positions is not None:
        has_normals = "normals" in geometry_obj
        has_uvs = "uvs" in geometry_obj
        has_colors = "colors" in geometry_obj

        if has_normals and has_uvs:
            normals = geometry_obj["normals"].astype(np.float32)
            uvs = geometry_obj["uvs"].astype(np.float32)
            n_verts = len(positions)
            interleaved = np.zeros((n_verts, 8), dtype=np.float32)
            interleaved[:, 0:3] = positions
            interleaved[:, 3:6] = normals
            interleaved[:, 6:8] = uvs
            data = interleaved.tobytes()
            buffers["stride"] = 8 * 4  # pos(3) + normal(3) + uv(2)
            buffers["has_normals"] = True
            buffers["has_uvs"] = True
            buffers["has_colors"] = False
        elif has_normals and has_colors:
            normals = geometry_obj["normals"].astype(np.float32)
            colors = geometry_obj["colors"].astype(np.float32)
            n_verts = len(positions)
            interleaved = np.zeros((n_verts, 10), dtype=np.float32)
            interleaved[:, 0:3] = positions
            interleaved[:, 3:6] = normals
            interleaved[:, 6:9] = colors
            if "surfaces" in geometry_obj:  # the surface pattern id, 0 for none
                interleaved[:, 9] = np.asarray(geometry_obj["surfaces"], dtype=np.float32)
            data = interleaved.tobytes()
            buffers["stride"] = 10 * 4  # pos(3) + normal(3) + color(3) + surface id(1)
            buffers["has_normals"] = True
            buffers["has_uvs"] = False
            buffers["has_colors"] = True
        elif has_normals:
            normals = geometry_obj["normals"].astype(np.float32)
            n_verts = len(positions)
            interleaved = np.zeros((n_verts, 6), dtype=np.float32)
            interleaved[:, 0:3] = positions
            interleaved[:, 3:6] = normals
            data = interleaved.tobytes()
            buffers["stride"] = 6 * 4
            buffers["has_normals"] = True
            buffers["has_uvs"] = False
            buffers["has_colors"] = False
        else:
            data = positions.tobytes()
            buffers["stride"] = 3 * 4
            buffers["has_normals"] = False
            buffers["has_uvs"] = False
            buffers["has_colors"] = False

        # Skinned geometry: one uint32 joint index after the floats above,
        # so every other attribute keeps its offset.
        buffers["has_joints"] = "joints" in geometry_obj
        if buffers["has_joints"]:
            from manifoldx.skinning import append_joint_column, validate_joints

            joints = validate_joints(geometry_obj["joints"], len(positions))
            rows = np.frombuffer(data, dtype=np.float32).reshape(len(positions), -1)
            data = append_joint_column(rows, joints).tobytes()
            buffers["joints_offset"] = buffers["stride"]
            buffers["stride"] += 4

        buffers["vertex_count"] = len(positions)
    return data, buffers


class GeometryRegistry:
    """
    Cache of GPU geometry resources.

    Lazily creates WebGPU buffers on first use.
    """

    def __init__(self, device=None):
        from manifoldx.viz.geometry import (
            AXIS_LINE_X as _axis_line_x,
            AXIS_LINE_Y as _axis_line_y,
            AXIS_LINE_Z as _axis_line_z,
            SPRITE_QUAD as _sprite_quad,
        )

        self._device = device
        self._geometries: Dict[int, Any] = {}  # id -> geometry dict
        self._object_to_id: Dict[int, int] = {}  # object id -> registry id
        self._name_to_id: Dict[str, int] = {}  # name -> registry id
        self._next_id = 1
        self._gpu_buffers: Dict[int, dict] = {}  # id -> {vertex_buffer, index_buffer}

        # Register built-in geometries by name
        self._register_builtin(_sprite_quad["name"], _sprite_quad)
        self._register_builtin(_axis_line_x["name"], _axis_line_x)
        self._register_builtin(_axis_line_y["name"], _axis_line_y)
        self._register_builtin(_axis_line_z["name"], _axis_line_z)

    def create_buffers(self, geometry_id: int, geometry_obj: dict, queue):
        """Create GPU buffers for geometry.

        Creates interleaved vertex buffer: [pos.x, pos.y, pos.z, norm.x, norm.y, norm.z] per vertex.
        Falls back to position-only if no normals.
        """
        if self._device is None or queue is None:
            return None

        data, buffers = pack_vertices(geometry_obj)
        if data is not None:
            vertex_buffer = self._device.create_buffer(
                size=len(data),
                usage=wgpu.BufferUsage.VERTEX | wgpu.BufferUsage.COPY_DST,
            )
            queue.write_buffer(vertex_buffer, 0, data)
            buffers["vertex_buffer"] = vertex_buffer

        # Create index buffer
        if "indices" in geometry_obj:
            indices = geometry_obj["indices"]
            data = indices.astype(np.uint32).tobytes()

            index_buffer = self._device.create_buffer(
                size=len(data),
                usage=wgpu.BufferUsage.INDEX | wgpu.BufferUsage.COPY_DST,
            )
            queue.write_buffer(index_buffer, 0, data)
            buffers["index_buffer"] = index_buffer
            buffers["index_count"] = int(np.asarray(indices).size)  # (M, 3) triangles too

        self._gpu_buffers[geometry_id] = buffers
        return buffers

    def adopt_buffers(self, geometry_id: int, buffers: dict) -> None:
        """Take GPU buffers built elsewhere for a registered geometry: the dict
        create_buffers would have made (pack_vertices' info plus vertex_buffer,
        and index_buffer and index_count when indexed)."""
        self._gpu_buffers[geometry_id] = buffers

    def get_gpu_buffers(self, geometry_id: int) -> dict:
        """Get GPU buffers for geometry."""
        return self._gpu_buffers.get(geometry_id)

    def release(self, geometry_id: int) -> None:
        """Forget a geometry and destroy its GPU buffers. Named built-ins stay."""
        if geometry_id in self._name_to_id.values():
            return
        for key in ("vertex_buffer", "index_buffer"):
            buf = self._gpu_buffers.get(geometry_id, {}).get(key)
            if buf is not None:
                buf.destroy()
        self._gpu_buffers.pop(geometry_id, None)
        geo = self._geometries.pop(geometry_id, None)
        if geo is not None:
            self._object_to_id.pop(id(geo), None)

    def register(self, geometry_obj) -> int:
        """Register a geometry, return ID. Same object returns same ID."""
        obj_id = id(geometry_obj)
        if obj_id in self._object_to_id:
            return self._object_to_id[obj_id]

        new_id = self._next_id
        self._next_id += 1
        self._object_to_id[obj_id] = new_id
        self._geometries[new_id] = geometry_obj
        return new_id

    def _register_builtin(self, name: str, geometry_obj: dict) -> int:
        """Register a named built-in geometry. Returns the assigned ID."""
        gid = self.register(geometry_obj)
        self._name_to_id[name] = gid
        return gid

    def register_by_name(self, name: str, geometry_obj: dict) -> int:
        """Register a geometry by name. Returns the assigned ID."""
        return self._register_builtin(name, geometry_obj)

    def get_id(self, name: str) -> int:
        """Get the numeric ID for a named geometry."""
        return self._name_to_id[name]

    def get_by_name(self, name: str) -> Any:
        """Get a geometry dict by name."""
        gid = self._name_to_id[name]
        return self._geometries[gid]

    def __contains__(self, name: str) -> bool:
        """Check if a named geometry is registered."""
        return name in self._name_to_id

    def get(self, geometry_id: int) -> Any:
        """Get geometry by ID."""
        return self._geometries.get(geometry_id)


# =============================================================================
# Material Registry
# =============================================================================


class MaterialRegistry:
    """Cache of GPU material pipelines."""

    def __init__(self, device=None):
        self._device = device
        self._materials: Dict[int, Any] = {}
        self._object_to_id: Dict[int, int] = {}
        self._next_id = 1

    def register(self, material_obj) -> int:
        """Register material, return ID. Same object returns same ID."""
        obj_id = id(material_obj)
        if obj_id in self._object_to_id:
            return self._object_to_id[obj_id]

        new_id = self._next_id
        self._next_id += 1
        self._object_to_id[obj_id] = new_id
        self._materials[new_id] = material_obj
        return new_id

    def get(self, material_id: int) -> Any:
        """Get material by ID."""
        return self._materials.get(material_id)


# =============================================================================
# Geometry Factories
# =============================================================================


def cube(width: float, height: float, depth: float) -> dict:
    """
    Create cube geometry with per-face vertices and normals for flat shading.

    Returns dict with 'positions', 'normals', and 'indices' arrays.
    24 vertices (4 per face), 36 indices.
    """
    w, h, d = width / 2, height / 2, depth / 2

    # 6 faces × 4 vertices each = 24 vertices (unshared for flat normals)
    positions = np.array(
        [
            # Front face (z+)
            [-w, -h, d],
            [w, -h, d],
            [w, h, d],
            [-w, h, d],
            # Back face (z-)
            [w, -h, -d],
            [-w, -h, -d],
            [-w, h, -d],
            [w, h, -d],
            # Right face (x+)
            [w, -h, d],
            [w, -h, -d],
            [w, h, -d],
            [w, h, d],
            # Left face (x-)
            [-w, -h, -d],
            [-w, -h, d],
            [-w, h, d],
            [-w, h, -d],
            # Top face (y+)
            [-w, h, d],
            [w, h, d],
            [w, h, -d],
            [-w, h, -d],
            # Bottom face (y-)
            [-w, -h, -d],
            [w, -h, -d],
            [w, -h, d],
            [-w, -h, d],
        ],
        dtype=np.float32,
    )

    normals = np.array(
        [
            # Front
            [0, 0, 1],
            [0, 0, 1],
            [0, 0, 1],
            [0, 0, 1],
            # Back
            [0, 0, -1],
            [0, 0, -1],
            [0, 0, -1],
            [0, 0, -1],
            # Right
            [1, 0, 0],
            [1, 0, 0],
            [1, 0, 0],
            [1, 0, 0],
            # Left
            [-1, 0, 0],
            [-1, 0, 0],
            [-1, 0, 0],
            [-1, 0, 0],
            # Top
            [0, 1, 0],
            [0, 1, 0],
            [0, 1, 0],
            [0, 1, 0],
            # Bottom
            [0, -1, 0],
            [0, -1, 0],
            [0, -1, 0],
            [0, -1, 0],
        ],
        dtype=np.float32,
    )

    # 6 faces × 2 triangles × 3 indices = 36
    indices = np.array(
        [
            0,
            1,
            2,
            0,
            2,
            3,  # front
            4,
            5,
            6,
            4,
            6,
            7,  # back
            8,
            9,
            10,
            8,
            10,
            11,  # right
            12,
            13,
            14,
            12,
            14,
            15,  # left
            16,
            17,
            18,
            16,
            18,
            19,  # top
            20,
            21,
            22,
            20,
            22,
            23,  # bottom
        ],
        dtype=np.uint32,
    )

    return {"positions": positions, "normals": normals, "indices": indices}


def sphere(radius: float, segments: int = 32) -> dict:
    """
    Create UV sphere geometry with normals and UVs.

    Normals point outward (normalized position for unit sphere).
    UVs use standard (u = lon/lon_lines, v = lat/lat_lines) spherical mapping.
    Winding order is CCW when viewed from outside.
    """
    lat_lines = segments
    lon_lines = segments * 2

    positions = []
    normals = []
    uvs = []
    for lat in range(lat_lines + 1):
        theta = lat * np.pi / lat_lines
        sin_theta = np.sin(theta)
        cos_theta = np.cos(theta)
        v = lat / lat_lines

        for lon in range(lon_lines + 1):
            phi = lon * 2 * np.pi / lon_lines
            nx = sin_theta * np.cos(phi)
            ny = cos_theta
            nz = sin_theta * np.sin(phi)
            normals.append([nx, ny, nz])
            positions.append([nx * radius, ny * radius, nz * radius])
            uvs.append([lon / lon_lines, v])

    positions = np.array(positions, dtype=np.float32)
    normals = np.array(normals, dtype=np.float32)
    uvs = np.array(uvs, dtype=np.float32)

    indices = []
    for lat in range(lat_lines):
        for lon in range(lon_lines):
            first = lat * (lon_lines + 1) + lon
            second = first + lon_lines + 1

            indices.extend([first, first + 1, second])
            indices.extend([second, first + 1, second + 1])

    indices = np.array(indices, dtype=np.uint32)

    return {"positions": positions, "normals": normals, "uvs": uvs, "indices": indices}


def plane(width: float, height: float) -> dict:
    """Create plane geometry with normals facing +Z and UVs in [0,1]²."""
    w, h = width / 2, height / 2

    positions = np.array(
        [
            [-w, -h, 0],
            [w, -h, 0],
            [w, h, 0],
            [-w, h, 0],
        ],
        dtype=np.float32,
    )

    normals = np.array(
        [[0, 0, 1], [0, 0, 1], [0, 0, 1], [0, 0, 1]],
        dtype=np.float32,
    )

    uvs = np.array(
        [[0, 0], [1, 0], [1, 1], [0, 1]],
        dtype=np.float32,
    )

    indices = np.array([0, 1, 2, 0, 2, 3], dtype=np.uint32)

    return {"positions": positions, "normals": normals, "uvs": uvs, "indices": indices}


# =============================================================================
# Material Factories
# =============================================================================


def basic(color):
    """Create unlit material."""
    return BasicMaterial(color)


class PhongMaterial(BasicMaterial):
    """Phong shading material (uses BasicMaterial shader for now)."""

    def __init__(self, color, shininess: float = 32.0):
        super().__init__(color)
        self.shininess = shininess


def phong(color, shininess: float = 32.0):
    """Create Phong shading material."""
    return PhongMaterial(color, shininess)


def standard(color, roughness: float = 0.5, metallic: float = 0.0) -> StandardMaterial:
    """Create PBR material."""
    return StandardMaterial(color, roughness, metallic)


# =============================================================================
# Light Classes
# =============================================================================


class PointLight:
    """Point light for GPU."""

    def __init__(self, color, intensity, position, distance=0, decay=2):
        self.color = color
        self.intensity = intensity
        self.position = position
        self.distance = distance
        self.decay = decay

    @classmethod
    def uniform_type(cls) -> dict:
        return {
            "position": "vec3<f32>",
            "padding0": "f32",
            "color": "vec3<f32>",
            "intensity": "f32",
        }

    def get_data(self) -> np.ndarray:
        """Return light data as numpy array for GPU upload."""
        color_hex = self.color.lstrip("#")
        r = int(color_hex[0:2], 16) / 255.0
        g = int(color_hex[2:4], 16) / 255.0
        b = int(color_hex[4:6], 16) / 255.0

        data = np.zeros(8, dtype=np.float32)
        data[0:3] = self.position
        data[4:7] = [r, g, b]
        data[7] = self.intensity
        return data


class SpotLight:
    """Spot light for GPU."""

    def __init__(
        self,
        color,
        intensity,
        position,
        direction,
        inner_angle,
        outer_angle,
        distance=0,
        decay=2,
    ):
        self.color = color
        self.intensity = intensity
        self.position = position
        self.direction = direction
        self.inner_angle = inner_angle
        self.outer_angle = outer_angle
        self.distance = distance
        self.decay = decay

    @classmethod
    def uniform_type(cls) -> dict:
        return {
            "position": "vec3<f32>",
            "padding0": "f32",
            "color": "vec3<f32>",
            "intensity": "f32",
            "direction": "vec3<f32>",
            "inner_angle": "f32",
        }

    def get_data(self) -> np.ndarray:
        """Return light data as numpy array for GPU upload."""
        color_hex = self.color.lstrip("#")
        r = int(color_hex[0:2], 16) / 255.0
        g = int(color_hex[2:4], 16) / 255.0
        b = int(color_hex[4:6], 16) / 255.0

        data = np.zeros(12, dtype=np.float32)
        data[0:3] = self.position
        data[4:7] = [r, g, b]
        data[7] = self.intensity
        data[8:11] = self.direction
        data[11] = self.outer_angle
        return data


class DirectionalLight:
    """Directional light for GPU."""

    def __init__(self, color, intensity, direction):
        self.color = color
        self.intensity = intensity
        self.direction = direction

    @classmethod
    def uniform_type(cls) -> dict:
        return {
            "direction": "vec3<f32>",
            "padding0": "f32",
            "color": "vec3<f32>",
            "intensity": "f32",
        }

    def get_data(self) -> np.ndarray:
        """Return light data as numpy array for GPU upload."""
        color_hex = self.color.lstrip("#")
        r = int(color_hex[0:2], 16) / 255.0
        g = int(color_hex[2:4], 16) / 255.0
        b = int(color_hex[4:6], 16) / 255.0

        data = np.zeros(8, dtype=np.float32)
        data[0:3] = self.direction
        data[4:7] = [r, g, b]
        data[7] = self.intensity
        return data


class _VolumeResource:
    """Internal: one registered volume's CPU + (eventual) GPU state."""

    __slots__ = ("data", "name", "dirty", "texture")

    def __init__(self, data: np.ndarray, name: str):
        self.data = data
        self.name = name
        self.dirty = True  # set on register/update; cleared post-upload
        self.texture = None  # GPU texture; created lazily on first upload


class VolumeRegistry:
    """Cache of GPU 3D scalar field resources.

    Mirrors the GeometryRegistry / MaterialRegistry shape:
    - `register(numpy_array, name=...) -> int handle`
    - `update(handle, numpy_array)` — same shape only; flips dirty bit.
    - `get(handle) -> _VolumeResource`
    - GPU texture creation is lazy (handled by the renderer at the
      first frame in which the volume is needed).
    """

    def __init__(self, device=None):
        self._device = device
        self._volumes: Dict[int, _VolumeResource] = {}
        self._next_id = 1

    def register(self, data: np.ndarray, *, name: str | None = None) -> int:
        if data.ndim != 3:
            raise ValueError(f"volume data must be 3D (Nz, Ny, Nx); got {data.ndim}D")
        if data.dtype != np.float32:
            raise ValueError(
                f"volume data must be float32; got {data.dtype}. "
                f"Convert with `array.astype(np.float32)`."
            )
        if not data.flags["C_CONTIGUOUS"]:
            raise ValueError(
                "volume data must be C-contiguous; "
                "call `np.ascontiguousarray(array)` before registering."
            )
        handle = self._next_id
        self._next_id += 1
        self._volumes[handle] = _VolumeResource(data=data, name=name or f"volume_{handle}")
        return handle

    def update(self, handle: int, data: np.ndarray) -> None:
        res = self.get(handle)
        if data.shape != res.data.shape:
            raise ValueError(
                f"update_volume: shape mismatch — registered {res.data.shape}, "
                f"got {data.shape}. Re-register if you need a different size."
            )
        if data.dtype != np.float32:
            raise ValueError(f"volume data must be float32; got {data.dtype}.")
        if not data.flags["C_CONTIGUOUS"]:
            raise ValueError("volume data must be C-contiguous.")
        res.data = data
        res.dirty = True

    def get(self, handle: int) -> _VolumeResource:
        if handle not in self._volumes:
            raise KeyError(f"unknown volume handle: {handle}")
        return self._volumes[handle]

    def upload_to_gpu(self, handle: int, queue) -> None:
        """Lazily create a `texture_3d` r32float for this volume and write
        the current numpy data into it. Clears the dirty bit. Re-creates the
        texture only if shape changed (which `update` already disallows).
        """
        if self._device is None or queue is None:
            return
        res = self.get(handle)
        if not res.dirty and res.texture is not None:
            return

        nz, ny, nx = res.data.shape
        if res.texture is None:
            res.texture = self._device.create_texture(
                size=(nx, ny, nz),
                format=wgpu.TextureFormat.r32float,
                usage=wgpu.TextureUsage.TEXTURE_BINDING | wgpu.TextureUsage.COPY_DST,
                dimension=wgpu.TextureDimension.d3,
                mip_level_count=1,
                sample_count=1,
            )

        data_bytes = res.data.tobytes()
        bytes_per_row = nx * 4  # 4 bytes per f32 voxel
        rows_per_image = ny
        queue.write_texture(
            {"texture": res.texture, "mip_level": 0, "origin": (0, 0, 0)},
            data_bytes,
            {
                "offset": 0,
                "bytes_per_row": bytes_per_row,
                "rows_per_image": rows_per_image,
            },
            (nx, ny, nz),
        )
        res.dirty = False


__all__ = [
    "Material",
    "BasicMaterial",
    "StandardMaterial",
    "GeometryRegistry",
    "MaterialRegistry",
    "VolumeRegistry",
    "cube",
    "sphere",
    "plane",
    "basic",
    "phong",
    "standard",
    "PointLight",
    "SpotLight",
    "DirectionalLight",
]
