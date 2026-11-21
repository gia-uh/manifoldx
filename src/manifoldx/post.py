"""Screen-space effects after the main pass: ambient occlusion, bloom, light shafts, a colour grade and FXAA.

`engine.set_post(...)` turns them on. With any of them on, the main pass draws into an offscreen colour
target of the swapchain's format, these passes read it and the scene's depth, and the last one writes the
swapchain; the GUI is drawn after them, untouched. With all of them off the frame is drawn as before.

Lit shaders keep their tonemap, so the colour these passes see is already in [0, 1]: bloom picks what is
brighter than a threshold of that range, and light shafts blur the sky's pixels outward from the sun.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np


@dataclass(frozen=True)
class SSAO:
    """Ambient occlusion from the depth buffer, at half resolution: corners and the foot of walls darken.

    radius is in world metres; intensity 1 applies the occlusion fully; power deepens it (the occlusion raised to
    it); samples is up to 32."""

    radius: float = 1.0
    intensity: float = 1.0
    samples: int = 16
    bias: float = 0.03
    power: float = 2.0


@dataclass(frozen=True)
class Bloom:
    """A glow around what is brighter than threshold (0 to 1 of the tonemapped colour), blurred over four
    halvings of the frame and added back at intensity."""

    threshold: float = 0.85
    knee: float = 0.1
    intensity: float = 0.5


@dataclass(frozen=True)
class LightShafts:
    """Rays from the sun: the sky's pixels blurred along lines from the sun's place on screen, at quarter
    resolution. They fade as the sun leaves the frame or sets."""

    intensity: float = 0.5
    decay: float = 0.96
    density: float = 0.9
    samples: int = 48


@dataclass(frozen=True)
class Grade:
    """A colour grade on the final image: saturation and contrast around 1, a vignette from 0."""

    saturation: float = 1.0
    contrast: float = 1.0
    vignette: float = 0.0


@dataclass(frozen=True)
class Sky:
    """The sky behind everything drawn, in place of the flat background: a gradient from horizon to zenith
    (linear colours; horizon None takes the engine's background) and the sun's disk and halo far brighter than
    anything lit, so bloom and light shafts start from it. sun_size is the disk's angular radius in degrees;
    sun_intensity is how many times brighter than white the disk is."""

    zenith: tuple = (0.18, 0.36, 0.78)
    horizon: tuple | None = None
    sun_size: float = 0.6
    sun_intensity: float = 40.0
    glow: float = 1.0


@dataclass(frozen=True)
class Post:
    """What runs after the main pass; None or False leaves an effect out."""

    ssao: SSAO | None = None
    bloom: Bloom | None = None
    shafts: LightShafts | None = None
    grade: Grade | None = None
    sky: Sky | None = None
    fxaa: bool = False
    _kernel_seed: int = field(default=7, repr=False)

    def active(self) -> bool:
        return bool(self.ssao or self.bloom or self.shafts or self.grade or self.sky or self.fxaa)

    def changed(self, **changes) -> "Post":
        """A copy with these fields set; True for an effect means its defaults."""
        defaults = {"ssao": SSAO, "bloom": Bloom, "shafts": LightShafts, "grade": Grade, "sky": Sky}
        for k, v in changes.items():
            if k not in defaults and k != "fxaa":
                raise ValueError(f"unknown post effect {k!r}; known: ssao, bloom, shafts, grade, sky, fxaa")
            if v is True and k in defaults:
                changes[k] = defaults[k]()
            elif v is False and k in defaults:
                changes[k] = None
        return replace(self, **changes)


def ssao_kernel(n: int, seed: int = 7) -> np.ndarray:
    """n points in the unit hemisphere around +z, denser near the centre, as (32, 4) float32 (rows past n are 0)."""
    rng = np.random.default_rng(seed)
    out = np.zeros((32, 4), np.float32)
    for i in range(n):
        v = rng.uniform((-1, -1, 0.15), (1, 1, 1))
        v = v / np.linalg.norm(v) * rng.uniform(0.3, 1.0)
        s = i / n
        out[i, :3] = v * (0.3 + 0.7 * s * s)
    return out


def sun_on_screen(camera, aspect: float, sun) -> tuple[float, float, float]:
    """(u, v, visible): where the sun is in the frame (0..1, origin top-left) and how much its shafts show,
    1 with the sun in front and inside the frame, fading to 0 as it leaves it, goes behind or sets."""
    if sun is None:
        return 0.5, 0.5, 0.0
    to_sun = -np.asarray(sun.direction, float)
    to_sun = to_sun / (np.linalg.norm(to_sun) or 1.0)
    if to_sun[1] <= -0.02:
        return 0.5, 0.5, 0.0
    view = camera.get_view_matrix()
    proj = camera.get_projection_matrix(aspect, near=camera.near, far=camera.far)
    d = view[:3, :3] @ to_sun  # the sun is at infinity: only the view's rotation applies
    if d[2] >= -1e-3:
        return 0.5, 0.5, 0.0
    c = proj @ np.array([*d, 0.0])
    x, y = c[0] / c[3], c[1] / c[3]
    edge = max(abs(x), abs(y))
    visible = float(np.clip((1.6 - edge) / 0.6, 0.0, 1.0)) * float(np.clip(to_sun[1] / 0.08 + 0.25, 0.0, 1.0))
    return float(x * 0.5 + 0.5), float(0.5 - y * 0.5), visible


_VS = """
struct VOut { @builtin(position) pos: vec4<f32>, @location(0) uv: vec2<f32> };

@vertex
fn vs(@builtin(vertex_index) i: u32) -> VOut {
    let p = vec2<f32>(f32((i << 1u) & 2u), f32(i & 2u));
    var o: VOut;
    o.pos = vec4<f32>(p * 2.0 - 1.0, 0.0, 1.0);
    o.uv = vec2<f32>(p.x, 1.0 - p.y);
    return o;
}
"""

_UNIFORMS = """
struct Post {
    proj: mat4x4<f32>,
    inv_proj: mat4x4<f32>,
    size: vec2<f32>,
    sun_uv: vec2<f32>,
    ssao: vec4<f32>,       // radius, intensity, samples, bias
    bloom: vec4<f32>,      // threshold, knee, intensity, -
    shafts: vec4<f32>,     // intensity, decay, density, visible
    grade: vec4<f32>,      // saturation, contrast, vignette, the occlusion's power
    flags: vec4<u32>,      // ssao, bloom, shafts, samples of the shafts
    sun_color: vec4<f32>,  // the sun's linear colour times its intensity, clamped to 1
    inv_view: mat4x4<f32>, // camera to world
    sun_dir: vec4<f32>,    // toward the sun, world; w 1 when there is a sun
    zenith: vec4<f32>,
    horizon: vec4<f32>,
    sky: vec4<f32>,        // cos of the disk's radius, its intensity, the halo's strength, 1 when Sky is on
    kernel: array<vec4<f32>, 32>,
};
@group(0) @binding(0) var<uniform> u: Post;

fn ray(uv: vec2<f32>) -> vec3<f32> {
    let v = u.inv_proj * vec4<f32>(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0, 0.5, 1.0);
    return normalize((u.inv_view * vec4<f32>(normalize(v.xyz / v.w), 0.0)).xyz);
}

// The sun's disk and halo along a ray, unbounded: what makes the sky glare.
fn sun_light(d: vec3<f32>) -> vec3<f32> {
    if u.sun_dir.w <= 0.0 {
        return vec3<f32>(0.0);
    }
    let mu = dot(d, u.sun_dir.xyz);
    let edge = 1.0 - u.sky.x;
    let disk = smoothstep(u.sky.x - edge * 0.3, u.sky.x + edge * 0.3, mu) * u.sky.y;
    let m = max(mu, 0.0);
    let halo = (0.8 * pow(m, 900.0) + 0.35 * pow(m, 60.0) + 0.12 * pow(m, 6.0)) * u.sky.z;
    let low = clamp(u.sun_dir.y * 8.0 + 1.0, 0.0, 1.0);  // set below the horizon
    return u.sun_color.rgb * (disk + halo * 2.0) * low * select(1.0, 0.0, d.y < -0.01);
}

// The sky's gradient along a ray, in [0, 1].
fn sky_base(d: vec3<f32>) -> vec3<f32> {
    let t = clamp(d.y, 0.0, 1.0);
    return mix(u.horizon.rgb, u.zenith.rgb, pow(t, 0.55));
}
"""

UNIFORM_BYTES = 64 * 2 + 16 + 16 * 6 + 64 + 16 * 4 + 16 * 32

_SSAO = _VS + _UNIFORMS + """
@group(0) @binding(1) var depth_tex: texture_depth_2d;

fn view_at(px: vec2<i32>) -> vec3<f32> {
    let q = clamp(px, vec2<i32>(0), vec2<i32>(u.size) - vec2<i32>(1));
    let d = textureLoad(depth_tex, q, 0);
    let uv = (vec2<f32>(q) + 0.5) / u.size;
    let v = u.inv_proj * vec4<f32>(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0, d, 1.0);
    return v.xyz / v.w;
}

fn hash(p: vec2<f32>) -> f32 {
    return fract(sin(dot(p, vec2<f32>(12.9898, 78.233))) * 43758.5453);
}

@fragment
fn fs(in: VOut) -> @location(0) vec4<f32> {
    let px = vec2<i32>(floor(in.pos.xy) * 2.0);
    let q = clamp(px, vec2<i32>(0), vec2<i32>(u.size) - vec2<i32>(1));
    if textureLoad(depth_tex, q, 0) <= 0.0 {
        return vec4<f32>(1.0);  // the sky
    }
    let p = view_at(px);
    let r = view_at(px + vec2<i32>(1, 0));
    let l = view_at(px - vec2<i32>(1, 0));
    let d = view_at(px + vec2<i32>(0, 1));
    let t = view_at(px - vec2<i32>(0, 1));
    let dx = select(p - l, r - p, abs(r.z - p.z) < abs(p.z - l.z));
    let dy = select(p - t, d - p, abs(d.z - p.z) < abs(p.z - t.z));
    var n = normalize(cross(dx, dy));
    if dot(n, -p) < 0.0 {
        n = -n;
    }
    let a = hash(floor(in.pos.xy)) * 6.2831853;
    let rv = vec3<f32>(cos(a), sin(a), 0.0);
    let tg = normalize(rv - n * dot(rv, n) + vec3<f32>(1e-4, 0.0, 0.0));
    let bt = cross(n, tg);
    let radius = u.ssao.x;
    let count = i32(u.ssao.z);
    var occ = 0.0;
    for (var i = 0; i < count; i = i + 1) {
        let k = u.kernel[i].xyz;
        let s = p + (tg * k.x + bt * k.y + n * k.z) * radius;
        let c = u.proj * vec4<f32>(s, 1.0);
        let suv = vec2<f32>(c.x / c.w * 0.5 + 0.5, 0.5 - c.y / c.w * 0.5);
        if any(suv < vec2<f32>(0.0)) || any(suv > vec2<f32>(1.0)) {
            continue;
        }
        let sz = view_at(vec2<i32>(suv * u.size)).z;
        let range = smoothstep(0.0, 1.0, radius / max(abs(p.z - sz), 1e-4));
        occ = occ + select(0.0, 1.0, sz >= s.z + u.ssao.w) * range;
    }
    let ao = pow(1.0 - occ / f32(max(count, 1)), max(u.grade.w, 0.01));
    return vec4<f32>(ao, ao, ao, 1.0);
}
"""

_SSAO_BLUR = _VS + _UNIFORMS + """
@group(0) @binding(1) var depth_tex: texture_depth_2d;
@group(0) @binding(2) var ao_tex: texture_2d<f32>;

fn vz(px: vec2<i32>) -> f32 {
    let q = clamp(px, vec2<i32>(0), vec2<i32>(u.size) - vec2<i32>(1));
    let d = textureLoad(depth_tex, q, 0);
    let v = u.inv_proj * vec4<f32>(0.0, 0.0, d, 1.0);
    return v.z / v.w;
}

@fragment
fn fs(in: VOut) -> @location(0) vec4<f32> {
    let h = vec2<i32>(floor(in.pos.xy));
    let hs = vec2<i32>(textureDimensions(ao_tex));
    let z0 = vz(h * 2);
    var sum = 0.0;
    var w = 0.0;
    for (var y = -2; y < 2; y = y + 1) {
        for (var x = -2; x < 2; x = x + 1) {
            let q = clamp(h + vec2<i32>(x, y), vec2<i32>(0), hs - vec2<i32>(1));
            let wz = exp(-abs(vz(q * 2) - z0) * 4.0 / max(abs(z0) * 0.02, 0.05));
            sum = sum + textureLoad(ao_tex, q, 0).r * wz;
            w = w + wz;
        }
    }
    let ao = sum / max(w, 1e-4);
    return vec4<f32>(ao, ao, ao, 1.0);
}
"""

_BRIGHT = _VS + _UNIFORMS + """
@group(0) @binding(1) var src: texture_2d<f32>;
@group(0) @binding(2) var lin: sampler;
@group(0) @binding(3) var depth_tex: texture_depth_2d;

fn tap(uv: vec2<f32>) -> vec3<f32> {
    let px = clamp(vec2<i32>(uv * u.size), vec2<i32>(0), vec2<i32>(u.size) - vec2<i32>(1));
    var c = textureSampleLevel(src, lin, uv, 0.0).rgb;
    if textureLoad(depth_tex, px, 0) <= 0.0 {
        let d = ray(uv);
        c = select(c, sky_base(d), u.sky.w > 0.0) + sun_light(d);
    }
    return min(c, vec3<f32>(64.0));
}

@fragment
fn fs(in: VOut) -> @location(0) vec4<f32> {
    let px = 1.0 / vec2<f32>(textureDimensions(src));
    var c = vec3<f32>(0.0);
    c = c + tap(in.uv + vec2<f32>(-0.5, -0.5) * px);
    c = c + tap(in.uv + vec2<f32>(0.5, -0.5) * px);
    c = c + tap(in.uv + vec2<f32>(-0.5, 0.5) * px);
    c = c + tap(in.uv + vec2<f32>(0.5, 0.5) * px);
    c = c * 0.25;
    let br = dot(c, vec3<f32>(0.2126, 0.7152, 0.0722));
    let t = u.bloom.x;
    let k = max(u.bloom.y, 1e-4);
    var soft = clamp(br - t + k, 0.0, 2.0 * k);
    soft = soft * soft / (4.0 * k);
    let wgt = max(soft, br - t) / max(br, 1e-4);
    return vec4<f32>(c * wgt, 1.0);
}
"""

_DOWN = _VS + """
@group(0) @binding(1) var src: texture_2d<f32>;
@group(0) @binding(2) var lin: sampler;

@fragment
fn fs(in: VOut) -> @location(0) vec4<f32> {
    let px = 1.0 / vec2<f32>(textureDimensions(src));
    var c = textureSampleLevel(src, lin, in.uv, 0.0).rgb * 0.5;
    c = c + textureSampleLevel(src, lin, in.uv + vec2<f32>(-1.0, -1.0) * px, 0.0).rgb * 0.125;
    c = c + textureSampleLevel(src, lin, in.uv + vec2<f32>(1.0, -1.0) * px, 0.0).rgb * 0.125;
    c = c + textureSampleLevel(src, lin, in.uv + vec2<f32>(-1.0, 1.0) * px, 0.0).rgb * 0.125;
    c = c + textureSampleLevel(src, lin, in.uv + vec2<f32>(1.0, 1.0) * px, 0.0).rgb * 0.125;
    return vec4<f32>(c, 1.0);
}
"""

_UP = _VS + """
@group(0) @binding(1) var low: texture_2d<f32>;
@group(0) @binding(2) var lin: sampler;
@group(0) @binding(3) var cur: texture_2d<f32>;

@fragment
fn fs(in: VOut) -> @location(0) vec4<f32> {
    let px = 1.0 / vec2<f32>(textureDimensions(low));
    var c = textureSampleLevel(low, lin, in.uv, 0.0).rgb * 4.0;
    c = c + textureSampleLevel(low, lin, in.uv + vec2<f32>(-1.0, 0.0) * px, 0.0).rgb * 2.0;
    c = c + textureSampleLevel(low, lin, in.uv + vec2<f32>(1.0, 0.0) * px, 0.0).rgb * 2.0;
    c = c + textureSampleLevel(low, lin, in.uv + vec2<f32>(0.0, -1.0) * px, 0.0).rgb * 2.0;
    c = c + textureSampleLevel(low, lin, in.uv + vec2<f32>(0.0, 1.0) * px, 0.0).rgb * 2.0;
    c = c + textureSampleLevel(low, lin, in.uv + vec2<f32>(-1.0, -1.0) * px, 0.0).rgb;
    c = c + textureSampleLevel(low, lin, in.uv + vec2<f32>(1.0, -1.0) * px, 0.0).rgb;
    c = c + textureSampleLevel(low, lin, in.uv + vec2<f32>(-1.0, 1.0) * px, 0.0).rgb;
    c = c + textureSampleLevel(low, lin, in.uv + vec2<f32>(1.0, 1.0) * px, 0.0).rgb;
    return vec4<f32>(c / 16.0 + textureSampleLevel(cur, lin, in.uv, 0.0).rgb, 1.0);
}
"""

_SHAFTS = _VS + _UNIFORMS + """
@group(0) @binding(1) var depth_tex: texture_depth_2d;

fn lit(uv: vec2<f32>) -> vec3<f32> {
    let px = clamp(vec2<i32>(uv * u.size), vec2<i32>(0), vec2<i32>(u.size) - vec2<i32>(1));
    if textureLoad(depth_tex, px, 0) > 0.0 {
        return vec3<f32>(0.0);  // geometry blocks the sky
    }
    // the sun's disk and halo, HDR, wherever the sky shows
    return min(sun_light(ray(uv)), vec3<f32>(16.0));
}

@fragment
fn fs(in: VOut) -> @location(0) vec4<f32> {
    if u.shafts.w <= 0.0 {
        return vec4<f32>(0.0, 0.0, 0.0, 1.0);
    }
    let n = i32(u.flags.w);
    let delta = (in.uv - u.sun_uv) * u.shafts.z / f32(max(n, 1));
    var uv = in.uv;
    var fall = 1.0;
    var sum = vec3<f32>(0.0);
    for (var i = 0; i < n; i = i + 1) {
        uv = uv - delta;
        sum = sum + lit(uv) * fall;
        fall = fall * u.shafts.y;
    }
    return vec4<f32>(sum * (1.0 / f32(max(n, 1))) * u.shafts.w, 1.0);
}
"""

_COMPOSITE = _VS + _UNIFORMS + """
@group(0) @binding(1) var scene: texture_2d<f32>;
@group(0) @binding(2) var lin: sampler;
@group(0) @binding(3) var ao_tex: texture_2d<f32>;
@group(0) @binding(4) var bloom_tex: texture_2d<f32>;
@group(0) @binding(5) var shafts_tex: texture_2d<f32>;
@group(0) @binding(6) var depth_tex: texture_depth_2d;

@fragment
fn fs(in: VOut) -> @location(0) vec4<f32> {
    let px = vec2<i32>(floor(in.pos.xy));
    var c = textureLoad(scene, px, 0).rgb;
    if u.sky.w > 0.0 && textureLoad(depth_tex, px, 0) <= 0.0 {
        let d = ray(in.uv);
        let sun = sun_light(d);
        c = sky_base(d) + vec3<f32>(1.0) - exp(-sun);
    }
    if u.flags.x != 0u {
        c = c * mix(1.0, textureSampleLevel(ao_tex, lin, in.uv, 0.0).r, u.ssao.y);
    }
    var add = vec3<f32>(0.0);
    if u.flags.y != 0u {
        add = add + textureSampleLevel(bloom_tex, lin, in.uv, 0.0).rgb * u.bloom.z;
    }
    if u.flags.z != 0u {
        add = add + textureSampleLevel(shafts_tex, lin, in.uv, 0.0).rgb * u.shafts.x;
    }
    c = c + vec3<f32>(1.0) - exp(-add);  // added light saturates toward white rather than clipping
    let lum = dot(c, vec3<f32>(0.2126, 0.7152, 0.0722));
    c = max(mix(vec3<f32>(lum), c, u.grade.x), vec3<f32>(0.0));
    let g = (sqrt(c) - 0.5) * u.grade.y + 0.5;
    c = max(g, vec3<f32>(0.0)) * max(g, vec3<f32>(0.0));
    let d = (in.uv - 0.5) * vec2<f32>(u.size.x / u.size.y, 1.0);
    c = c * (1.0 - u.grade.z * dot(d, d));
    return vec4<f32>(clamp(c, vec3<f32>(0.0), vec3<f32>(1.0)), 1.0);
}
"""

_FXAA = _VS + """
@group(0) @binding(1) var src: texture_2d<f32>;
@group(0) @binding(2) var lin: sampler;

fn luma(c: vec3<f32>) -> f32 {
    return sqrt(dot(c, vec3<f32>(0.299, 0.587, 0.114)));
}

fn at(uv: vec2<f32>) -> vec3<f32> {
    return textureSampleLevel(src, lin, uv, 0.0).rgb;
}

@fragment
fn fs(in: VOut) -> @location(0) vec4<f32> {
    // FXAA in Lottes' PC console form: five luma taps find the edge, four along it blend.
    let px = 1.0 / vec2<f32>(textureDimensions(src));
    let uv = in.uv;
    let m = at(uv);
    let l_nw = luma(at(uv + vec2<f32>(-1.0, -1.0) * px));
    let l_ne = luma(at(uv + vec2<f32>(1.0, -1.0) * px));
    let l_sw = luma(at(uv + vec2<f32>(-1.0, 1.0) * px));
    let l_se = luma(at(uv + vec2<f32>(1.0, 1.0) * px));
    let l_m = luma(m);
    let lo = min(l_m, min(min(l_nw, l_ne), min(l_sw, l_se)));
    let hi = max(l_m, max(max(l_nw, l_ne), max(l_sw, l_se)));
    if hi - lo < max(0.0312, hi * 0.125) {
        return vec4<f32>(m, 1.0);
    }
    var dir = vec2<f32>(-((l_nw + l_ne) - (l_sw + l_se)), (l_nw + l_sw) - (l_ne + l_se));
    let reduce = max((l_nw + l_ne + l_sw + l_se) * 0.25 * 0.125, 1.0 / 128.0);
    let rcp = 1.0 / (min(abs(dir.x), abs(dir.y)) + reduce);
    dir = clamp(dir * rcp, vec2<f32>(-8.0), vec2<f32>(8.0)) * px;
    let a = 0.5 * (at(uv + dir * (1.0 / 3.0 - 0.5)) + at(uv + dir * (2.0 / 3.0 - 0.5)));
    let b = a * 0.5 + 0.25 * (at(uv - dir * 0.5) + at(uv + dir * 0.5));
    let l_b = luma(b);
    if l_b < lo || l_b > hi {
        return vec4<f32>(a, 1.0);
    }
    return vec4<f32>(b, 1.0);
}
"""


class PostPasses:
    """The GPU side of Post: its targets at the frame's size, its pipelines and its bind groups."""

    HDR = "rgba16float"

    def __init__(self, device, fmt):
        import wgpu

        self.device, self.fmt = device, fmt
        self.size = (0, 0)
        self.depth_view = None
        self.ubuf = device.create_buffer(size=UNIFORM_BYTES, usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST)
        self.lin = device.create_sampler(mag_filter="linear", min_filter="linear",
                                         address_mode_u="clamp-to-edge", address_mode_v="clamp-to-edge")
        self._pipelines: dict = {}
        self._groups: dict = {}
        self._kernel = None

    def _pipeline(self, name, code, fmt):
        key = (name, fmt)
        if key not in self._pipelines:
            module = self.device.create_shader_module(code=code)
            self._pipelines[key] = self.device.create_render_pipeline(
                layout="auto",
                vertex={"module": module, "entry_point": "vs"},
                fragment={"module": module, "entry_point": "fs", "targets": [{"format": fmt}]},
                primitive={"topology": "triangle-list"},
            )
        return self._pipelines[key]

    def _tex(self, w, h, fmt):
        import wgpu

        t = self.device.create_texture(size=(max(1, w), max(1, h), 1), format=fmt,
                                       usage=wgpu.TextureUsage.RENDER_ATTACHMENT | wgpu.TextureUsage.TEXTURE_BINDING)
        return t.create_view()

    def resize(self, size):
        if size == self.size:
            return
        w, h = size
        self.size = size
        self.scene = self._tex(w, h, self.fmt)
        self.ldr = self._tex(w, h, self.fmt)
        self.ao = self._tex((w + 1) // 2, (h + 1) // 2, "r8unorm")
        self.ao_blur = self._tex((w + 1) // 2, (h + 1) // 2, "r8unorm")
        self.down = [self._tex(max(1, w >> k), max(1, h >> k), self.HDR) for k in (1, 2, 3, 4)]
        self.up = [self._tex(max(1, w >> k), max(1, h >> k), self.HDR) for k in (1, 2, 3)]
        self.shafts = self._tex(max(1, w // 4), max(1, h // 4), self.HDR)
        self._groups = {}

    def _group(self, name, pipeline, views):
        """views[i] is binding i: "u" the uniforms, None no binding, else a texture view or sampler."""
        key = (name, id(self.depth_view))
        if key not in self._groups:
            entries = []
            for i, v in enumerate(views):
                if v is None:
                    continue
                res = {"buffer": self.ubuf, "offset": 0, "size": UNIFORM_BYTES} if isinstance(v, str) else v
                entries.append({"binding": i, "resource": res})
            self._groups[key] = self.device.create_bind_group(layout=pipeline.get_bind_group_layout(0),
                                                              entries=entries)
        return self._groups[key]

    def upload(self, post: Post, camera, sun, background=(0.0, 0.0, 0.0)):
        w, h = self.size
        proj = camera.get_projection_matrix(w / h, near=camera.near, far=camera.far).astype(np.float64)
        sun_u, sun_v, visible = sun_on_screen(camera, w / h, sun) if post.shafts else (0.5, 0.5, 0.0)
        s, b, sh, g = post.ssao or SSAO(), post.bloom or Bloom(), post.shafts or LightShafts(), post.grade or Grade()
        sun = sun if sun is not None and float(sun.intensity) > 0 else None
        if self._kernel is None or self._kernel[0] != s.samples:
            self._kernel = (s.samples, ssao_kernel(min(int(s.samples), 32)))
        data = np.zeros(UNIFORM_BYTES // 4, np.float32)
        data[0:16] = proj.T.astype(np.float32).ravel()
        data[16:32] = np.linalg.inv(proj).T.astype(np.float32).ravel()
        data[32:36] = (w, h, sun_u, sun_v)
        data[36:40] = (s.radius, s.intensity, min(int(s.samples), 32), s.bias)
        data[40:44] = (b.threshold, b.knee, b.intensity, 0.0)
        data[44:48] = (sh.intensity, sh.decay, sh.density, visible)
        data[48:52] = (g.saturation, g.contrast, g.vignette, s.power)
        flags = np.array([post.ssao is not None, post.bloom is not None, post.shafts is not None and visible > 0,
                          int(sh.samples)], np.uint32)
        data[52:56] = flags.view(np.float32)
        if sun is not None:
            from manifoldx.resources import material_rgb

            data[56:59] = np.clip(material_rgb(sun.color) * min(float(sun.intensity), 1.0), 0.0, 1.0)
            if np.max(data[56:59]) > 0:
                data[56:59] /= np.max(data[56:59])  # the disk's colour; its brightness is Sky.sun_intensity
        view = camera.get_view_matrix().astype(np.float64)
        data[60:76] = np.linalg.inv(view).T.astype(np.float32).ravel()
        if sun is not None:
            to_sun = -np.asarray(sun.direction, float)
            data[76:80] = (*(to_sun / (np.linalg.norm(to_sun) or 1.0)), 1.0)
        sky = post.sky or Sky()
        data[80:83] = sky.zenith
        data[84:87] = sky.horizon if sky.horizon is not None else tuple(background)[:3]
        data[88:92] = (np.cos(np.radians(sky.sun_size)), sky.sun_intensity, sky.glow, 1.0 if post.sky else 0.0)
        data[92:92 + 128] = self._kernel[1].ravel()
        from manifoldx import counters

        counters.write_buffer(self.device.queue, self.ubuf, 0, data.tobytes())
        return bool(flags[2])

    def _draw(self, encoder, target, pipeline, group, timer=None):
        rp = encoder.begin_render_pass(color_attachments=[{
            "view": target, "resolve_target": None, "clear_value": (0, 0, 0, 1),
            "load_op": "clear", "store_op": "store"}], timestamp_writes=timer)
        rp.set_pipeline(pipeline)
        rp.set_bind_group(0, group)
        rp.draw(3)
        rp.end()

    def run(self, post: Post, encoder, out_view, depth_view, camera, sun, timer=None, background=(0.0, 0.0, 0.0)):
        """Draw the effects from self.scene and the depth into out_view (the swapchain's)."""
        self.depth_view = depth_view
        shafts_on = self.upload(post, camera, sun, background)
        steps = []
        if post.ssao:
            p = self._pipeline("ssao", _SSAO, "r8unorm")
            steps.append((self.ao, p, self._group("ssao", p, ["u", depth_view])))
            p = self._pipeline("ssao_blur", _SSAO_BLUR, "r8unorm")
            steps.append((self.ao_blur, p, self._group("ssao_blur", p, ["u", depth_view, self.ao])))
        if post.bloom:
            p = self._pipeline("bright", _BRIGHT, self.HDR)
            steps.append((self.down[0], p, self._group("bright", p, ["u", self.scene, self.lin, depth_view])))
            for k in (1, 2, 3):
                p = self._pipeline("down", _DOWN, self.HDR)
                steps.append((self.down[k], p, self._group(f"down{k}", p, [None, self.down[k - 1], self.lin])))
            for k in (2, 1, 0):  # up[k] = down[k] + the blurred level under it
                low = self.down[3] if k == 2 else self.up[k + 1]
                p = self._pipeline("up", _UP, self.HDR)
                steps.append((self.up[k], p, self._group(f"up{k}", p, [None, low, self.lin, self.down[k]])))
        if shafts_on:
            p = self._pipeline("shafts", _SHAFTS, self.HDR)
            steps.append((self.shafts, p, self._group("shafts", p, ["u", depth_view])))
        final = self.ldr if post.fxaa else out_view
        p = self._pipeline("composite", _COMPOSITE, self.fmt)
        ao = self.ao_blur if post.ssao else self.ao
        steps.append((final, p, self._group(f"composite{post.fxaa}", p,
                                            ["u", self.scene, self.lin, ao, self.up[0], self.shafts, depth_view])))
        if post.fxaa:
            p = self._pipeline("fxaa", _FXAA, self.fmt)
            steps.append((out_view, p, self._group("fxaa", p, [None, self.ldr, self.lin])))
        for i, (target, pipeline, group) in enumerate(steps):
            writes = None
            if timer is not None:
                writes = timer.writes("post", begin=i == 0, end=i == len(steps) - 1)
            self._draw(encoder, target, pipeline, group, writes)
