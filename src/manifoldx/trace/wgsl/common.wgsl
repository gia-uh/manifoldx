// Scene buffers, random numbers, BVH traversal and surface lookup, shared by every trace kernel.

struct Params {
    cam_pos: vec3<f32>, tan_half_fov: f32,
    cam_fwd: vec3<f32>, aspect: f32,
    cam_right: vec3<f32>, width: u32,
    cam_up: vec3<f32>, height: u32,
    frame: u32, seed: u32, max_bounces: u32, light_count: u32,
    env_width: u32, env_height: u32, env_intensity: f32, emissive_count: u32,
    sun_dir: vec3<f32>, sun_cos: f32,       // direction towards the sun, cos of its angular radius
    sun_radiance: vec3<f32>, env_sum: f32,  // env_sum: total of the env sampling weights
    clamp_max: f32, scene_radius: f32, emissive_total: f32, flags: u32,
};

@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var<storage, read> nodes: array<vec4<f32>>;   // 2 per node
@group(0) @binding(2) var<storage, read> tris: array<vec4<f32>>;    // 3 per triangle: v0, e1, e2
@group(0) @binding(3) var<storage, read> shade: array<vec4<f32>>;   // 5 per triangle
@group(0) @binding(4) var<storage, read> mats: array<vec4<f32>>;    // 3 per material
@group(0) @binding(5) var<storage, read> texels: array<u32>;        // RGBA8, sRGB colour
@group(0) @binding(6) var<storage, read> tex_table: array<vec4<u32>>; // offset, width, height, 0

const PI: f32 = 3.14159265358979;
const INF: f32 = 3.0e38;
const EPS: f32 = 1.0e-4;

// ---------------------------------------------------------------- random numbers (PCG)

fn pcg(v: u32) -> u32 {
    let s = v * 747796405u + 2891336453u;
    let w = ((s >> ((s >> 28u) + 4u)) ^ s) * 277803737u;
    return (w >> 22u) ^ w;
}

struct Rng { state: u32 };

fn rng_init(pixel: u32, frame: u32, seed: u32) -> Rng {
    return Rng(pcg(pixel ^ pcg(frame * 9781u + seed * 6271u)));
}

fn rand(r: ptr<function, Rng>) -> f32 {
    (*r).state = pcg((*r).state);
    return f32((*r).state >> 8u) * (1.0 / 16777216.0);
}

// ---------------------------------------------------------------- traversal

struct Hit { t: f32, u: f32, v: f32, tri: u32 };

fn box_hit(lo: vec3<f32>, hi: vec3<f32>, o: vec3<f32>, inv: vec3<f32>, tmax: f32) -> f32 {
    let t0 = (lo - o) * inv;
    let t1 = (hi - o) * inv;
    let tn = max(max(min(t0.x, t1.x), min(t0.y, t1.y)), max(min(t0.z, t1.z), 0.0));
    let tf = min(min(max(t0.x, t1.x), max(t0.y, t1.y)), min(max(t0.z, t1.z), tmax));
    return select(INF, tn, tn <= tf);
}

fn safe_inv(d: vec3<f32>) -> vec3<f32> {
    let s = select(vec3<f32>(-1.0), vec3<f32>(1.0), d >= vec3<f32>(0.0));
    return 1.0 / select(d, s * 1.0e-20, abs(d) < vec3<f32>(1.0e-20));
}

// Closest hit along o + t d with t in (EPS, tmax); hit.tri == 0xffffffff on a miss.
fn trace(o: vec3<f32>, d: vec3<f32>, tmax: f32) -> Hit {
    var hit = Hit(tmax, 0.0, 0.0, 0xffffffffu);
    let inv = safe_inv(d);
    var stack: array<u32, 40>;
    var sp = 0u;
    var idx = 0u;
    if (box_hit(nodes[0].xyz, nodes[1].xyz, o, inv, tmax) >= INF) { return hit; }
    loop {
        let a = nodes[2u * idx];
        let b = nodes[2u * idx + 1u];
        let first = bitcast<u32>(a.w);
        let count = bitcast<u32>(b.w);
        if (count > 0u) {
            for (var i = first; i < first + count; i = i + 1u) {
                let v0 = tris[3u * i].xyz;
                let e1 = tris[3u * i + 1u].xyz;
                let e2 = tris[3u * i + 2u].xyz;
                let h = cross(d, e2);
                let det = dot(e1, h);
                if (abs(det) < 1.0e-12) { continue; }
                let f = 1.0 / det;
                let s = o - v0;
                let u = f * dot(s, h);
                let q = cross(s, e1);
                let v = f * dot(d, q);
                let t = f * dot(e2, q);
                if (u >= 0.0 && v >= 0.0 && u + v <= 1.0 && t > EPS && t < hit.t) {
                    hit = Hit(t, u, v, i);
                }
            }
            if (sp == 0u) { break; }
            sp = sp - 1u;
            idx = stack[sp];
            continue;
        }
        let l = first;
        let r = first + 1u;
        var tl = box_hit(nodes[2u * l].xyz, nodes[2u * l + 1u].xyz, o, inv, hit.t);
        var tr = box_hit(nodes[2u * r].xyz, nodes[2u * r + 1u].xyz, o, inv, hit.t);
        var near = l;
        var far = r;
        if (tr < tl) { let tt = tl; tl = tr; tr = tt; near = r; far = l; }
        if (tl >= INF) {
            if (sp == 0u) { break; }
            sp = sp - 1u;
            idx = stack[sp];
            continue;
        }
        idx = near;
        if (tr < INF && sp < 40u) { stack[sp] = far; sp = sp + 1u; }
    }
    return hit;
}

// ---------------------------------------------------------------- surfaces

struct Surface {
    p: vec3<f32>,
    n: vec3<f32>,      // shading normal, facing the incoming ray
    ng: vec3<f32>,     // geometric normal, facing the incoming ray
    front: bool,       // the ray hit the side the geometric normal points to
    albedo: vec3<f32>,
    metallic: f32,
    roughness: f32,
    emission: vec3<f32>,
    transmission: f32,
    ior: f32,
    thin: bool,
};

fn unpack_color(c: u32) -> vec3<f32> {
    return vec3<f32>(f32(c & 2047u) / 2047.0, f32((c >> 11u) & 2047u) / 2047.0, f32(c >> 22u) / 1023.0);
}

fn srgb_to_linear(c: vec3<f32>) -> vec3<f32> {
    return select(pow((c + 0.055) / 1.055, vec3<f32>(2.4)), c / 12.92, c <= vec3<f32>(0.04045));
}

fn texel(base: u32, w: u32, h: u32, x: i32, y: i32) -> vec4<f32> {
    let xi = u32(((x % i32(w)) + i32(w)) % i32(w));
    let yi = u32(((y % i32(h)) + i32(h)) % i32(h));
    return unpack4x8unorm(texels[base + yi * w + xi]);
}

fn sample_texture(t: u32, uv: vec2<f32>) -> vec3<f32> {
    let e = tex_table[t];
    let p = uv * vec2<f32>(f32(e.y), f32(e.z)) - 0.5;
    let f = fract(p);
    let i = vec2<i32>(floor(p));
    let a = mix(texel(e.x, e.y, e.z, i.x, i.y), texel(e.x, e.y, e.z, i.x + 1, i.y), f.x);
    let b = mix(texel(e.x, e.y, e.z, i.x, i.y + 1), texel(e.x, e.y, e.z, i.x + 1, i.y + 1), f.x);
    return srgb_to_linear(mix(a, b, f.y).rgb);
}

fn surface(hit: Hit, o: vec3<f32>, d: vec3<f32>) -> Surface {
    let i = hit.tri;
    let w = 1.0 - hit.u - hit.v;
    let s0 = shade[5u * i];
    let s1 = shade[5u * i + 1u];
    let s2 = shade[5u * i + 2u];
    let s3 = shade[5u * i + 3u];
    let s4 = bitcast<vec4<u32>>(shade[5u * i + 4u]);
    let mat = bitcast<u32>(s0.w);
    let m0 = mats[3u * mat];
    let m1 = mats[3u * mat + 1u];
    let m2 = mats[3u * mat + 2u];

    var s: Surface;
    s.p = o + d * hit.t;
    var ng = normalize(cross(tris[3u * i + 1u].xyz, tris[3u * i + 2u].xyz));
    var n = normalize(s0.xyz * w + s1.xyz * hit.u + s2.xyz * hit.v);
    s.front = dot(ng, d) < 0.0;
    if (!s.front) { ng = -ng; }
    if (dot(n, ng) < 0.0) { n = -n; }
    s.ng = ng;
    s.n = n;
    var albedo = m0.xyz * (unpack_color(s4.x) * w + unpack_color(s4.y) * hit.u + unpack_color(s4.z) * hit.v);
    let tex = i32(m2.w);
    if (tex >= 0) {
        let uv = vec2<f32>(s1.w, s2.w) * w + s3.xy * hit.u + s3.zw * hit.v;
        albedo = albedo * sample_texture(u32(tex), uv);
    }
    s.albedo = albedo;
    s.metallic = m0.w;
    s.roughness = clamp(m1.w, 0.02, 1.0);
    s.emission = m1.xyz;
    s.transmission = m2.x;
    s.ior = m2.y;
    s.thin = m2.z > 0.5;
    return s;
}

fn camera_ray(px: vec2<f32>) -> vec3<f32> {
    let ndc = vec2<f32>(px.x / f32(P.width) * 2.0 - 1.0, 1.0 - px.y / f32(P.height) * 2.0);
    return normalize(P.cam_fwd + P.cam_right * (ndc.x * P.tan_half_fov * P.aspect)
                     + P.cam_up * (ndc.y * P.tan_half_fov));
}
