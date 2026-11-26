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
