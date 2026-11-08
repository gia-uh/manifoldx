// The denoiser's input from the accumulation: radiance divided by first-hit albedo, its
// per-pixel variance blurred 3×3, and the guide (unit normal, distance). P.frame is unused.

@group(0) @binding(5) var<storage, read> accum: array<vec4<f32>>;
@group(0) @binding(6) var<storage, read> aov: array<vec4<f32>>;     // 2 per pixel, as path.wgsl
@group(0) @binding(7) var<storage, read_write> src: array<vec4<f32>>;   // irradiance rgb, variance
@group(0) @binding(8) var<storage, read_write> guide: array<vec4<f32>>; // normal xyz, depth (< 0: miss)

fn lum(c: vec3<f32>) -> f32 { return dot(c, vec3<f32>(0.2126, 0.7152, 0.0722)); }

fn at(x: i32, y: i32) -> u32 {
    let cx = clamp(x, 0, i32(P.width) - 1);
    let cy = clamp(y, 0, i32(P.height) - 1);
    return u32(cy) * P.width + u32(cx);
}

fn albedo_of(i: u32, n: f32) -> vec3<f32> { return max(aov[2u * i].xyz / n, vec3<f32>(0.02)); }

// variance of the mean luminance, demodulated by the albedo's luminance
fn variance(i: u32) -> f32 {
    let a = accum[i];
    let n = max(a.w, 1.0);
    let l = lum(a.xyz / n);
    let la = max(lum(albedo_of(i, n)), 0.02);
    return max(aov[2u * i].w / n - l * l, 0.0) / n / (la * la);
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= P.width || gid.y >= P.height) { return; }
    let x = i32(gid.x);
    let y = i32(gid.y);
    let i = at(x, y);
    let a = accum[i];
    let n = max(a.w, 1.0);
    var k = array<f32, 3>(1.0, 2.0, 1.0);
    var v = 0.0;
    for (var dy = -1; dy <= 1; dy = dy + 1) {
        for (var dx = -1; dx <= 1; dx = dx + 1) {
            v += k[dx + 1] * k[dy + 1] * variance(at(x + dx, y + dy));
        }
    }
    src[i] = vec4<f32>(a.xyz / n / albedo_of(i, n), v / 16.0);
    let nd = aov[2u * i + 1u] / n;
    guide[i] = vec4<f32>(nd.xyz / max(length(nd.xyz), 1.0e-6), nd.w);
}
