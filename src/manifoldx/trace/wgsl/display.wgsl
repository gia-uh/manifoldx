// The page's last step: auto exposure (a one-workgroup reduction), then ACES and sRGB onto the
// canvas through a full-screen triangle. Python does the same in numpy (tracer.to_srgb8).

struct View {
    width: u32, height: u32,   // the trace
    mode: u32,                 // 0: accumulated radiance, 1: filtered (remodulated by albedo)
    exposure: f32,             // a multiplier; <= 0 means auto. The page copies scale[0] here to draw
    out_width: f32, out_height: f32,  // the canvas
    tonemap: u32,              // 0: aces, 1: reinhard, 2: clamp
    pad: u32,
};

@group(0) @binding(0) var<uniform> V: View;
@group(0) @binding(1) var<storage, read> accum: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> aov: array<vec4<f32>>;       // 2 per pixel, as path.wgsl
@group(0) @binding(3) var<storage, read> filtered: array<vec4<f32>>;  // the last à-trous output
@group(0) @binding(4) var<storage, read_write> scale: array<f32>;     // [0]: the exposure, expose only

fn lum(c: vec3<f32>) -> f32 { return dot(c, vec3<f32>(0.2126, 0.7152, 0.0722)); }

fn radiance(i: u32) -> vec3<f32> {
    let a = accum[i];
    let n = max(a.w, 1.0);
    if (V.mode == 0u) { return a.xyz / n; }
    return filtered[i].xyz * max(aov[2u * i].xyz / n, vec3<f32>(0.02));
}

var<workgroup> partial: array<f32, 256>;

@compute @workgroup_size(256)
fn expose(@builtin(local_invocation_index) k: u32) {
    let total = V.width * V.height;
    var s = 0.0;
    for (var i = k; i < total; i = i + 256u) { s += log(max(lum(radiance(i)), 1.0e-4)); }
    partial[k] = s;
    workgroupBarrier();
    for (var w = 128u; w > 0u; w = w >> 1u) {
        if (k < w) { partial[k] += partial[k + w]; }
        workgroupBarrier();
    }
    if (k == 0u) { scale[0] = select(V.exposure, 0.18 / exp(partial[0] / f32(total)), V.exposure <= 0.0); }
}

@vertex
fn vs(@builtin(vertex_index) i: u32) -> @builtin(position) vec4<f32> {
    let p = vec2<f32>(f32((i << 1u) & 2u), f32(i & 2u));
    return vec4<f32>(p * 2.0 - 1.0, 0.0, 1.0);
}

@fragment
fn fs(@builtin(position) pos: vec4<f32>) -> @location(0) vec4<f32> {
    let x = min(u32(pos.x * f32(V.width) / V.out_width), V.width - 1u);
    let y = min(u32(pos.y * f32(V.height) / V.out_height), V.height - 1u);
    var c = max(radiance(y * V.width + x) * V.exposure, vec3<f32>(0.0));
    if (V.tonemap == 0u) {  // Narkowicz's fit of the ACES filmic curve
        c = clamp((c * (2.51 * c + 0.03)) / (c * (2.43 * c + 0.59) + 0.14), vec3<f32>(0.0), vec3<f32>(1.0));
    } else if (V.tonemap == 1u) {
        c = c / (1.0 + c);
    }
    c = clamp(c, vec3<f32>(0.0), vec3<f32>(1.0));
    c = select(1.055 * pow(c, vec3<f32>(1.0 / 2.4)) - 0.055, c * 12.92, c <= vec3<f32>(0.0031308));
    return vec4<f32>(c, 1.0);
}
