// One à-trous pass over demodulated radiance, edge-stopped by normal, depth and luminance
// (Dammertz et al. 2010; variance-guided as in SVGF). P.frame holds this pass's step in pixels.

@group(0) @binding(7) var<storage, read> src: array<vec4<f32>>;         // irradiance rgb, variance
@group(0) @binding(8) var<storage, read_write> dst: array<vec4<f32>>;
@group(0) @binding(9) var<storage, read> guide: array<vec4<f32>>;       // normal xyz, depth (< 0: miss)

const SIGMA_L: f32 = 4.0;
const SIGMA_N: f32 = 64.0;
const SIGMA_Z: f32 = 1.0;

fn at(x: i32, y: i32) -> u32 {
    let cx = clamp(x, 0, i32(P.width) - 1);
    let cy = clamp(y, 0, i32(P.height) - 1);
    return u32(cy) * P.width + u32(cx);
}

fn lum(c: vec3<f32>) -> f32 { return dot(c, vec3<f32>(0.2126, 0.7152, 0.0722)); }

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= P.width || gid.y >= P.height) { return; }
    let x = i32(gid.x);
    let y = i32(gid.y);
    let i = at(x, y);
    let c = src[i];
    let g = guide[i];
    if (g.w < 0.0) { dst[i] = c; return; }
    let step = i32(P.frame);

    // how fast depth changes per pixel here, so slanted surfaces are not cut apart
    let zx = abs(guide[at(x + 1, y)].w - guide[at(x - 1, y)].w) * 0.5;
    let zy = abs(guide[at(x, y + 1)].w - guide[at(x, y - 1)].w) * 0.5;
    let grad = min(max(zx, zy), g.w * 0.5);
    let sigma_l = SIGMA_L * sqrt(max(c.w, 0.0)) + 1.0e-6;
    let l = lum(c.xyz);

    var kernel = array<f32, 3>(0.375, 0.25, 0.0625);
    var sum = vec3<f32>(0.0);
    var var_sum = 0.0;
    var w_sum = 0.0;
    for (var dy = -2; dy <= 2; dy = dy + 1) {
        for (var dx = -2; dx <= 2; dx = dx + 1) {
            let j = at(x + dx * step, y + dy * step);
            let gq = guide[j];
            if (gq.w < 0.0) { continue; }
            let q = src[j];
            let h = kernel[abs(dx)] * kernel[abs(dy)];
            let dist = length(vec2<f32>(f32(dx), f32(dy))) * f32(step);
            let wn = pow(max(dot(g.xyz, gq.xyz), 0.0), SIGMA_N);
            let wz = exp(-abs(g.w - gq.w) / (SIGMA_Z * grad * dist + 1.0e-3 * g.w));
            let wl = exp(-abs(l - lum(q.xyz)) / sigma_l);
            let w = h * wn * wz * wl;
            sum += q.xyz * w;
            var_sum += q.w * w * w;
            w_sum += w;
        }
    }
    if (w_sum <= 1.0e-12) { dst[i] = c; return; }
    dst[i] = vec4<f32>(sum / w_sum, var_sum / (w_sum * w_sum));
}
