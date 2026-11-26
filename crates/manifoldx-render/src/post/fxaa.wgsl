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
