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
