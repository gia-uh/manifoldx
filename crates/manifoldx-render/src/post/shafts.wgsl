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
