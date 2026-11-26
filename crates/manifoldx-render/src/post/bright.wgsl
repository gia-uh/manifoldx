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
    return min(c, vec3<f32>(16.0));
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
