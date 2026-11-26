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
