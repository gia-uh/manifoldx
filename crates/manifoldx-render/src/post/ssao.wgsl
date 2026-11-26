@group(0) @binding(1) var depth_tex: texture_depth_2d;

fn view_at(px: vec2<i32>) -> vec3<f32> {
    let q = clamp(px, vec2<i32>(0), vec2<i32>(u.size) - vec2<i32>(1));
    let d = textureLoad(depth_tex, q, 0);
    let uv = (vec2<f32>(q) + 0.5) / u.size;
    let v = u.inv_proj * vec4<f32>(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0, d, 1.0);
    return v.xyz / v.w;
}

fn hash(p: vec2<f32>) -> f32 {
    return fract(sin(dot(p, vec2<f32>(12.9898, 78.233))) * 43758.5453);
}

@fragment
fn fs(in: VOut) -> @location(0) vec4<f32> {
    let px = vec2<i32>(floor(in.pos.xy) * 2.0);
    let q = clamp(px, vec2<i32>(0), vec2<i32>(u.size) - vec2<i32>(1));
    if textureLoad(depth_tex, q, 0) <= 0.0 {
        return vec4<f32>(1.0);  // the sky
    }
    let p = view_at(px);
    let r = view_at(px + vec2<i32>(1, 0));
    let l = view_at(px - vec2<i32>(1, 0));
    let d = view_at(px + vec2<i32>(0, 1));
    let t = view_at(px - vec2<i32>(0, 1));
    let dx = select(p - l, r - p, abs(r.z - p.z) < abs(p.z - l.z));
    let dy = select(p - t, d - p, abs(d.z - p.z) < abs(p.z - t.z));
    var n = normalize(cross(dx, dy));
    if dot(n, -p) < 0.0 {
        n = -n;
    }
    let a = hash(floor(in.pos.xy)) * 6.2831853;
    let rv = vec3<f32>(cos(a), sin(a), 0.0);
    let tg = normalize(rv - n * dot(rv, n) + vec3<f32>(1e-4, 0.0, 0.0));
    let bt = cross(n, tg);
    let radius = u.ssao.x;
    let count = i32(u.ssao.z);
    var occ = 0.0;
    for (var i = 0; i < count; i = i + 1) {
        let k = u.kernel[i].xyz;
        let s = p + (tg * k.x + bt * k.y + n * k.z) * radius;
        let c = u.proj * vec4<f32>(s, 1.0);
        let suv = vec2<f32>(c.x / c.w * 0.5 + 0.5, 0.5 - c.y / c.w * 0.5);
        if any(suv < vec2<f32>(0.0)) || any(suv > vec2<f32>(1.0)) {
            continue;
        }
        let sz = view_at(vec2<i32>(suv * u.size)).z;
        let range = smoothstep(0.0, 1.0, radius / max(abs(p.z - sz), 1e-4));
        occ = occ + select(0.0, 1.0, sz >= s.z + u.ssao.w) * range;
    }
    let ao = pow(1.0 - occ / f32(max(count, 1)), max(u.grade.w, 0.01));
    return vec4<f32>(ao, ao, ao, 1.0);
}
