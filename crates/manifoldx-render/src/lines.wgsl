struct Globals { vp: mat4x4<f32>, viewport: vec4<f32> };
struct Seg { a: vec4<f32>, b: vec4<f32>, color: vec4<f32> };
struct Params { bias: f32, feather: f32, scale: f32, alpha: f32 };

@group(0) @binding(0) var<uniform> globals: Globals;
@group(0) @binding(1) var<storage, read> segs: array<Seg>;
@group(0) @binding(2) var<uniform> params: Params;

struct VSOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) across: f32,
    @location(1) half_w: f32,
    @location(2) color: vec4<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> VSOut {
    let s = segs[ii];
    var ca = globals.vp * vec4<f32>(s.a.xyz, 1.0);
    var cb = globals.vp * vec4<f32>(s.b.xyz, 1.0);
    let eps = 1e-3;  // an end behind the camera moves along the segment to just in front of it
    if (ca.w < eps && cb.w >= eps) { ca = mix(ca, cb, (eps - ca.w) / (cb.w - ca.w)); }
    if (cb.w < eps && ca.w >= eps) { cb = mix(cb, ca, (eps - cb.w) / (ca.w - cb.w)); }
    let half_vp = globals.viewport.xy * 0.5;
    let sa = ca.xy / ca.w * half_vp;
    let sb = cb.xy / cb.w * half_vp;
    let d = sb - sa;
    let len = length(d);
    var dir = vec2<f32>(1.0, 0.0);
    if (len > 1e-4) { dir = d / len; }
    let nrm = vec2<f32>(-dir.y, dir.x);
    let half_w = s.a.w * params.scale * 0.5 + params.feather;
    var ends = array<f32, 6>(0.0, 1.0, 1.0, 0.0, 1.0, 0.0);
    var sides = array<f32, 6>(-1.0, -1.0, 1.0, -1.0, 1.0, 1.0);
    let at_b = ends[vi] > 0.5;
    let side = sides[vi];
    var c = ca;
    var along = -dir;
    if (at_b) { c = cb; along = dir; }
    let off = (nrm * side + along) * half_w / half_vp * c.w;
    var out: VSOut;
    out.pos = vec4<f32>(c.x + off.x, c.y + off.y, c.z * (1.0 + params.bias), c.w);
    if (ca.w < eps && cb.w < eps) { out.pos = vec4<f32>(0.0, 0.0, -1.0, 1.0); }  // all behind: clipped away
    out.across = side * half_w;
    out.half_w = half_w;
    out.color = vec4<f32>(s.color.rgb, s.color.a * params.alpha);
    return out;
}

@fragment
fn fs_main(in: VSOut) -> @location(0) vec4<f32> {
    let a = in.color.a * clamp((in.half_w - abs(in.across)) / max(params.feather, 1e-3), 0.0, 1.0);
    return vec4<f32>(in.color.rgb * a, a);
}
