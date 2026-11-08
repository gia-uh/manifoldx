// Primary rays through pixel centres: albedo, shading normal and distance of the first hit.

@group(0) @binding(5) var<storage, read_write> out_albedo: array<vec4<f32>>;
@group(0) @binding(6) var<storage, read_write> out_normal_depth: array<vec4<f32>>;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= P.width || gid.y >= P.height) { return; }
    let pixel = gid.y * P.width + gid.x;
    let d = camera_ray(vec2<f32>(gid.xy) + 0.5);
    let hit = trace(P.cam_pos, d, INF);
    if (hit.tri == 0xffffffffu) {
        out_albedo[pixel] = vec4<f32>(0.0);
        out_normal_depth[pixel] = vec4<f32>(0.0, 0.0, 0.0, -1.0);
        return;
    }
    let s = surface(hit, P.cam_pos, d);
    out_albedo[pixel] = vec4<f32>(s.albedo, 1.0);
    out_normal_depth[pixel] = vec4<f32>(s.n, hit.t);
}
