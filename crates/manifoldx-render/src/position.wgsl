struct Args {
    width: u32,
    offset: u32,
    count: u32,
    unused: u32,
};

@group(0) @binding(0) var<storage, read> positions: array<f32>;
@group(0) @binding(1) var<storage, read> bounds: array<u32>;
@group(0) @binding(2) var<storage, read_write> instances: array<f32>;
@group(0) @binding(3) var<uniform> view_projection: mat4x4<f32>;
@group(0) @binding(4) var<uniform> args: Args;

fn outside(clip: mat4x4<f32>, lo: vec3<f32>, hi: vec3<f32>) -> bool {
    var outside_plane = array<bool, 6>(true, true, true, true, true, true);
    for (var i = 0u; i < 8u; i++) {
        let p = select(lo, hi, (vec3<u32>(i) & vec3<u32>(1u, 2u, 4u)) != vec3<u32>(0u));
        let v = clip * vec4<f32>(p, 1.0);
        let tests = array<bool, 6>(v.x < -v.w, v.x > v.w, v.y < -v.w, v.y > v.w, v.z < 0.0, v.z > v.w);
        for (var plane = 0u; plane < 6u; plane++) {
            outside_plane[plane] = outside_plane[plane] && tests[plane];
        }
    }
    for (var plane = 0u; plane < 6u; plane++) {
        if outside_plane[plane] { return true; }
    }
    return false;
}

@compute @workgroup_size(64)
fn place(@builtin(global_invocation_id) id: vec3<u32>) {
    if id.x >= args.count { return; }
    let row = id.x * 8u;
    let slot = bounds[row];
    if slot == 0xffffffffu { return; }
    let p = slot * args.width + args.offset;
    let pos = vec3<f32>(positions[p], positions[p + 1u], positions[p + 2u]);
    let m = id.x * 24u;
    let model = mat4x4<f32>(
        vec4<f32>(instances[m], instances[m + 1u], instances[m + 2u], 0.0),
        vec4<f32>(instances[m + 4u], instances[m + 5u], instances[m + 6u], 0.0),
        vec4<f32>(instances[m + 8u], instances[m + 9u], instances[m + 10u], 0.0),
        vec4<f32>(pos, 1.0),
    );
    let lo = bitcast<vec3<f32>>(vec3<u32>(bounds[row + 1u], bounds[row + 2u], bounds[row + 3u]));
    let hi = bitcast<vec3<f32>>(vec3<u32>(bounds[row + 4u], bounds[row + 5u], bounds[row + 6u]));
    instances[m + 12u] = pos.x;
    instances[m + 13u] = pos.y;
    instances[m + 14u] = pos.z;
    instances[m + 15u] = select(1.0, 0.0, outside(view_projection * model, lo, hi));
}
