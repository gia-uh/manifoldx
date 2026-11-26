struct Globals {
    vp:            mat4x4<f32>,
    view:          mat4x4<f32>,
    proj:          mat4x4<f32>,
    camera_pos:    vec3<f32>,
    _pad0:         f32,
    viewport_size: vec2<f32>,
    _pad1:         vec2<f32>,
};

struct VolumeUniforms {
    model:         mat4x4<f32>,
    inv_model:     mat4x4<f32>,
    vmin:          f32,
    vmax:          f32,
    density_scale: f32,
    step_size:     f32,
    max_steps:     u32,
    _pad0:         u32,
    _pad1:         u32,
    _pad2:         u32,
};

@group(0) @binding(0) var<uniform> globals: Globals;
@group(0) @binding(1) var scene_depth: texture_depth_2d;
@group(1) @binding(0) var volume_tex:  texture_3d<f32>;
@group(1) @binding(1) var vol_sampler: sampler;
@group(1) @binding(2) var color_lut:   texture_2d<f32>;
@group(1) @binding(3) var opacity_lut: texture_2d<f32>;
@group(1) @binding(4) var lut_sampler: sampler;
@group(1) @binding(5) var<uniform> vu: VolumeUniforms;

struct VertexOut {
    @builtin(position) clip_pos: vec4<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) vid: u32) -> VertexOut {
    // Oversized fullscreen triangle (covers NDC [-1,1]^2 with 3 verts).
    var pos = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>( 3.0, -1.0),
        vec2<f32>(-1.0,  3.0),
    );
    var out: VertexOut;
    out.clip_pos = vec4<f32>(pos[vid], 1.0, 1.0);
    return out;
}

fn ray_unit_cube(
    ro: vec3<f32>, rd: vec3<f32>,
    t_near: ptr<function, f32>, t_far: ptr<function, f32>,
) -> bool {
    // Slab method against [-0.5, 0.5]^3.
    let inv_rd = vec3<f32>(1.0) / rd;
    let t1 = (vec3<f32>(-0.5) - ro) * inv_rd;
    let t2 = (vec3<f32>( 0.5) - ro) * inv_rd;
    let tmin = min(t1, t2);
    let tmax = max(t1, t2);
    let near = max(max(tmin.x, tmin.y), tmin.z);
    let far  = min(min(tmax.x, tmax.y), tmax.z);
    *t_near = near;
    *t_far  = far;
    return far >= max(near, 0.0);
}

// Cofactor-based 4x4 inverse. Used once per fragment for unprojection.
fn inverse_mat4(m: mat4x4<f32>) -> mat4x4<f32> {
    let a = m[0]; let b = m[1]; let c = m[2]; let d = m[3];
    let det =
        a.x*(b.y*(c.z*d.w - c.w*d.z) - b.z*(c.y*d.w - c.w*d.y) + b.w*(c.y*d.z - c.z*d.y))
      - a.y*(b.x*(c.z*d.w - c.w*d.z) - b.z*(c.x*d.w - c.w*d.x) + b.w*(c.x*d.z - c.z*d.x))
      + a.z*(b.x*(c.y*d.w - c.w*d.y) - b.y*(c.x*d.w - c.w*d.x) + b.w*(c.x*d.y - c.y*d.x))
      - a.w*(b.x*(c.y*d.z - c.z*d.y) - b.y*(c.x*d.z - c.z*d.x) + b.z*(c.x*d.y - c.y*d.x));
    let inv_det = 1.0 / det;
    var r: mat4x4<f32>;
    r[0][0] =  (b.y*(c.z*d.w-c.w*d.z) - b.z*(c.y*d.w-c.w*d.y) + b.w*(c.y*d.z-c.z*d.y)) * inv_det;
    r[0][1] = -(a.y*(c.z*d.w-c.w*d.z) - a.z*(c.y*d.w-c.w*d.y) + a.w*(c.y*d.z-c.z*d.y)) * inv_det;
    r[0][2] =  (a.y*(b.z*d.w-b.w*d.z) - a.z*(b.y*d.w-b.w*d.y) + a.w*(b.y*d.z-b.z*d.y)) * inv_det;
    r[0][3] = -(a.y*(b.z*c.w-b.w*c.z) - a.z*(b.y*c.w-b.w*c.y) + a.w*(b.y*c.z-b.z*c.y)) * inv_det;
    r[1][0] = -(b.x*(c.z*d.w-c.w*d.z) - b.z*(c.x*d.w-c.w*d.x) + b.w*(c.x*d.z-c.z*d.x)) * inv_det;
    r[1][1] =  (a.x*(c.z*d.w-c.w*d.z) - a.z*(c.x*d.w-c.w*d.x) + a.w*(c.x*d.z-c.z*d.x)) * inv_det;
    r[1][2] = -(a.x*(b.z*d.w-b.w*d.z) - a.z*(b.x*d.w-b.w*d.x) + a.w*(b.x*d.z-b.z*d.x)) * inv_det;
    r[1][3] =  (a.x*(b.z*c.w-b.w*c.z) - a.z*(b.x*c.w-b.w*c.x) + a.w*(b.x*c.z-b.z*c.x)) * inv_det;
    r[2][0] =  (b.x*(c.y*d.w-c.w*d.y) - b.y*(c.x*d.w-c.w*d.x) + b.w*(c.x*d.y-c.y*d.x)) * inv_det;
    r[2][1] = -(a.x*(c.y*d.w-c.w*d.y) - a.y*(c.x*d.w-c.w*d.x) + a.w*(c.x*d.y-c.y*d.x)) * inv_det;
    r[2][2] =  (a.x*(b.y*d.w-b.w*d.y) - a.y*(b.x*d.w-b.w*d.x) + a.w*(b.x*d.y-b.y*d.x)) * inv_det;
    r[2][3] = -(a.x*(b.y*c.w-b.w*c.y) - a.y*(b.x*c.w-b.w*c.x) + a.w*(b.x*c.y-b.y*c.x)) * inv_det;
    r[3][0] = -(b.x*(c.y*d.z-c.z*d.y) - b.y*(c.x*d.z-c.z*d.x) + b.z*(c.x*d.y-c.y*d.x)) * inv_det;
    r[3][1] =  (a.x*(c.y*d.z-c.z*d.y) - a.y*(c.x*d.z-c.z*d.x) + a.z*(c.x*d.y-c.y*d.x)) * inv_det;
    r[3][2] = -(a.x*(b.y*d.z-b.z*d.y) - a.y*(b.x*d.z-b.z*d.x) + a.z*(b.x*d.y-b.y*d.x)) * inv_det;
    r[3][3] =  (a.x*(b.y*c.z-b.z*c.y) - a.y*(b.x*c.z-b.z*c.x) + a.z*(b.x*c.y-b.y*c.x)) * inv_det;
    return r;
}

@fragment
fn fs_main(in: VertexOut) -> @location(0) vec4<f32> {
    // Reconstruct world-space ray from gl_FragCoord + globals.
    let pixel = in.clip_pos.xy;
    let ndc = vec2<f32>(
        (pixel.x / globals.viewport_size.x) * 2.0 - 1.0,
        1.0 - (pixel.y / globals.viewport_size.y) * 2.0,
    );
    // Unproject NDC near/far into world space.
    // Reversed depth: the near plane is z = 1, the far plane z = 0.
    let inv_vp = inverse_mat4(globals.vp);
    let inv_vp_near = inv_vp * vec4<f32>(ndc, 1.0, 1.0);
    let inv_vp_far  = inv_vp * vec4<f32>(ndc, 0.0, 1.0);
    let world_near = inv_vp_near.xyz / inv_vp_near.w;
    let world_far  = inv_vp_far.xyz  / inv_vp_far.w;
    let ro_world = world_near;
    let rd_world = normalize(world_far - world_near);

    // Transform ray into local space and intersect against the unit cube.
    let ro_local = (vu.inv_model * vec4<f32>(ro_world, 1.0)).xyz;
    let rd_local = (vu.inv_model * vec4<f32>(rd_world, 0.0)).xyz;

    var t_near: f32 = 0.0;
    var t_far:  f32 = 0.0;
    if (!ray_unit_cube(ro_local, rd_local, &t_near, &t_far)) {
        discard;
    }

    var t = max(t_near, 0.0);
    // A cleared reverse-Z pixel has depth zero. For opaque geometry,
    // unproject its actual depth and measure along the same world-space ray.
    let depth = textureLoad(scene_depth, vec2<i32>(pixel), 0);
    var clipped = false;
    if depth > 0.0 {
        let hit = inv_vp * vec4<f32>(ndc, depth, 1.0);
        let t_opaque = dot(hit.xyz / hit.w - ro_world, rd_world);
        if t_opaque <= t {
            discard;
        }
        clipped = t_opaque < t_far;
        t_far = min(t_far, t_opaque);
    }
    var accum: vec4<f32> = vec4<f32>(0.0);
    var step_count: u32 = 0u;
    let inv_range = 1.0 / max(vu.vmax - vu.vmin, 1e-12);

    loop {
        if (step_count >= vu.max_steps) { break; }
        if (t > t_far)                  { break; }
        if (accum.a > 0.99)             { break; }

        let p_local = ro_local + t * rd_local;
        let p_uvw = p_local + vec3<f32>(0.5);
        let s = textureSampleLevel(volume_tex, vol_sampler, p_uvw, 0.0).r;
        let s_n = clamp((s - vu.vmin) * inv_range, 0.0, 1.0);

        // 1D LUTs encoded as 256x1 2D textures (wgpu portability).
        let rgb = textureSampleLevel(color_lut,   lut_sampler, vec2<f32>(s_n, 0.5), 0.0).rgb;
        // Integrate only the visible part of the last step at an opaque hit.
        let step = select(vu.step_size, min(vu.step_size, t_far - t), clipped);
        let a = textureSampleLevel(opacity_lut, lut_sampler, vec2<f32>(s_n, 0.5), 0.0).r
                * vu.density_scale * step;

        accum = accum + (1.0 - accum.a) * vec4<f32>(rgb * a, a);
        t = t + vu.step_size;
        step_count = step_count + 1u;
    }
    return accum;
}
