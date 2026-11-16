"""Volume DVR render pass.

Fullscreen-quad fragment shader, ray/box-AABB intersection in entity-local
space, fixed-step front-to-back composite. See design doc
``.knowledge/analysis/2026-05-08-volume-rendering-v1-design.md``.
"""

import numpy as np
import wgpu


VOLUME_SHADER_SOURCE = """
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
    out.clip_pos = vec4<f32>(pos[vid], 1.0, 1.0);  // the near plane under reversed depth: always passes
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
    let inv_vp_near = inverse_mat4(globals.vp) * vec4<f32>(ndc, 1.0, 1.0);
    let inv_vp_far  = inverse_mat4(globals.vp) * vec4<f32>(ndc, 0.0, 1.0);
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
        let a   = textureSampleLevel(opacity_lut, lut_sampler, vec2<f32>(s_n, 0.5), 0.0).r
                  * vu.density_scale * vu.step_size;

        accum = accum + (1.0 - accum.a) * vec4<f32>(rgb * a, a);
        t = t + vu.step_size;
        step_count = step_count + 1u;
    }
    return accum;
}
""".strip()


def get_or_create_volume_pipeline(rp, device, texture_format):
    """Return (pipeline, bgl_globals, bgl_volume) for the volume DVR pass.

    Cache key: ("volume",) — independent of mesh/sprite/label/axis caches.
    Pipeline state: depth-test LESS_EQUAL, depth-write OFF, alpha-blend ON.
    """
    cache_key = ("volume",)
    cached = rp._pipelines.get(cache_key)
    if cached is not None:
        return cached

    shader = device.create_shader_module(code=VOLUME_SHADER_SOURCE)

    bgl_globals = device.create_bind_group_layout(
        entries=[
            {
                "binding": 0,
                "visibility": wgpu.ShaderStage.FRAGMENT,
                "buffer": {"type": wgpu.BufferBindingType.uniform},
            },
        ],
    )

    # Volume tex + opacity LUT use R32Float, which is non-filterable on most
    # backends; pair them with non-filtering samplers (nearest-neighbor).
    # Color LUT uses RGBA8Unorm (filterable) but shares the non-filtering
    # sampler since both LUTs must come through one binding (per shader).
    bgl_volume = device.create_bind_group_layout(
        entries=[
            {
                "binding": 0,
                "visibility": wgpu.ShaderStage.FRAGMENT,
                "texture": {
                    "sample_type": wgpu.TextureSampleType.unfilterable_float,
                    "view_dimension": wgpu.TextureViewDimension.d3,
                    "multisampled": False,
                },
            },
            {
                "binding": 1,
                "visibility": wgpu.ShaderStage.FRAGMENT,
                "sampler": {"type": wgpu.SamplerBindingType.non_filtering},
            },
            {
                "binding": 2,
                "visibility": wgpu.ShaderStage.FRAGMENT,
                "texture": {
                    "sample_type": wgpu.TextureSampleType.unfilterable_float,
                    "view_dimension": wgpu.TextureViewDimension.d2,
                    "multisampled": False,
                },
            },
            {
                "binding": 3,
                "visibility": wgpu.ShaderStage.FRAGMENT,
                "texture": {
                    "sample_type": wgpu.TextureSampleType.unfilterable_float,
                    "view_dimension": wgpu.TextureViewDimension.d2,
                    "multisampled": False,
                },
            },
            {
                "binding": 4,
                "visibility": wgpu.ShaderStage.FRAGMENT,
                "sampler": {"type": wgpu.SamplerBindingType.non_filtering},
            },
            {
                "binding": 5,
                "visibility": wgpu.ShaderStage.FRAGMENT,
                "buffer": {
                    "type": wgpu.BufferBindingType.uniform,
                    "min_binding_size": 160,
                },
            },
        ],
    )

    layout = device.create_pipeline_layout(
        bind_group_layouts=[bgl_globals, bgl_volume]
    )

    pipeline = device.create_render_pipeline(
        layout=layout,
        vertex={"module": shader, "entry_point": "vs_main", "buffers": []},
        fragment={
            "module": shader,
            "entry_point": "fs_main",
            "targets": [{
                "format": texture_format,
                "blend": {
                    "color": {
                        "src_factor": wgpu.BlendFactor.one,
                        "dst_factor": wgpu.BlendFactor.one_minus_src_alpha,
                        "operation": wgpu.BlendOperation.add,
                    },
                    "alpha": {
                        "src_factor": wgpu.BlendFactor.one,
                        "dst_factor": wgpu.BlendFactor.one_minus_src_alpha,
                        "operation": wgpu.BlendOperation.add,
                    },
                },
                "write_mask": wgpu.ColorWrite.ALL,
            }],
        },
        primitive={
            "topology": wgpu.PrimitiveTopology.triangle_list,
            "cull_mode": wgpu.CullMode.none,
        },
        depth_stencil={
            "format": wgpu.TextureFormat.depth32float,
            "depth_write_enabled": False,
            "depth_compare": wgpu.CompareFunction.greater_equal,
        },
        multisample={"count": 1, "mask": 0xFFFFFFFF, "alpha_to_coverage_enabled": False},
    )
    rp._pipelines[cache_key] = (pipeline, bgl_globals, bgl_volume)
    return pipeline, bgl_globals, bgl_volume


def get_or_create_color_lut_view(rp, cmap_name: str):
    """Cache 256x1 RGBA8 colormap textures by name."""
    cache = getattr(rp, "_volume_color_lut_views", None)
    if cache is None:
        cache = {}
        rp._volume_color_lut_views = cache
    if cmap_name in cache:
        return cache[cmap_name]
    from manifoldx.viz.colormaps import get_colormap
    lut = get_colormap(cmap_name)   # (256, 4) uint8
    tex = rp._device.create_texture(
        size=(256, 1, 1),
        format=wgpu.TextureFormat.rgba8unorm,
        usage=wgpu.TextureUsage.TEXTURE_BINDING | wgpu.TextureUsage.COPY_DST,
        dimension=wgpu.TextureDimension.d2,
    )
    rp._device.queue.write_texture(
        {"texture": tex, "mip_level": 0, "origin": (0, 0, 0)},
        lut.tobytes(),
        {"offset": 0, "bytes_per_row": 256 * 4, "rows_per_image": 1},
        (256, 1, 1),
    )
    view = tex.create_view()
    cache[cmap_name] = view
    return view


def get_or_create_opacity_lut_view(rp, mat_id: int, lut: np.ndarray):
    """Cache 256x1 R32F opacity textures by material id."""
    cache = getattr(rp, "_volume_opacity_lut_views", None)
    if cache is None:
        cache = {}
        rp._volume_opacity_lut_views = cache
    if mat_id in cache:
        return cache[mat_id]
    tex = rp._device.create_texture(
        size=(256, 1, 1),
        format=wgpu.TextureFormat.r32float,
        usage=wgpu.TextureUsage.TEXTURE_BINDING | wgpu.TextureUsage.COPY_DST,
        dimension=wgpu.TextureDimension.d2,
    )
    rp._device.queue.write_texture(
        {"texture": tex, "mip_level": 0, "origin": (0, 0, 0)},
        lut.astype(np.float32).tobytes(),
        {"offset": 0, "bytes_per_row": 256 * 4, "rows_per_image": 1},
        (256, 1, 1),
    )
    view = tex.create_view()
    cache[mat_id] = view
    return view


def get_or_create_volume_uniform_buffer(rp, mat_id: int):
    """Per-material uniform buffer (160 bytes; rewritten each frame)."""
    cache = getattr(rp, "_volume_uniform_buffers", None)
    if cache is None:
        cache = {}
        rp._volume_uniform_buffers = cache
    if mat_id not in cache:
        cache[mat_id] = rp._device.create_buffer(
            size=160,
            usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST,
        )
    return cache[mat_id]


def write_volume_uniforms(rp, buf, entity_local_idx, mat_obj, vol_res, engine):
    """Pack and upload the VolumeUniforms struct (160 bytes)."""
    model = rp._transform_cache.get_transforms(
        rp._store, np.array([entity_local_idx], dtype=np.int64)
    )[0].reshape(4, 4)
    inv_model = np.linalg.inv(model)

    nz, ny, nx = vol_res.data.shape
    scale_diag = np.array([model[0, 0], model[1, 1], model[2, 2]], dtype=np.float32)
    voxel_dims = scale_diag / np.array([nx, ny, nz], dtype=np.float32)
    auto_step = float(np.min(np.abs(voxel_dims)) * 0.5)
    step_size = mat_obj.step_size if mat_obj.step_size is not None else auto_step

    packed = np.zeros(160 // 4, dtype=np.float32)
    # column-major mat4 (WGSL convention)
    packed[0:16]  = model.T.astype(np.float32).ravel()
    packed[16:32] = inv_model.T.astype(np.float32).ravel()
    packed[32]    = mat_obj.vmin
    packed[33]    = mat_obj.vmax
    packed[34]    = mat_obj.density_scale
    packed[35]    = step_size
    u32_view = packed.view(np.uint32)
    u32_view[36] = mat_obj.max_steps
    # slots 37, 38, 39 are pad (already zero).
    rp._device.queue.write_buffer(buf, 0, packed.tobytes())


def render_volume_pass(rp, engine, render_pass, volume_batches):
    """One draw call per volume entity. Each entity is a fullscreen triangle
    whose fragment shader raymarches the entity-local AABB.
    """
    if not volume_batches:
        return

    pipeline, bgl_globals, bgl_volume = get_or_create_volume_pipeline(
        rp, rp._device, engine._texture_format
    )

    if not hasattr(rp, "_volume_sampler") or rp._volume_sampler is None:
        # Nearest-filter sampler (required for R32Float in v1).
        rp._volume_sampler = rp._device.create_sampler(
            mag_filter=wgpu.FilterMode.nearest,
            min_filter=wgpu.FilterMode.nearest,
            address_mode_u=wgpu.AddressMode.clamp_to_edge,
            address_mode_v=wgpu.AddressMode.clamp_to_edge,
            address_mode_w=wgpu.AddressMode.clamp_to_edge,
        )

    globals_bg = rp._device.create_bind_group(
        layout=bgl_globals,
        entries=[
            {
                "binding": 0,
                "resource": {"buffer": rp._globals_buffer, "offset": 0, "size": 240},
            },
        ],
    )

    for entity_local_idx, mat_id, vol_id in volume_batches:
        mat_obj = engine._material_registry.get(mat_id)
        res = engine._volume_registry.get(vol_id)
        engine._volume_registry.upload_to_gpu(vol_id, rp._device.queue)
        volume_view = res.texture.create_view()

        color_view = get_or_create_color_lut_view(rp, mat_obj.cmap)
        opacity_view = get_or_create_opacity_lut_view(rp, mat_id, mat_obj.opacity_lut)

        mat_uniform_buf = get_or_create_volume_uniform_buffer(rp, mat_id)
        write_volume_uniforms(rp, mat_uniform_buf, entity_local_idx, mat_obj, res, engine)

        volume_bg = rp._device.create_bind_group(
            layout=bgl_volume,
            entries=[
                {"binding": 0, "resource": volume_view},
                {"binding": 1, "resource": rp._volume_sampler},
                {"binding": 2, "resource": color_view},
                {"binding": 3, "resource": opacity_view},
                {"binding": 4, "resource": rp._volume_sampler},
                {
                    "binding": 5,
                    "resource": {
                        "buffer": mat_uniform_buf, "offset": 0, "size": 160,
                    },
                },
            ],
        )

        render_pass.set_pipeline(pipeline)
        render_pass.set_bind_group(0, globals_bg, [], 0, 0)
        render_pass.set_bind_group(1, volume_bg, [], 0, 0)
        render_pass.draw(3, 1, 0, 0)   # fullscreen triangle
