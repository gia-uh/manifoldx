"""Skybox render pass — draws the prefiltered environment map as the scene background."""

import wgpu

_SKYBOX_SHADER = """
struct Globals {
    vp:            mat4x4<f32>,
    view:          mat4x4<f32>,
    proj:          mat4x4<f32>,
    camera_pos:    vec3<f32>,
    _pad0:         f32,
    viewport_size: vec2<f32>,
    _pad1:         vec2<f32>,
    ibl_intensity: f32,
    ibl_enabled:   u32,
    _pad_ibl:      vec2<f32>,
};

@group(0) @binding(0) var<uniform> globals: Globals;
@group(1) @binding(0) var env_sampler:     sampler;
@group(1) @binding(2) var prefiltered_map: texture_cube<f32>;

struct VertexOutput {
    @builtin(position) pos: vec4<f32>,
    @location(0)       dir: vec3<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) vi: u32) -> VertexOutput {
    var positions = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>( 3.0, -1.0),
        vec2<f32>(-1.0,  3.0),
    );
    let p = positions[vi];

    // Unproject NDC → camera-space direction using focal lengths from proj matrix.
    // proj[0][0] = f/aspect, proj[1][1] = f  (WGSL column-major notation)
    let cam_dir = vec3<f32>(p.x / globals.proj[0][0], p.y / globals.proj[1][1], -1.0);

    // Camera → world: multiply by transpose of the view rotation submatrix (R^T).
    // WGSL view[i] is the i-th column of the uploaded matrix.  Because the renderer
    // uploads view.T, view[i].xyz equals the i-th column of the *original* view
    // rotation matrix R, and  (R^T · v)[i] = dot(col_i(R), v).
    let world_dir = vec3<f32>(
        dot(globals.view[0].xyz, cam_dir),
        dot(globals.view[1].xyz, cam_dir),
        dot(globals.view[2].xyz, cam_dir),
    );

    var out: VertexOutput;
    // z = 0 → depth 0.0, the far plane under reversed depth, so all scene geometry draws in front.
    out.pos = vec4<f32>(p, 0.0, 1.0);
    out.dir = world_dir;
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    // Sample the lowest-roughness mip (mip 0 = mirror reflection / raw environment).
    let color = textureSampleLevel(prefiltered_map, env_sampler, normalize(in.dir), 0.0).rgb
              * globals.ibl_intensity;
    // Reinhard tone map + sRGB gamma
    let mapped = color / (color + vec3<f32>(1.0));
    let srgb   = mapped;  // the sRGB render target encodes gamma
    return vec4<f32>(srgb, 1.0);
}
"""


def render_skybox(rp, engine, render_pass):
    """Render environment map as background before mesh geometry.

    Draws a fullscreen triangle at depth=1.0 (far plane) using the prefiltered
    environment cubemap so all scene geometry appears in front.
    Only called when engine.environment is not None and show_skybox is True.
    """
    env = engine._ready_environment()
    if env is None or not env.show_skybox:
        return

    if id(env) != rp._ibl_env_id:
        rp._upload_ibl_env(rp._device, env)

    if "skybox" not in rp._pipelines:
        bg0_layout = rp._device.create_bind_group_layout(entries=[
            {
                "binding": 0,
                "visibility": wgpu.ShaderStage.VERTEX | wgpu.ShaderStage.FRAGMENT,
                "buffer": {"type": wgpu.BufferBindingType.uniform},
            }
        ])
        pipeline_layout = rp._device.create_pipeline_layout(
            bind_group_layouts=[bg0_layout, rp._ibl_bind_group_layout]
        )
        shader = rp._device.create_shader_module(code=_SKYBOX_SHADER)
        pipeline = rp._device.create_render_pipeline(
            layout=pipeline_layout,
            vertex={"module": shader, "entry_point": "vs_main", "buffers": []},
            primitive={"topology": wgpu.PrimitiveTopology.triangle_list},
            depth_stencil={
                "format": wgpu.TextureFormat.depth32float,
                "depth_write_enabled": False,
                "depth_compare": wgpu.CompareFunction.greater_equal,
            },
            fragment={
                "module": shader,
                "entry_point": "fs_main",
                "targets": [{"format": engine._texture_format}],
            },
        )
        rp._pipelines["skybox"] = pipeline
        rp._skybox_bg0_layout = bg0_layout

    pipeline = rp._pipelines["skybox"]
    bg0 = rp._device.create_bind_group(
        layout=rp._skybox_bg0_layout,
        entries=[{
            "binding": 0,
            "resource": {"buffer": rp._globals_buffer, "offset": 0, "size": 240},
        }],
    )
    ibl_bg = rp._ibl_active_bind_group or rp._ibl_placeholder_bind_group

    render_pass.set_pipeline(pipeline)
    render_pass.set_bind_group(0, bg0)
    render_pass.set_bind_group(1, ibl_bg)
    render_pass.draw(3, 1, 0, 0)
