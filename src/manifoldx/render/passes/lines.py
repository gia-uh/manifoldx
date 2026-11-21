"""Lines pass: engine.lines (manifoldx/lines.py) as quads a fixed number of screen pixels wide.

Each segment is one instance of six vertices; the vertex shader projects both ends, widens the segment across
its screen direction by half its width (plus the feather) and past each end by as much, so consecutive segments
of a polyline overlap at their joins. Depth test on (reversed Z, greater-equal) against each line brought `bias`
of its distance toward the camera, depth write off, premultiplied alpha blending.
"""

import numpy as np
import wgpu

from manifoldx import counters

SHADER = """
struct Globals {
    vp: mat4x4<f32>, view: mat4x4<f32>, proj: mat4x4<f32>,
    camera_pos: vec3<f32>, _pad0: f32,
    viewport_size: vec2<f32>, _pad1: vec2<f32>,
};
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
    let half_vp = globals.viewport_size * 0.5;
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
"""


def _pipeline(rp, engine):
    key = ("lines", engine._texture_format)
    if key in rp._pipelines:
        return rp._pipelines[key]
    device = rp._device
    shader = device.create_shader_module(code=SHADER)
    vis = wgpu.ShaderStage.VERTEX | wgpu.ShaderStage.FRAGMENT
    bgl = device.create_bind_group_layout(entries=[
        {"binding": 0, "visibility": vis, "buffer": {"type": wgpu.BufferBindingType.uniform}},
        {"binding": 1, "visibility": wgpu.ShaderStage.VERTEX,
         "buffer": {"type": wgpu.BufferBindingType.read_only_storage}},
        {"binding": 2, "visibility": vis, "buffer": {"type": wgpu.BufferBindingType.uniform}},
    ])
    blend = {"src_factor": wgpu.BlendFactor.one, "dst_factor": wgpu.BlendFactor.one_minus_src_alpha,
             "operation": wgpu.BlendOperation.add}
    pipeline = device.create_render_pipeline(
        layout=device.create_pipeline_layout(bind_group_layouts=[bgl]),
        vertex={"module": shader, "entry_point": "vs_main", "buffers": []},
        fragment={"module": shader, "entry_point": "fs_main",
                  "targets": [{"format": engine._texture_format, "blend": {"color": blend, "alpha": blend},
                               "write_mask": wgpu.ColorWrite.ALL}]},
        primitive={"topology": wgpu.PrimitiveTopology.triangle_list, "cull_mode": wgpu.CullMode.none},
        depth_stencil={"format": wgpu.TextureFormat.depth32float, "depth_write_enabled": False,
                       "depth_compare": wgpu.CompareFunction.greater_equal},
        multisample={"count": 1, "mask": 0xFFFFFFFF, "alpha_to_coverage_enabled": False},
    )
    rp._pipelines[key] = (pipeline, bgl)
    return rp._pipelines[key]


def render_lines_pass(rp, engine, render_pass) -> None:
    """Draw engine.lines' layers, each in one instanced draw; a layer's buffer is uploaded when it changed."""
    lines = getattr(engine, "lines", None)
    if lines is None or not lines.visible or not lines.layers or rp._globals_buffer is None:
        return
    pipeline, bgl = _pipeline(rp, engine)
    device = rp._device
    gpu = rp.__dict__.setdefault("_line_layers", {})  # key -> (version, buffer, capacity, group, globals, uniform)
    for key in [k for k in gpu if k not in lines.layers]:
        old = gpu.pop(key)
        old[1].destroy()
        old[5].destroy()
    render_pass.set_pipeline(pipeline)
    for key, data in lines.layers.items():
        if not len(data):
            continue
        version = lines.versions.get(key, 0)
        have = gpu.get(key)
        if have is not None and have[4] is not rp._globals_buffer:  # the globals were made again: a new group
            have[1].destroy()
            have[5].destroy()
            gpu.pop(key)
            have = None
        if have is None or have[0] != version:
            if have is None or have[2] < data.nbytes:
                if have is not None:
                    have[1].destroy()
                    have[5].destroy()
                size = max(data.nbytes, 48)
                buf = device.create_buffer(size=size, usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST)
                uni = device.create_buffer(size=16, usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST)
                group = device.create_bind_group(layout=bgl, entries=[
                    {"binding": 0, "resource": {"buffer": rp._globals_buffer, "offset": 0, "size": 240}},
                    {"binding": 1, "resource": {"buffer": buf, "offset": 0, "size": size}},
                    {"binding": 2, "resource": {"buffer": uni, "offset": 0, "size": 16}},
                ])
                have = (version, buf, size, group, rp._globals_buffer, uni)
            else:
                have = (version, *have[1:])
            counters.write_buffer(device.queue, have[1], 0, data.tobytes())
            gpu[key] = have
        scale, alpha = lines.looks.get(key, (1.0, 1.0))
        counters.write_buffer(device.queue, have[5], 0,
                              np.array([lines.bias, lines.feather, scale, alpha], np.float32).tobytes())
        render_pass.set_bind_group(0, have[3], [], 0, 0)
        counters.add("draws")
        render_pass.draw(6, len(data), 0, 0)
