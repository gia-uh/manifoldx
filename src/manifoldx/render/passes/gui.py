"""GUI render pass — runs last in the pipeline.

Draws in the widget tree's paint order (a later panel covers an earlier
one's images and text); each run of same-kind ops is one draw:

1. Rects — one instanced draw via `RectMaterial`.
2. Images — one draw per Image, each from its own texture: the SVG rasterized
   at the pixel size it lands at on the render target (see `_prepare_images`).
3. Glyphs — one instanced draw via the local `_GLYPH_WGSL` shader, which
   reads from LabelTextureAtlas (a texture_2d_array) and a per-instance
   storage buffer encoding screen-space billboard coords + atlas slice.

Pass attributes (set at pipeline-creation time in `_ensure_gui_pipeline`):
- depth-test off, depth-write off
- alpha blend on (pre-multiplied alpha)
- no culling
"""

from __future__ import annotations

import functools
import math
from dataclasses import replace

import numpy as np
import wgpu

from manifoldx.gui.layout import LayoutBox, anchored_slot, compute_layout
from manifoldx.gui.material import RectMaterial
from manifoldx.gui.painter import Painter, paint
from manifoldx.gui.widgets import Panel, _measure_text


# ---------------------------------------------------------------------------
# Glyph shader — screen-anchored billboard per TextOp.
#
# Instance layout (3 × vec4<f32> = 48 bytes):
#   row0 = (xy_top_left_px.x, xy_top_left_px.y, size_px.w, size_px.h)
#   row1 = (slice_idx, fg.r, fg.g, fg.b)
#   row2 = (u0, v0, u1, v1)  — normalized UV bounds of the live glyph region
#
# row2 clamps shader sampling to the actual rasterized pixels inside the tile
# instead of stretching the whole 256×64 tile into the layout box — this is
# what makes text legible at small widget sizes.
#
# The Globals struct is the same 224-byte block used by the rect shader;
# only viewport_size (bytes 208-215) is read here.
# ---------------------------------------------------------------------------
_GLYPH_WGSL = """
struct Globals {
    vp:           mat4x4<f32>,
    view:         mat4x4<f32>,
    proj:         mat4x4<f32>,
    camera_pos:   vec3<f32>,
    _pad0:        f32,
    viewport_size: vec2<f32>,
    _pad1:        vec2<f32>,
};
@group(0) @binding(0) var<uniform> globals: Globals;

struct GlyphInstance {
    // Three packed vec4<f32> rows (48 bytes) — GPU-friendly alignment.
    // row0: (xy_top_left_px.x, xy_top_left_px.y, size_px.x, size_px.y)
    // row1: (slice_idx, fg.r, fg.g, fg.b)
    // row2: (u0, v0, u1, v1) — normalized UV of the live glyph bbox in the tile
    row0: vec4<f32>,
    row1: vec4<f32>,
    row2: vec4<f32>,
};
@group(0) @binding(1) var<storage, read> glyphs: array<GlyphInstance>;
@group(0) @binding(2) var glyph_atlas: texture_2d_array<f32>;
@group(0) @binding(3) var glyph_sampler: sampler;

struct VsIn {
    @location(0) position: vec3<f32>,         // unit-quad corner [-0.5, 0.5]
    @builtin(instance_index) instance_id: u32,
};

struct VsOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) uv:    vec2<f32>,
    @location(1) slice: f32,
    @location(2) fg:    vec3<f32>,
};

@vertex
fn vs_main(in: VsIn) -> VsOut {
    let inst       = glyphs[in.instance_id];
    let xy         = vec2<f32>(inst.row0.x, inst.row0.y);
    let sz         = vec2<f32>(inst.row0.z, inst.row0.w);
    let slice      = inst.row1.x;
    let fg         = vec3<f32>(inst.row1.y, inst.row1.z, inst.row1.w);
    let uv0        = vec2<f32>(inst.row2.x, inst.row2.y);
    let uv1        = vec2<f32>(inst.row2.z, inst.row2.w);

    let center_px  = xy + sz * 0.5;
    let corner_px  = center_px + vec2<f32>(in.position.x, in.position.y) * sz;
    let ndc_x      = (corner_px.x / globals.viewport_size.x) * 2.0 - 1.0;
    let ndc_y      = 1.0 - (corner_px.y / globals.viewport_size.y) * 2.0;

    // Map quad-local [-0.5, 0.5]^2 into UV [uv0, uv1]; flip V so PIL
    // top-left aligns with quad top-left.
    // Pixel-space Y here matches the PIL tile convention (origin top-left),
    // unlike LabelMaterial which uses an NDC-Y-up quad. So both U and V map
    // straight from quad-local [-0.5, 0.5] → UV [0, 1] with NO flip.
    let uv_t = vec2<f32>(in.position.x + 0.5, in.position.y + 0.5);

    var out: VsOut;
    out.clip  = vec4<f32>(ndc_x, ndc_y, 0.0, 1.0);
    out.uv    = mix(uv0, uv1, uv_t);
    out.slice = slice;
    out.fg    = fg;
    return out;
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    let layer = i32(in.slice + 0.5);
    let texel = textureSample(glyph_atlas, glyph_sampler, in.uv, layer);
    // Tint by fg colour; output is pre-multiplied alpha.
    let a = texel.a;
    return vec4<f32>(in.fg * a, a);
}
"""


# ---------------------------------------------------------------------------
# Image shader — one screen-space quad per ImageOp, sampled from that image's
# own texture (group 1). Instance layout (3 x vec4<f32> = 48 bytes):
#   row0 = (xy_top_left_px.x, xy_top_left_px.y, size_px.w, size_px.h)
#   row1 = tint rgba (alpha already times opacity)
#   row2 = (corner radius px, 0, 0, 0)
#   row3 = (u0, v0, u1, v1): the source rect drawn into the quad
# The texture holds sRGB colour with straight alpha; the output is
# pre-multiplied, blended like the glyphs, and the tint is applied like a
# glyph's fg.
# ---------------------------------------------------------------------------
_IMAGE_WGSL = """
struct Globals {
    vp:           mat4x4<f32>,
    view:         mat4x4<f32>,
    proj:         mat4x4<f32>,
    camera_pos:   vec3<f32>,
    _pad0:        f32,
    viewport_size: vec2<f32>,
    _pad1:        vec2<f32>,
};
@group(0) @binding(0) var<uniform> globals: Globals;

struct ImageInstance {
    row0: vec4<f32>,
    row1: vec4<f32>,
    row2: vec4<f32>,
    row3: vec4<f32>,
};
@group(0) @binding(1) var<storage, read> images: array<ImageInstance>;
@group(1) @binding(0) var image_tex: texture_2d<f32>;
@group(1) @binding(1) var image_sampler: sampler;

struct VsIn {
    @location(0) position: vec3<f32>,
    @builtin(instance_index) instance_id: u32,
};

struct VsOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) uv:   vec2<f32>,
    @location(1) tint: vec4<f32>,
    @location(2) size: vec2<f32>,
    @location(3) radius: f32,
    @location(4) local: vec2<f32>,
};

@vertex
fn vs_main(in: VsIn) -> VsOut {
    let inst      = images[in.instance_id];
    let xy        = inst.row0.xy;
    let sz        = inst.row0.zw;
    let corner_px = xy + sz * 0.5 + in.position.xy * sz;
    var out: VsOut;
    out.clip = vec4<f32>(
        (corner_px.x / globals.viewport_size.x) * 2.0 - 1.0,
        1.0 - (corner_px.y / globals.viewport_size.y) * 2.0,
        0.0, 1.0,
    );
    out.local  = in.position.xy + vec2<f32>(0.5, 0.5);
    out.uv     = mix(inst.row3.xy, inst.row3.zw, out.local);
    out.tint   = inst.row1;
    out.size   = sz;
    out.radius = inst.row2.x;
    return out;
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    let texel = textureSample(image_tex, image_sampler, in.uv);
    // Rounded corners: signed distance to the rounded box, one pixel of antialias.
    let half = in.size * 0.5;
    let r = min(in.radius, min(half.x, half.y));
    let q = abs(in.local * in.size - half) - (half - vec2<f32>(r, r));
    let d = length(max(q, vec2<f32>(0.0, 0.0))) + min(max(q.x, q.y), 0.0) - r;
    let cover = select(1.0, clamp(0.5 - d / max(fwidth(d), 1e-4), 0.0, 1.0), r > 0.0);
    let a = texel.a * in.tint.a * cover;
    return vec4<f32>(texel.rgb * in.tint.rgb * a, a);
}
"""

# Bytes of image textures the GUI pass keeps on the GPU before it destroys the
# least recently drawn.
IMAGE_TEXTURE_BUDGET = 64 * 1024 * 1024


# Unit-quad geometry shared by all rect instances (positions in [-0.5, 0.5]).
_UNIT_QUAD_VERTS = np.array(
    [
        [-0.5, -0.5, 0.0],
        [+0.5, -0.5, 0.0],
        [+0.5, +0.5, 0.0],
        [-0.5, +0.5, 0.0],
    ],
    dtype=np.float32,
)
_UNIT_QUAD_INDICES = np.array([0, 1, 2, 0, 2, 3], dtype=np.uint32)


def _refresh_value_displays(widget) -> None:
    from manifoldx.gui.value_display import ValueDisplay
    if isinstance(widget, ValueDisplay):
        widget.refresh()
    for child in getattr(widget, "children", []) or []:
        _refresh_value_displays(child)


def render_gui_pass(rp, engine, render_pass) -> None:
    """Entry point — called from RenderPipeline.render after the axis pass."""
    if not engine.gui:
        return

    # Drive dynamic text widgets once per frame.
    for panel in engine.gui:
        _refresh_value_displays(panel)

    painter = layout_gui(engine)

    _ensure_gui_pipeline(rp, engine)

    # Guarantee viewport_size is populated even when the entity pass short-
    # circuited early (no alive entities).  We write only the two floats we
    # need (bytes 208-215); the rest of the globals struct is irrelevant for
    # the GUI shader, which only reads viewport_size from it.
    _ensure_globals_viewport(rp, engine)

    # Upload every kind's instances once, then draw in the tree's paint order:
    # each run of same-kind ops is one draw over its slice of the instances.
    rect_data = RectMaterial.pack_instances(
        painter.rect_ops, exact=getattr(engine.gui, "exact_colors", True)
    )
    if rect_data.size > 0:
        _upload_rect_instances(rp, rect_data)
    if painter.image_ops:
        _ensure_image_pipeline(rp, engine)
        image_groups = _prepare_images(rp, engine, painter.image_ops)
    if painter.text_ops:
        atlas = engine.get_label_atlas()
        glyph_data = _pack_glyph_instances(painter.text_ops, atlas)
        atlas.upload_dirty(rp._device, rp._device.queue)
        _ensure_glyph_pipeline(rp, engine)
        _upload_glyph_instances(rp, glyph_data)

    for kind, first, count in paint_runs(painter.order):
        if kind == "rect":
            _draw_rects(rp, engine, render_pass, first, count)
        elif kind == "image":
            _draw_images(rp, render_pass, image_groups, first, count)
        else:
            _draw_glyphs(rp, engine, render_pass, first, count)


def paint_runs(order: list[tuple[str, int]]) -> list[tuple[str, int, int]]:
    """Group the paint order into (kind, first index, count) runs of one kind."""
    runs: list[list] = []
    for kind, index in order:
        if runs and runs[-1][0] == kind:
            runs[-1][2] += 1
        else:
            runs.append([kind, index, 1])
    return [tuple(r) for r in runs]


def layout_gui(engine) -> Painter:
    """Lay out and paint every root panel against the engine's viewport.

    Pure Python (no GPU): returns the painter holding the frame's rect and
    text ops in screen pixels. Layout runs in UI pixels against a viewport
    of (w / scale, h / scale); the ops are then scaled to the screen. The
    scale is applied here rather than through the globals' viewport_size,
    which the sprite and label passes share within the frame.
    """
    scale = engine.gui.effective_scale(engine.h)
    viewport = LayoutBox(0.0, 0.0, engine.w / scale, engine.h / scale)
    painter = Painter()
    for panel in engine.gui:
        spec = panel.build_layout_spec()
        slot = anchored_slot(panel.anchor, panel.offset, spec, viewport)
        boxes = compute_layout(spec, viewport=slot)
        paint(panel, spec, boxes, painter)
    if scale != 1.0:
        _scale_ops(painter, scale)
    if getattr(engine.gui, "exact_colors", True):
        _decode_colors(painter)
    return painter


def _linear(c: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    return _linear_cached(tuple(float(v) for v in c))


@functools.lru_cache(maxsize=4096)
def _linear_cached(c: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    """A GUI uses a few colours on many widgets every frame: decode each once."""
    from manifoldx.resources import srgb_to_linear

    r, g, b = (float(v) for v in srgb_to_linear(c[:3]))
    return (r, g, b, c[3])


def _decode_colors(painter: Painter) -> None:
    """Decode every op colour from sRGB to the linear light the shaders write,
    so the sRGB target shows each hex as itself. Alpha stays as it is."""
    painter.rect_ops[:] = [
        replace(op, fill=_linear(op.fill), border_color=_linear(op.border_color))
        for op in painter.rect_ops
    ]
    painter.text_ops[:] = [replace(op, fg=_linear(op.fg)) for op in painter.text_ops]
    painter.image_ops[:] = [replace(op, tint=_linear(op.tint)) for op in painter.image_ops]


def _scale_box(box: LayoutBox, s: float) -> LayoutBox:
    return LayoutBox(box.x * s, box.y * s, box.w * s, box.h * s)


def _scale_ops(painter: Painter, s: float) -> None:
    """Map UI-pixel ops to screen pixels. Text is re-measured at the scaled
    font size, so glyphs rasterize at their on-screen size instead of being
    stretched. A string that would not fit one atlas tile at that size keeps
    its UI font size and is stretched instead, so it is not cut off."""
    from manifoldx.viz.text import TILE_HEIGHT, TILE_WIDTH

    painter.rect_ops[:] = [
        replace(op, box=_scale_box(op.box, s), border=op.border * s, radius=op.radius * s)
        for op in painter.rect_ops
    ]
    scaled_text = []
    for op in painter.text_ops:
        font_size = max(1, round(op.font_size * s))
        w, h = _measure_text(op.text, font_size)
        # the atlas rasterizes at x=4 inside a TILE_WIDTH x TILE_HEIGHT tile
        if w + 4 <= TILE_WIDTH and h <= TILE_HEIGHT:
            box = LayoutBox(op.box.x * s, op.box.y * s, w, h)
            scaled_text.append(replace(op, box=box, font_size=font_size))
        else:
            scaled_text.append(replace(op, box=_scale_box(op.box, s)))
    painter.text_ops[:] = scaled_text
    painter.image_ops[:] = [
        replace(op, box=_scale_box(op.box, s), radius=op.radius * s) for op in painter.image_ops
    ]


def target_density(engine) -> float:
    """Render-target pixels per window pixel: the display's pixel ratio, or a
    still's supersampling factor. 1 before the first frame."""
    th = getattr(engine, "_depth_texture_size", (0, 0))[1]
    return th / engine.h if th and engine.h else 1.0


def image_pixel_box(box: LayoutBox, density: float) -> tuple[LayoutBox, int, int]:
    """Where an image lands and the pixel size to rasterize it at: the box
    snapped to whole render-target pixels, so texels map 1:1 to the screen."""
    pw = max(1, round(box.w * density))
    ph = max(1, round(box.h * density))
    x = round(box.x * density) / density
    y = round(box.y * density) / density
    return LayoutBox(x, y, pw / density, ph / density), pw, ph


def _ensure_globals_viewport(rp, engine) -> None:
    """Write viewport_size into the globals uniform buffer.

    The entity render passes upload the full 224-byte globals struct, which
    includes viewport_size at offset 208.  When there are no alive entities
    those passes short-circuit before the upload, leaving stale bytes in the
    buffer.  The GUI shader only reads viewport_size, so we write just those
    8 bytes unconditionally to keep the rect NDC conversion correct.
    """
    viewport_size = np.array([float(engine.w), float(engine.h)], dtype=np.float32)
    rp._device.queue.write_buffer(rp._globals_buffer, 208, viewport_size.tobytes())


def _anchored(panel: Panel, viewport: LayoutBox) -> LayoutBox:
    """The panel's root slot from its anchor, offset and size (gui.layout.anchored_slot).
    The hit test places panels through the same function, so paint and hit agree."""
    return anchored_slot(panel.anchor, panel.offset, panel.build_layout_spec(), viewport)


def _ensure_gui_pipeline(rp, engine) -> None:
    if getattr(rp, "_gui_pipeline", None) is not None:
        return
    device = rp._device
    texture_format = engine._texture_format
    shader = device.create_shader_module(code=RectMaterial._compile())

    # Bind group: 0 = globals uniform, 1 = instance storage buffer.
    bgl = device.create_bind_group_layout(
        entries=[
            {
                "binding": 0,
                "visibility": wgpu.ShaderStage.VERTEX | wgpu.ShaderStage.FRAGMENT,
                "buffer": {"type": wgpu.BufferBindingType.uniform},
            },
            {
                "binding": 1,
                "visibility": wgpu.ShaderStage.VERTEX | wgpu.ShaderStage.FRAGMENT,
                "buffer": {"type": wgpu.BufferBindingType.read_only_storage},
            },
        ]
    )
    layout = device.create_pipeline_layout(bind_group_layouts=[bgl])

    pipeline = device.create_render_pipeline(
        layout=layout,
        vertex={
            "module": shader,
            "entry_point": "vs_main",
            "buffers": [
                {
                    "array_stride": 12,
                    "step_mode": wgpu.VertexStepMode.vertex,
                    "attributes": [
                        {
                            "shader_location": 0,
                            "offset": 0,
                            "format": wgpu.VertexFormat.float32x3,
                        },
                    ],
                },
            ],
        },
        fragment={
            "module": shader,
            "entry_point": "fs_main",
            "targets": [
                {
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
                }
            ],
        },
        primitive={
            "topology": wgpu.PrimitiveTopology.triangle_list,
            "cull_mode": wgpu.CullMode.none,
        },
        # Depth format must match the render pass attachment even when
        # depth-test is logically off. We declare always-pass + no write.
        depth_stencil={
            "format": wgpu.TextureFormat.depth24plus,
            "depth_write_enabled": False,
            "depth_compare": wgpu.CompareFunction.always,
        },
        multisample={
            "count": 1,
            "mask": 0xFFFFFFFF,
            "alpha_to_coverage_enabled": False,
        },
    )
    rp._gui_pipeline = pipeline
    rp._gui_bgl = bgl

    # Unit-quad vertex + index buffers.
    rp._gui_quad_vbuf = device.create_buffer_with_data(
        data=_UNIT_QUAD_VERTS.tobytes(),
        usage=wgpu.BufferUsage.VERTEX,
    )
    rp._gui_quad_ibuf = device.create_buffer_with_data(
        data=_UNIT_QUAD_INDICES.tobytes(),
        usage=wgpu.BufferUsage.INDEX,
    )
    rp._gui_instance_buf = None
    rp._gui_instance_capacity = 0


def _upload_rect_instances(rp, data: np.ndarray) -> None:
    needed = data.nbytes
    if rp._gui_instance_buf is None or needed > rp._gui_instance_capacity:
        rp._gui_instance_buf = rp._device.create_buffer(
            size=max(needed, 4096),
            usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST,
        )
        rp._gui_instance_capacity = max(needed, 4096)
    rp._device.queue.write_buffer(rp._gui_instance_buf, 0, data.tobytes())


def _bind_group(rp, key, make):
    """The bind group for `key`, made once. Keys hold the GPU objects they bind, compared by
    identity, so a grown buffer or a new atlas texture makes a new group. Creating one per draw
    cost a game ~20 ms a frame with 80 labels on screen."""
    cache = rp.__dict__.setdefault("_gui_bind_groups", {})
    group = cache.get(key)
    if group is None:
        if len(cache) >= 16:  # stale buffers and atlases: let them go
            cache.clear()
        group = cache[key] = make()
    return group


def _draw_rects(rp, engine, render_pass, first: int, instance_count: int) -> None:
    key = ("rects", rp._globals_buffer, rp._gui_instance_buf, rp._gui_instance_capacity)
    bind_group = _bind_group(rp, key, lambda: rp._device.create_bind_group(
        layout=rp._gui_bgl,
        entries=[
            {
                "binding": 0,
                "resource": {
                    "buffer": rp._globals_buffer,
                    "offset": 0,
                    "size": 240,
                },
            },
            {
                "binding": 1,
                "resource": {
                    "buffer": rp._gui_instance_buf,
                    "offset": 0,
                    "size": rp._gui_instance_capacity,
                },
            },
        ],
    ))
    render_pass.set_pipeline(rp._gui_pipeline)
    render_pass.set_bind_group(0, bind_group, [], 0, 0)
    render_pass.set_vertex_buffer(0, rp._gui_quad_vbuf)
    render_pass.set_index_buffer(rp._gui_quad_ibuf, wgpu.IndexFormat.uint32)
    render_pass.draw_indexed(6, instance_count, 0, 0, first)


# ---------------------------------------------------------------------------
# Glyph pipeline helpers
# ---------------------------------------------------------------------------

def _pack_glyph_instances(text_ops, atlas) -> np.ndarray:
    """Pack TextOps into a (N, 12) float32 buffer with the layout the GUI
    glyph shader expects:

        row0 = (xy.x, xy.y, size.x, size.y)
        row1 = (slice, fg.r, fg.g, fg.b)
        row2 = (u0, v0, u1, v1)  — UV bounds of the live glyph in the tile

    Coordinates are pixel-space top-left (matching the rect path).
    """
    if not text_ops:
        return np.zeros((0, 12), dtype=np.float32)
    rows = []
    for op in text_ops:
        slice_idx = atlas.get_or_create(op.text, font_size=op.font_size)
        _w, _h, u0, v0, u1, v1 = atlas.measure_string(op.text, font_size=op.font_size)
        rows.append([
            op.box.x, op.box.y, op.box.w, op.box.h,
            float(slice_idx), op.fg[0], op.fg[1], op.fg[2],
            u0, v0, u1, v1,
        ])
    return np.asarray(rows, dtype=np.float32)


def _ensure_glyph_pipeline(rp, engine) -> None:
    if getattr(rp, "_gui_glyph_pipeline", None) is not None:
        return
    device = rp._device
    texture_format = engine._texture_format
    shader = device.create_shader_module(code=_GLYPH_WGSL)

    # 4-binding layout: globals uniform + instance storage + atlas texture + sampler.
    bgl = device.create_bind_group_layout(
        entries=[
            {
                "binding": 0,
                "visibility": wgpu.ShaderStage.VERTEX | wgpu.ShaderStage.FRAGMENT,
                "buffer": {"type": wgpu.BufferBindingType.uniform},
            },
            {
                "binding": 1,
                "visibility": wgpu.ShaderStage.VERTEX,
                "buffer": {"type": wgpu.BufferBindingType.read_only_storage},
            },
            {
                "binding": 2,
                "visibility": wgpu.ShaderStage.FRAGMENT,
                "texture": {
                    "sample_type": wgpu.TextureSampleType.float,
                    "view_dimension": wgpu.TextureViewDimension.d2_array,
                    "multisampled": False,
                },
            },
            {
                "binding": 3,
                "visibility": wgpu.ShaderStage.FRAGMENT,
                "sampler": {"type": wgpu.SamplerBindingType.filtering},
            },
        ]
    )
    layout = device.create_pipeline_layout(bind_group_layouts=[bgl])

    pipeline = device.create_render_pipeline(
        layout=layout,
        vertex={
            "module": shader,
            "entry_point": "vs_main",
            "buffers": [
                {
                    "array_stride": 12,
                    "step_mode": wgpu.VertexStepMode.vertex,
                    "attributes": [
                        {
                            "shader_location": 0,
                            "offset": 0,
                            "format": wgpu.VertexFormat.float32x3,
                        },
                    ],
                },
            ],
        },
        fragment={
            "module": shader,
            "entry_point": "fs_main",
            "targets": [
                {
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
                }
            ],
        },
        primitive={
            "topology": wgpu.PrimitiveTopology.triangle_list,
            "cull_mode": wgpu.CullMode.none,
        },
        # Depth format must match the render pass attachment even when
        # depth-test is logically off. Same state as the rect pipeline.
        depth_stencil={
            "format": wgpu.TextureFormat.depth24plus,
            "depth_write_enabled": False,
            "depth_compare": wgpu.CompareFunction.always,
        },
        multisample={
            "count": 1,
            "mask": 0xFFFFFFFF,
            "alpha_to_coverage_enabled": False,
        },
    )
    rp._gui_glyph_pipeline = pipeline
    rp._gui_glyph_bgl = bgl
    rp._gui_glyph_instance_buf = None
    rp._gui_glyph_instance_capacity = 0


def _upload_glyph_instances(rp, data: np.ndarray) -> None:
    needed = data.nbytes
    if (
        rp._gui_glyph_instance_buf is None
        or needed > rp._gui_glyph_instance_capacity
    ):
        rp._gui_glyph_instance_buf = rp._device.create_buffer(
            size=max(needed, 4096),
            usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST,
        )
        rp._gui_glyph_instance_capacity = max(needed, 4096)
    rp._device.queue.write_buffer(rp._gui_glyph_instance_buf, 0, data.tobytes())


def _draw_glyphs(rp, engine, render_pass, first: int, instance_count: int) -> None:
    atlas = engine.get_label_atlas()
    key = ("glyphs", rp._globals_buffer, rp._gui_glyph_instance_buf, rp._gui_glyph_instance_capacity,
           atlas.gpu_texture, atlas.gpu_sampler)
    bind_group = _bind_group(rp, key, lambda: rp._device.create_bind_group(
        layout=rp._gui_glyph_bgl,
        entries=[
            {
                "binding": 0,
                "resource": {
                    "buffer": rp._globals_buffer,
                    "offset": 0,
                    "size": 240,
                },
            },
            {
                "binding": 1,
                "resource": {
                    "buffer": rp._gui_glyph_instance_buf,
                    "offset": 0,
                    "size": rp._gui_glyph_instance_capacity,
                },
            },
            {
                "binding": 2,
                # Must explicitly request a 2D-array view; create_view() defaults to D2.
                "resource": atlas.gpu_texture.create_view(dimension=wgpu.TextureViewDimension.d2_array),
            },
            {
                "binding": 3,
                "resource": atlas.gpu_sampler,
            },
        ],
    ))
    render_pass.set_pipeline(rp._gui_glyph_pipeline)
    render_pass.set_bind_group(0, bind_group, [], 0, 0)
    render_pass.set_vertex_buffer(0, rp._gui_quad_vbuf)
    render_pass.set_index_buffer(rp._gui_quad_ibuf, wgpu.IndexFormat.uint32)
    render_pass.draw_indexed(6, instance_count, 0, 0, first)


# ---------------------------------------------------------------------------
# Image pipeline helpers
# ---------------------------------------------------------------------------

def _ensure_image_pipeline(rp, engine) -> None:
    if getattr(rp, "_gui_image_pipeline", None) is not None:
        return
    from manifoldx.svg import LruCache

    device = rp._device
    shader = device.create_shader_module(code=_IMAGE_WGSL)
    bgl0 = device.create_bind_group_layout(
        entries=[
            {
                "binding": 0,
                "visibility": wgpu.ShaderStage.VERTEX,
                "buffer": {"type": wgpu.BufferBindingType.uniform},
            },
            {
                "binding": 1,
                "visibility": wgpu.ShaderStage.VERTEX,
                "buffer": {"type": wgpu.BufferBindingType.read_only_storage},
            },
        ]
    )
    bgl1 = device.create_bind_group_layout(
        entries=[
            {
                "binding": 0,
                "visibility": wgpu.ShaderStage.FRAGMENT,
                "texture": {
                    "sample_type": wgpu.TextureSampleType.float,
                    "view_dimension": wgpu.TextureViewDimension.d2,
                    "multisampled": False,
                },
            },
            {
                "binding": 1,
                "visibility": wgpu.ShaderStage.FRAGMENT,
                "sampler": {"type": wgpu.SamplerBindingType.filtering},
            },
        ]
    )
    blend = {
        "src_factor": wgpu.BlendFactor.one,
        "dst_factor": wgpu.BlendFactor.one_minus_src_alpha,
        "operation": wgpu.BlendOperation.add,
    }
    rp._gui_image_pipeline = device.create_render_pipeline(
        layout=device.create_pipeline_layout(bind_group_layouts=[bgl0, bgl1]),
        vertex={
            "module": shader,
            "entry_point": "vs_main",
            "buffers": [
                {
                    "array_stride": 12,
                    "step_mode": wgpu.VertexStepMode.vertex,
                    "attributes": [
                        {"shader_location": 0, "offset": 0, "format": wgpu.VertexFormat.float32x3},
                    ],
                },
            ],
        },
        fragment={
            "module": shader,
            "entry_point": "fs_main",
            "targets": [
                {
                    "format": engine._texture_format,
                    "blend": {"color": blend, "alpha": blend},
                    "write_mask": wgpu.ColorWrite.ALL,
                }
            ],
        },
        primitive={
            "topology": wgpu.PrimitiveTopology.triangle_list,
            "cull_mode": wgpu.CullMode.none,
        },
        depth_stencil={
            "format": wgpu.TextureFormat.depth24plus,
            "depth_write_enabled": False,
            "depth_compare": wgpu.CompareFunction.always,
        },
        multisample={"count": 1, "mask": 0xFFFFFFFF, "alpha_to_coverage_enabled": False},
    )
    rp._gui_image_bgl0 = bgl0
    rp._gui_image_bgl1 = bgl1
    rp._gui_image_sampler = device.create_sampler(
        address_mode_u=wgpu.AddressMode.clamp_to_edge,
        address_mode_v=wgpu.AddressMode.clamp_to_edge,
        mag_filter=wgpu.FilterMode.linear,
        min_filter=wgpu.FilterMode.linear,
    )
    rp._gui_image_instance_buf = None
    rp._gui_image_instance_capacity = 0
    rp._gui_image_last = {}  # id(Image) -> its pixel size the frame before
    rp._gui_image_external = {}  # TextureHandle.id -> (handle, bind group) for texture images
    # (digest, width px, height px) -> (texture, bind group). An evicted
    # texture may still be bound in this frame's pass: destroy it next frame.
    rp._gui_image_evicted = []
    rp._gui_image_textures = LruCache(
        IMAGE_TEXTURE_BUDGET, on_evict=lambda entry: rp._gui_image_evicted.append(entry[0])
    )


def _texture_bind_group(rp, handle):
    """The bind group of an existing texture (a render_to_texture view, say)."""
    entry = rp._gui_image_external.get(handle.id)
    if entry is None or entry[0] is not handle:
        group = rp._device.create_bind_group(
            layout=rp._gui_image_bgl1,
            entries=[
                # level 0 only: write_texture patches it, and the mips would lag
                {"binding": 0, "resource": handle.texture.create_view(
                    base_mip_level=0, mip_level_count=1)},
                {"binding": 1, "resource": rp._gui_image_sampler},
            ],
        )
        entry = rp._gui_image_external[handle.id] = (handle, group)
    return entry[1]


def _image_bind_group(rp, op, pw: int, ph: int):
    """The bind group of `op`'s raster at pw x ph, rasterized and uploaded on
    first use."""
    key = (op.key, pw, ph)
    entry = rp._gui_image_textures.get(key)
    if entry is None:
        rgba = op.raster(pw, ph)
        device = rp._device
        texture = device.create_texture(
            size=(pw, ph, 1),
            format=wgpu.TextureFormat.rgba8unorm_srgb,
            usage=wgpu.TextureUsage.TEXTURE_BINDING | wgpu.TextureUsage.COPY_DST,
        )
        device.queue.write_texture(
            {"texture": texture, "origin": (0, 0, 0)},
            rgba.tobytes(),
            {"bytes_per_row": pw * 4, "rows_per_image": ph},
            (pw, ph, 1),
        )
        bind_group = device.create_bind_group(
            layout=rp._gui_image_bgl1,
            entries=[
                {"binding": 0, "resource": texture.create_view()},
                {"binding": 1, "resource": rp._gui_image_sampler},
            ],
        )
        entry = (texture, bind_group)
        rp._gui_image_textures.put(key, entry, pw * ph * 4)
    return entry[1]


def bucket_size(pw: int, ph: int) -> tuple[int, int]:
    """The size to rasterize an image at while its size animates: the longest
    side rounded up to a step of 2^(1/8), so a pulse or a pop-in reuses a few
    rasters, each drawn at most 9% smaller than it was drawn."""
    m = max(pw, ph)
    k = 2.0 ** (math.ceil(math.log2(m) * 8 - 1e-9) / 8) / m
    return max(1, round(pw * k)), max(1, round(ph * k))


def settle_size(prev, pw: int, ph: int, have_exact: bool) -> tuple[int, int]:
    """Rasterize at the exact size unless the image's size changed since the
    last frame (`prev`, None on its first frame) and that size has no raster
    yet; then use `bucket_size`."""
    if have_exact or prev is None or prev == (pw, ph):
        return pw, ph
    return bucket_size(pw, ph)


def _prepare_images(rp, engine, image_ops) -> list:
    """Upload one instance row per ImageOp and return each op's texture bind
    group (None for an empty box)."""
    for texture in rp._gui_image_evicted:
        texture.destroy()
    rp._gui_image_evicted.clear()
    density = target_density(engine)
    last, rp._gui_image_last = rp._gui_image_last, {}
    rows, groups = [], []
    for op in image_ops:
        box, pw, ph = image_pixel_box(op.box, density)
        rows.append([box.x, box.y, box.w, box.h, *op.tint, op.radius, 0.0, 0.0, 0.0, *op.uv])
        if op.box.w <= 0 or op.box.h <= 0:
            groups.append(None)
            continue
        if op.texture is not None:
            groups.append(_texture_bind_group(rp, op.texture))
            continue
        rp._gui_image_last[op.widget] = (pw, ph)
        have_exact = (op.key, pw, ph) in rp._gui_image_textures
        rw, rh = settle_size(last.get(op.widget), pw, ph, have_exact)
        groups.append(_image_bind_group(rp, op, rw, rh))
    data = np.asarray(rows, dtype=np.float32)
    if rp._gui_image_instance_buf is None or data.nbytes > rp._gui_image_instance_capacity:
        rp._gui_image_instance_capacity = max(data.nbytes, 4096)
        rp._gui_image_instance_buf = rp._device.create_buffer(
            size=rp._gui_image_instance_capacity,
            usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST,
        )
    rp._device.queue.write_buffer(rp._gui_image_instance_buf, 0, data.tobytes())
    return groups


def _draw_images(rp, render_pass, groups: list, first: int, count: int) -> None:
    bind_group0 = rp._device.create_bind_group(
        layout=rp._gui_image_bgl0,
        entries=[
            {"binding": 0, "resource": {"buffer": rp._globals_buffer, "offset": 0, "size": 240}},
            {
                "binding": 1,
                "resource": {
                    "buffer": rp._gui_image_instance_buf,
                    "offset": 0,
                    "size": rp._gui_image_instance_capacity,
                },
            },
        ],
    )
    render_pass.set_pipeline(rp._gui_image_pipeline)
    render_pass.set_bind_group(0, bind_group0, [], 0, 0)
    render_pass.set_vertex_buffer(0, rp._gui_quad_vbuf)
    render_pass.set_index_buffer(rp._gui_quad_ibuf, wgpu.IndexFormat.uint32)
    for i in range(first, first + count):
        if groups[i] is not None:
            render_pass.set_bind_group(1, groups[i], [], 0, 0)
            render_pass.draw_indexed(6, 1, 0, 0, i)
