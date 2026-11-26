@group(0) @binding(0) var<uniform> viewport: vec4<f32>;

struct RectInstance {
    xy:           vec2<f32>,
    size:         vec2<f32>,
    radius:       f32,
    border:       f32,
    exact:        f32,          // 1: colours blend by their alpha (gui.exact_colors)
    bg:           vec4<f32>,
    border_color: vec4<f32>,
};

@group(0) @binding(1) var<storage, read> instances: array<RectInstance>;

struct VsOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) local: vec2<f32>,   // pixel offset from rect center
    @location(1) @interpolate(flat) flat_id: u32,
};

@vertex
fn vs_main(@builtin(vertex_index) vertex: u32, @builtin(instance_index) instance: u32) -> VsOut {
    let corners = array<vec2<f32>, 6>(
        vec2<f32>(-0.5, -0.5), vec2<f32>(-0.5, 0.5), vec2<f32>(0.5, -0.5),
        vec2<f32>(0.5, -0.5), vec2<f32>(-0.5, 0.5), vec2<f32>(0.5, 0.5),
    );
    let corner = corners[vertex];
    let inst = instances[instance];
    // Pixel-space center + corner.
    let center_px = inst.xy + inst.size * 0.5;
    let corner_px = center_px + corner * inst.size;
    // Convert pixel coords -> NDC. Viewport origin is top-left; flip Y.
    let ndc_x = (corner_px.x / viewport.x) * 2.0 - 1.0;
    let ndc_y = 1.0 - (corner_px.y / viewport.y) * 2.0;
    var out: VsOut;
    out.clip = vec4<f32>(ndc_x, ndc_y, 0.0, 1.0);
    out.local = corner * inst.size;
    out.flat_id = instance;
    return out;
}

fn rounded_rect_sdf(p: vec2<f32>, half: vec2<f32>, r: f32) -> f32 {
    let q = abs(p) - half + vec2<f32>(r, r);
    return length(max(q, vec2<f32>(0.0, 0.0))) +
           min(max(q.x, q.y), 0.0) - r;
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    let inst = instances[in.flat_id];
    let half = inst.size * 0.5;
    let d = rounded_rect_sdf(in.local, half, inst.radius);
    // 1 px smoothstep gives cheap AA at the outer edge.
    let outer = 1.0 - smoothstep(-1.0, 0.0, d);
    if (inst.exact > 0.5) {
        // Premultiply fill and border by their own alpha, so a translucent
        // colour blends as itself, with or without a border.
        let bg = vec4<f32>(inst.bg.rgb * inst.bg.a, inst.bg.a);
        let bc = vec4<f32>(inst.border_color.rgb * inst.border_color.a, inst.border_color.a);
        if (inst.border > 0.0) {
            let inner = 1.0 - smoothstep(-1.0, 0.0, d + inst.border);
            return bg * inner + bc * clamp(outer - inner, 0.0, 1.0);
        }
        return bg * outer;
    }
    var color = inst.bg;
    if (inst.border > 0.0) {
        let inside_d = d + inst.border;
        let inner = 1.0 - smoothstep(-1.0, 0.0, inside_d);
        let border_mask = clamp(outer - inner, 0.0, 1.0);
        color = inst.bg * inner + inst.border_color * border_mask;
        // Pre-multiplied alpha for blending.
        color.a = outer;
    } else {
        color = inst.bg * outer;
        color.a = inst.bg.a * outer;
    }
    return color;
}
