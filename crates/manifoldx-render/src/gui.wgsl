struct Quad {
    rect: vec4<f32>,
    clip: vec4<f32>,
    fill: vec4<f32>,
    outline: vec4<f32>,
    uv: vec4<f32>,
    style: vec4<f32>, // radius, border, exact colour, source kind
};

@group(0) @binding(0) var<uniform> viewport: vec4<f32>;
@group(0) @binding(1) var<storage, read> quads: array<Quad>;
@group(1) @binding(0) var image: texture_2d<f32>;
@group(1) @binding(1) var image_sampler: sampler;

struct Vertex {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) @interpolate(flat) index: u32,
};

@vertex
fn vertex(@builtin(vertex_index) index: u32, @builtin(instance_index) instance: u32) -> Vertex {
    let corners = array(
        vec2<f32>(0, 0), vec2<f32>(0, 1), vec2<f32>(1, 0),
        vec2<f32>(1, 0), vec2<f32>(0, 1), vec2<f32>(1, 1),
    );
    let uv = corners[index];
    let pixel = quads[instance].rect.xy + uv * quads[instance].rect.zw;
    let ndc = pixel / viewport.xy * vec2<f32>(2, -2) + vec2<f32>(-1, 1);
    return Vertex(vec4<f32>(ndc, 0, 1), uv, instance);
}

fn premultiply(color: vec4<f32>) -> vec4<f32> {
    return vec4<f32>(color.rgb * color.a, color.a);
}

@fragment
fn fragment(input: Vertex) -> @location(0) vec4<f32> {
    let q = quads[input.index];
    let sampled = textureSample(image, image_sampler, mix(q.uv.xy, q.uv.zw, input.uv));
    let p = input.position.xy;
    if any(p < q.clip.xy) || any(p >= q.clip.xy + q.clip.zw) {
        discard;
    }
    if q.style.w == 1.0 {
        return premultiply(q.fill) * sampled.r;
    }
    let half = q.rect.zw * 0.5;
    let radius = clamp(q.style.x, 0.0, min(half.x, half.y));
    let local = abs((input.uv - 0.5) * q.rect.zw) - half + radius;
    let distance = length(max(local, vec2<f32>(0))) + min(max(local.x, local.y), 0.0) - radius;
    let outer = 1.0 - smoothstep(-1.0, 0.0, distance);
    if q.style.w == 2.0 {
        return premultiply(sampled * q.fill) * outer;
    }
    let inner = 1.0 - smoothstep(-1.0, 0.0, distance + q.style.y);
    if q.style.z == 1.0 {
        return premultiply(q.fill) * inner + premultiply(q.outline) * (outer - inner);
    }
    if q.style.y > 0.0 {
        return vec4<f32>(q.fill.rgb * inner + q.outline.rgb * (outer - inner), outer);
    }
    return q.fill * outer;
}
