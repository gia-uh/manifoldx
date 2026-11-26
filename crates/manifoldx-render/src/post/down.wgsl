@group(0) @binding(1) var src: texture_2d<f32>;
@group(0) @binding(2) var lin: sampler;

@fragment
fn fs(in: VOut) -> @location(0) vec4<f32> {
    let px = 1.0 / vec2<f32>(textureDimensions(src));
    var c = textureSampleLevel(src, lin, in.uv, 0.0).rgb * 0.5;
    c = c + textureSampleLevel(src, lin, in.uv + vec2<f32>(-1.0, -1.0) * px, 0.0).rgb * 0.125;
    c = c + textureSampleLevel(src, lin, in.uv + vec2<f32>(1.0, -1.0) * px, 0.0).rgb * 0.125;
    c = c + textureSampleLevel(src, lin, in.uv + vec2<f32>(-1.0, 1.0) * px, 0.0).rgb * 0.125;
    c = c + textureSampleLevel(src, lin, in.uv + vec2<f32>(1.0, 1.0) * px, 0.0).rgb * 0.125;
    return vec4<f32>(c, 1.0);
}
