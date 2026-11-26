@group(0) @binding(1) var low: texture_2d<f32>;
@group(0) @binding(2) var lin: sampler;
@group(0) @binding(3) var cur: texture_2d<f32>;

@fragment
fn fs(in: VOut) -> @location(0) vec4<f32> {
    let px = 1.0 / vec2<f32>(textureDimensions(low));
    var c = textureSampleLevel(low, lin, in.uv, 0.0).rgb * 4.0;
    c = c + textureSampleLevel(low, lin, in.uv + vec2<f32>(-1.0, 0.0) * px, 0.0).rgb * 2.0;
    c = c + textureSampleLevel(low, lin, in.uv + vec2<f32>(1.0, 0.0) * px, 0.0).rgb * 2.0;
    c = c + textureSampleLevel(low, lin, in.uv + vec2<f32>(0.0, -1.0) * px, 0.0).rgb * 2.0;
    c = c + textureSampleLevel(low, lin, in.uv + vec2<f32>(0.0, 1.0) * px, 0.0).rgb * 2.0;
    c = c + textureSampleLevel(low, lin, in.uv + vec2<f32>(-1.0, -1.0) * px, 0.0).rgb;
    c = c + textureSampleLevel(low, lin, in.uv + vec2<f32>(1.0, -1.0) * px, 0.0).rgb;
    c = c + textureSampleLevel(low, lin, in.uv + vec2<f32>(-1.0, 1.0) * px, 0.0).rgb;
    c = c + textureSampleLevel(low, lin, in.uv + vec2<f32>(1.0, 1.0) * px, 0.0).rgb;
    return vec4<f32>(c / 16.0 + textureSampleLevel(cur, lin, in.uv, 0.0).rgb, 1.0);
}
