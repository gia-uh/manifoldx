// ---- surface patterns: world-space comic textures, picked by a per-vertex id ----
const SP_INK = vec3<f32>(0.0080, 0.0063, 0.0112);  // #1c1a21 as pow(c, 2.2), as the reference

fn sp_mod(x: f32, y: f32) -> f32 { return x - y * floor(x / y); }
fn sp_h1(p: vec2<f32>) -> f32 { return fract(sin(dot(p, vec2<f32>(127.1, 311.7))) * 43758.5453); }
fn sp_h2(p: vec2<f32>) -> vec2<f32> {
    return fract(sin(vec2<f32>(dot(p, vec2<f32>(127.1, 311.7)), dot(p, vec2<f32>(269.5, 183.3)))) * 43758.5453);
}
fn sp_vnoise(p: vec2<f32>) -> f32 {
    let i = floor(p);
    var f = fract(p);
    f = f * f * (3.0 - 2.0 * f);
    return mix(mix(sp_h1(i), sp_h1(i + vec2<f32>(1.0, 0.0)), f.x),
               mix(sp_h1(i + vec2<f32>(0.0, 1.0)), sp_h1(i + vec2<f32>(1.0, 1.0)), f.x), f.y);
}
fn sp_fbm(p: vec2<f32>) -> f32 {
    return 0.55 * sp_vnoise(p) + 0.3 * sp_vnoise(p * 2.03 + 7.1) + 0.15 * sp_vnoise(p * 4.1 + 3.7);
}
fn sp_seg(p: vec2<f32>, a: vec2<f32>, b: vec2<f32>) -> f32 {
    let pa = p - a;
    let ba = b - a;
    return length(pa - ba * clamp(dot(pa, ba) / dot(ba, ba), 0.0, 1.0));
}
// Each pattern returns (tone 0..1, ink 0..1).
fn sp_grass(u: vec2<f32>, aa: f32, w: f32) -> vec2<f32> {
    let t = sp_fbm(u * 0.6);
    let c = floor(u * 1.6);
    let f = fract(u * 1.6);
    var ink = 0.0;
    if (sp_h1(c + 3.1) > 0.45) {
        let p = f - (0.25 + 0.5 * sp_h2(c));
        let d = min(min(sp_seg(p, vec2<f32>(0.0), vec2<f32>(-0.09, 0.16)),
                        sp_seg(p, vec2<f32>(0.0), vec2<f32>(0.0, 0.2))),
                    sp_seg(p, vec2<f32>(0.0), vec2<f32>(0.09, 0.16)));
        ink = 1.0 - smoothstep(w * 1.6, w * 1.6 + aa * 1.6, d);
    }
    return vec2<f32>(t, ink);
}
fn sp_cobbles(u: vec2<f32>, aa: f32, w: f32) -> vec2<f32> {
    let i = floor(u);
    let f = fract(u);
    var f1 = 9.0;
    var f2 = 9.0;
    var best = vec2<f32>(0.0);
    for (var y = -1; y <= 1; y++) {
        for (var x = -1; x <= 1; x++) {
            let g = vec2<f32>(f32(x), f32(y));
            let r = g + 0.15 + 0.7 * sp_h2(i + g) - f;
            let d = dot(r, r);
            if (d < f1) { f2 = f1; f1 = d; best = i + g; } else if (d < f2) { f2 = d; }
        }
    }
    return vec2<f32>(sp_h1(best), 1.0 - smoothstep(w, w + aa, sqrt(f2) - sqrt(f1)));
}
fn sp_veins(u: vec2<f32>, aa: f32, w: f32) -> vec2<f32> {
    let t = sp_fbm(u * 0.45);
    let v = abs(sp_vnoise(u * 0.7 + 3.0 * sp_fbm(u * 0.3)) - 0.5);
    return vec2<f32>(t, 1.0 - smoothstep(w * 0.5, w * 0.5 + aa, v));
}
fn sp_bricks(u: vec2<f32>, size: vec2<f32>, aa: f32, w: f32) -> vec2<f32> {
    var q = u / size;
    let row = floor(q.y);
    q.x += 0.5 * sp_mod(row, 2.0);
    let f = fract(q);
    let c = floor(q);
    let dx = min(f.x, 1.0 - f.x) * size.x;
    let dy = min(f.y, 1.0 - f.y) * size.y;
    return vec2<f32>(sp_h1(c + row * 7.0), 1.0 - smoothstep(w, w + aa, min(dx, dy)));
}
fn sp_planks(u: vec2<f32>, aa: f32, w: f32) -> vec2<f32> {
    let bw = 0.4;
    let row = floor(u.y / bw);
    let y = fract(u.y / bw) * bw;
    let len = 3.0;
    let x = u.x / len + sp_h1(vec2<f32>(row, 1.0));
    let fx = fract(x) * len;
    let d = min(min(y, bw - y), min(fx, len - fx));
    let grain = sp_vnoise(vec2<f32>(u.x * 0.6, u.y * 9.0) + row * 13.0);
    return vec2<f32>(0.35 * sp_h1(vec2<f32>(row, floor(x))) + 0.65 * grain,
                     1.0 - smoothstep(w, w + aa, d));
}
fn sp_tiles(u: vec2<f32>, aa: f32, w: f32) -> vec2<f32> {
    let size = vec2<f32>(0.6, 0.45);
    var q = u / size;
    let row = floor(q.y);
    q.x += 0.5 * sp_mod(row, 2.0);
    let f = fract(q);
    let c = floor(q);
    let e = 2.0 * f.x - 1.0;
    let edge_y = 0.28 * (1.0 - e * e);  // each tile's rounded lower edge
    let dy = abs(f.y - edge_y) * size.y;
    let dx = min(f.x, 1.0 - f.x) * size.x;
    let side = select(9.0, dx, f.y > edge_y);
    return vec2<f32>(0.5 * sp_h1(c + row * 5.0) + 0.5 * clamp(f.y, 0.0, 1.0),
                     1.0 - smoothstep(w, w + aa, min(dy, side)));
}
// Square cement slabs a metre across with their joints, a tone per slab. worn adds what age does: a crack
// across some slabs, stains that spread over several, and here and there a broken slab gone dark.
fn sp_pavement(u: vec2<f32>, aa: f32, w: f32, worn: bool) -> vec2<f32> {
    let c = floor(u);
    let f = fract(u);
    let joint = min(min(f.x, 1.0 - f.x), min(f.y, 1.0 - f.y));
    var ink = 1.0 - smoothstep(w, w + aa, joint);
    var t = 0.42 + 0.22 * sp_h1(c) + 0.12 * (sp_vnoise(u * 2.7) - 0.5);
    if (worn) {
        if (sp_h1(c + 7.3) < 0.35) {  // one crack, corner to edge
            let a = sp_h2(c + 1.7) * vec2<f32>(1.0, 0.25);
            let b = vec2<f32>(0.2, 0.75) + sp_h2(c + 4.1) * vec2<f32>(0.6, 0.25);
            let wob = 0.04 * (sp_vnoise(u * 9.0) - 0.5);
            ink = max(ink, 1.0 - smoothstep(w * 0.6, w * 0.6 + aa, sp_seg(f + wob, a, b)));
        }
        let stain = sp_fbm(u * 0.35 + 11.0);
        t = t - 0.35 * smoothstep(0.58, 0.72, stain);
        if (sp_h1(c + 3.9) < 0.06) { t = 0.12; }  // a broken slab: the dark under it shows
    }
    return vec2<f32>(clamp(t, 0.0, 1.0), ink);
}
// The albedo of surface `id` (0 none, 1 grass .. 7 tiles, 8 pavement, 9 worn pavement) round the linear colour `base`,
// at pattern coordinate `proj` (world units) whose screen derivative length is `fw`.
fn surface_albedo(base: vec3<f32>, id: i32, proj: vec2<f32>, fw: f32) -> vec3<f32> {
    if (id <= 0 || id > 9) { return base; }
    let k0 = id - 1;
    let prm = surface_settings.patterns[k0];  // scale, contrast, tones, ink width
    let dark = surface_settings.darks[k0 / 4][k0 % 4];
    let u = proj / prm.x;
    let fwu = fw / prm.x;
    let aa = fwu * 0.7;
    var r = vec2<f32>(0.5, 0.0);
    switch k0 {
        case 0: { r = sp_grass(u, aa, prm.w); }
        case 1: { r = sp_cobbles(u, aa, prm.w); }
        case 2: { r = sp_veins(u, aa, prm.w); }
        case 3: { r = sp_bricks(u, vec2<f32>(0.5, 0.25), aa, prm.w); }
        case 4: { r = sp_bricks(u, vec2<f32>(1.0, 0.55), aa, prm.w); }
        case 5: { r = sp_planks(u, aa, prm.w); }
        case 6: { r = sp_tiles(u, aa, prm.w); }
        case 7: { r = sp_pavement(u, aa, prm.w, false); }
        default: { r = sp_pavement(u, aa, prm.w, true); }
    }
    let t = r.x;
    var k = select(-0.5, 0.5, t >= 0.5);
    if (prm.z > 2.5) { k = select(select(1.0, 0.0, t < 0.7), -1.0, t < 0.38); }
    var col = base * (1.0 + k * prm.y * 1.6);
    col = mix(col, mix(col, SP_INK, dark), r.y);
    // Fade to the flat colour once a pattern cell is only a few pixels on screen.
    let fade = smoothstep(4.0, 10.0, 1.0 / max(fwu, 1e-5));
    return mix(base, col, fade);
}

// ---- procedural detail: asphalt and ground from world-space value noise ----
const DT_FAR = vec2<f32>(250.0, 600.0);  // metres from the eye over which all detail fades out
const DT_DRY = vec3<f32>(0.266, 0.171, 0.051);  // dry yellow-brown grass, linear (#8c7340)
const DT_DEEP = vec3<f32>(0.033, 0.089, 0.014);  // deep green, linear (#33541f)
const DT_SOIL = vec3<f32>(0.133, 0.073, 0.033);  // soil, linear (#664d33)

// Dave Hoskins' hash without sine, at four lattice points (x, y) at once: 0..1 each. The points are under 4096
// plus the seed's offset, where an f32 still holds every bit the hash needs.
fn dt_hash4(x: vec4<f32>, y: vec4<f32>, seed: f32) -> vec4<f32> {
    let a = fract((x + seed * 317.0) * 0.1031);
    let b = fract(y * 0.1030);
    let d = a * (b + 33.33) + b * (a + 33.33) + a * (a + 33.33);
    return fract((a + b + 2.0 * d) * (a + d));
}
// Value noise in 0..1 of wavelength lam metres (a power of two up to 256) at p, metres from the corner of
// the origin's tile `tile`. The lattice wraps every 4096 waves, far wider than the eye could match.
fn dt_noise(p: vec2<f32>, lam: f32, tile: vec2<i32>, seed: f32) -> f32 {
    let q = p / lam;
    let i = floor(q);
    var f = q - i;
    f = f * f * (3.0 - 2.0 * f);
    let c = vec2<i32>(i) + tile * i32(256.0 / lam);
    let x = vec2<f32>(vec2<i32>(c.x, c.x + 1) & vec2<i32>(4095));
    let y = vec2<f32>(vec2<i32>(c.y, c.y + 1) & vec2<i32>(4095));
    let h = dt_hash4(vec4<f32>(x.x, x.y, x.x, x.y), vec4<f32>(y.x, y.x, y.y, y.y), seed);
    return mix(mix(h.x, h.y, f.x), mix(h.z, h.w, f.x), f.y);
}
// How much of an octave of wavelength lam shows where a pixel spans fw metres: all of it from eight pixels
// a wave, none under three. fw.x is a pixel's mean span (the geometric mean of its two sides), for the soft
// octaves of two metres and more; fw.y its longest side, for the fine ones, which would shimmer down a
// street seen at a grazing angle.
fn dt_w(lam: f32, fw: vec2<f32>) -> f32 { return smoothstep(3.0, 8.0, lam / select(fw.y, fw.x, lam >= 2.0)); }

// Asphalt: soft blotches metres across (repaired patches darker and smoother, worn stretches greyer and
// rougher), stains a metre across, grain, and the aggregate's light stones a few centimetres across.
fn dt_asphalt(base: vec3<f32>, p: vec2<f32>, t: vec2<i32>, fw: vec2<f32>, far: f32) -> vec4<f32> {
    var col = base;
    var r = 0.0;
    let w8 = dt_w(8.0, fw) * far;
    if (w8 > 0.0) {
        let n = dt_noise(p, 8.0, t, 1.0);
        let patched = smoothstep(0.62, 0.7, n) * w8;
        let worn = smoothstep(0.38, 0.3, n) * w8;
        let grey = vec3<f32>(dot(col, vec3<f32>(0.2126, 0.7152, 0.0722)));
        col = col * (1.0 + 0.6 * (n - 0.5) * w8) * (1.0 - 0.4 * patched);
        col = mix(col, grey * 1.7, 0.6 * worn);
        r += 0.06 * worn - 0.15 * patched;
    }
    let w1 = dt_w(1.0, fw) * far;
    if (w1 > 0.0) {
        col *= 1.0 + 0.8 * (dt_noise(p, 1.0, t, 2.0) - 0.5) * w1;
    }
    let w025 = dt_w(0.25, fw) * far;
    if (w025 > 0.0) {
        col *= 1.0 + 0.9 * (dt_noise(p, 0.25, t, 3.0) - 0.5) * w025;
    }
    let w003 = dt_w(0.03125, fw) * far;
    if (w003 > 0.0) {
        let s = dt_noise(p, 0.03125, t, 4.0);
        col *= 1.0 + (0.9 * smoothstep(0.62, 0.85, s) - 0.5 * smoothstep(0.38, 0.15, s)) * w003;
    }
    return vec4<f32>(col, r);
}

// Ground: mottled greens and dry yellow-brown over tens of metres (on top of what the vertex colours vary
// over larger distances), lighter and darker patches a few metres across with soil showing through, most
// where it is dry, then tufts down to a few centimetres.
fn dt_ground(base: vec3<f32>, p: vec2<f32>, t: vec2<i32>, fw: vec2<f32>, far: f32) -> vec4<f32> {
    var col = base;
    var dry = 0.5;
    let w16 = dt_w(16.0, fw) * far;
    if (w16 > 0.0) {
        dry = dt_noise(p, 16.0, t, 5.0);
        let lum = dot(base, vec3<f32>(0.2126, 0.7152, 0.0722));
        col = mix(col, DT_DRY * (1.3 * lum / 0.18), 0.7 * smoothstep(0.5, 0.7, dry) * w16);
        col = mix(col, DT_DEEP * (lum / 0.07), 0.5 * smoothstep(0.5, 0.3, dry) * w16);
    }
    let w2 = dt_w(2.0, fw) * far;
    if (w2 > 0.0) {
        let n = dt_noise(p, 2.0, t, 6.0);
        col *= 1.0 + 0.5 * (n - 0.5) * w2;
        col = mix(col, DT_SOIL, 0.7 * smoothstep(0.7, 0.8, 0.7 * n + 0.3 * dry) * w2);
    }
    let w025 = dt_w(0.25, fw) * far;
    if (w025 > 0.0) {
        col *= 1.0 + 0.5 * (dt_noise(p, 0.25, t, 7.0) - 0.5) * w025;
    }
    let w006 = dt_w(0.0625, fw) * far;
    if (w006 > 0.0) {
        col *= 1.0 + 0.4 * (dt_noise(p, 0.0625, t, 8.0) - 0.5) * w006;
    }
    return vec4<f32>(col, 0.0);
}

// The albedo (xyz) and roughness change (w) of surface `id` round the linear colour `base`, at p metres from
// the corner of the origin's tile, where a pixel spans fw metres (see dt_w) and the eye is dist metres away.
fn detail_surface(base: vec3<f32>, id: i32, p: vec2<f32>, fw: vec2<f32>, dist: f32) -> vec4<f32> {
    let far = 1.0 - smoothstep(DT_FAR.x, DT_FAR.y, dist);
    if (far <= 0.0 || (id != 10 && id != 11)) { return vec4<f32>(base, 0.0); }
    let t = vec2<i32>(camera.origin_tile.xy);
    if (id == 10) { return dt_asphalt(base, p, t, fw, far); }
    return dt_ground(base, p, t, fw, far);
}
