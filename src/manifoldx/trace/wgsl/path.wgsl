// One path per pixel per dispatch: GGX + Lambert, dielectrics, emission, environment and sun,
// punctual lights and emissive triangles, next-event estimation with MIS, Russian roulette.

@group(0) @binding(7) var<storage, read_write> accum: array<vec4<f32>>;
@group(0) @binding(8) var<storage, read_write> aov_albedo: array<vec4<f32>>;
@group(0) @binding(9) var<storage, read_write> aov_normal_depth: array<vec4<f32>>;
@group(0) @binding(10) var<storage, read> env: array<vec4<f32>>;         // radiance, pdf over uv
@group(0) @binding(11) var<storage, read> env_marginal: array<f32>;
@group(0) @binding(12) var<storage, read> env_conditional: array<f32>;
@group(0) @binding(13) var<storage, read> lights: array<vec4<f32>>;      // 3 per light
@group(0) @binding(14) var<storage, read> emissive: array<vec4<f32>>;    // tri (bits), cdf, -, -

const DIFFUSE_ONLY: u32 = 1u;

fn luminance(c: vec3<f32>) -> f32 { return dot(c, vec3<f32>(0.2126, 0.7152, 0.0722)); }

fn power_heuristic(a: f32, b: f32) -> f32 {
    let a2 = a * a;
    return a2 / max(a2 + b * b, 1.0e-30);
}

fn basis(n: vec3<f32>) -> mat3x3<f32> {
    let s = select(-1.0, 1.0, n.z >= 0.0);
    let a = -1.0 / (s + n.z);
    let b = n.x * n.y * a;
    return mat3x3<f32>(vec3<f32>(1.0 + s * n.x * n.x * a, s * b, -s * n.x),
                       vec3<f32>(b, s + n.y * n.y * a, -n.y), n);
}

// ---------------------------------------------------------------- environment

fn env_uv(d: vec3<f32>) -> vec2<f32> {
    let u = atan2(d.z, d.x) / (2.0 * PI) + 0.5;
    let v = 0.5 - asin(clamp(d.y, -1.0, 1.0)) / PI;
    return vec2<f32>(u, v);
}

fn env_texel(d: vec3<f32>) -> vec4<f32> {
    let uv = env_uv(d);
    let x = min(u32(uv.x * f32(P.env_width)), P.env_width - 1u);
    let y = min(u32(uv.y * f32(P.env_height)), P.env_height - 1u);
    return env[y * P.env_width + x];
}

fn env_pdf(d: vec3<f32>, pdf_uv: f32) -> f32 {
    let sin_theta = sqrt(max(1.0 - d.y * d.y, 1.0e-8));
    return pdf_uv / (2.0 * PI * PI * sin_theta);
}

// first index in [lo, lo + n) whose value exceeds u
fn search(lo: u32, n: u32, u: f32, which: u32) -> u32 {
    var a = 0u;
    var b = n - 1u;
    loop {
        if (a >= b) { break; }
        let m = (a + b) / 2u;
        var v: f32;
        if (which == 0u) { v = env_marginal[lo + m]; } else { v = env_conditional[lo + m]; }
        if (v > u) { b = m; } else { a = m + 1u; }
    }
    return a;
}

struct LightSample { dir: vec3<f32>, dist: f32, radiance: vec3<f32>, pdf: f32 };

fn sample_env(r: ptr<function, Rng>) -> LightSample {
    let y = search(0u, P.env_height, rand(r), 0u);
    let x = search(y * P.env_width, P.env_width, rand(r), 1u);
    let u = (f32(x) + rand(r)) / f32(P.env_width);
    let v = (f32(y) + rand(r)) / f32(P.env_height);
    let phi = (u - 0.5) * 2.0 * PI;
    let lat = (0.5 - v) * PI;
    let d = vec3<f32>(cos(lat) * cos(phi), sin(lat), cos(lat) * sin(phi));
    let e = env[y * P.env_width + x];
    return LightSample(d, INF, e.xyz, env_pdf(d, e.w));
}

fn sun_pdf() -> f32 { return 1.0 / (2.0 * PI * (1.0 - P.sun_cos)); }

fn sample_sun(r: ptr<function, Rng>) -> LightSample {
    let cos_t = 1.0 - rand(r) * (1.0 - P.sun_cos);
    let sin_t = sqrt(max(0.0, 1.0 - cos_t * cos_t));
    let phi = 2.0 * PI * rand(r);
    let d = basis(P.sun_dir) * vec3<f32>(sin_t * cos(phi), sin_t * sin(phi), cos_t);
    return LightSample(d, INF, P.sun_radiance, sun_pdf());
}

// ---------------------------------------------------------------- materials

fn fresnel_schlick(f0: vec3<f32>, c: f32) -> vec3<f32> {
    return f0 + (1.0 - f0) * pow(clamp(1.0 - c, 0.0, 1.0), 5.0);
}

fn fresnel_dielectric(cos_i: f32, eta: f32) -> f32 {
    // eta = n_incident / n_transmitted
    let sin2_t = eta * eta * max(0.0, 1.0 - cos_i * cos_i);
    if (sin2_t >= 1.0) { return 1.0; }
    let cos_t = sqrt(1.0 - sin2_t);
    let rs = (eta * cos_i - cos_t) / (eta * cos_i + cos_t);
    let rp = (cos_i - eta * cos_t) / (cos_i + eta * cos_t);
    return 0.5 * (rs * rs + rp * rp);
}

fn ggx_d(noh: f32, a2: f32) -> f32 {
    let k = noh * noh * (a2 - 1.0) + 1.0;
    return a2 / (PI * k * k);
}

fn smith_g1(nov: f32, a2: f32) -> f32 {
    return 2.0 * nov / (nov + sqrt(a2 + (1.0 - a2) * nov * nov));
}

fn spec_probability(s: Surface, nov: f32) -> f32 {
    let f0 = mix(vec3<f32>(0.04), s.albedo, s.metallic);
    let fs = luminance(fresnel_schlick(f0, nov));
    let fd = luminance(s.albedo) * (1.0 - s.metallic);
    if ((P.flags & DIFFUSE_ONLY) != 0u) { return 0.0; }
    return clamp(fs / max(fs + fd, 1.0e-6), 0.1, 0.95);
}

// BSDF × cos and pdf of the opaque layer, wo and wi pointing away from the surface.
struct Eval { f: vec3<f32>, pdf: f32 };

fn eval_opaque(s: Surface, wo: vec3<f32>, wi: vec3<f32>) -> Eval {
    let n = s.n;
    let nol = dot(n, wi);
    let nov = max(dot(n, wo), 1.0e-4);
    if (nol <= 0.0 || dot(s.ng, wi) <= 0.0) { return Eval(vec3<f32>(0.0), 0.0); }
    let p_spec = spec_probability(s, nov);
    let diffuse = (1.0 - s.metallic) * s.albedo / PI;
    if (p_spec == 0.0) {
        return Eval(diffuse * nol, nol / PI);
    }
    let a = s.roughness * s.roughness;
    let a2 = a * a;
    let h = normalize(wo + wi);
    let noh = max(dot(n, h), 0.0);
    let voh = max(dot(wo, h), 0.0);
    let f0 = mix(vec3<f32>(0.04), s.albedo, s.metallic);
    let F = fresnel_schlick(f0, voh);
    let D = ggx_d(noh, a2);
    let G = smith_g1(nov, a2) * smith_g1(nol, a2);
    let spec = D * G * F / (4.0 * nov * nol);
    let pdf_spec = D * smith_g1(nov, a2) / (4.0 * nov);
    let f = (diffuse * (1.0 - F) + spec) * nol;
    return Eval(f, p_spec * pdf_spec + (1.0 - p_spec) * nol / PI);
}

// Heitz 2018: sample a visible GGX normal, in the frame of n.
fn sample_vndf(wo_local: vec3<f32>, a: f32, u1: f32, u2: f32) -> vec3<f32> {
    let vh = normalize(vec3<f32>(a * wo_local.x, a * wo_local.y, wo_local.z));
    let lensq = vh.x * vh.x + vh.y * vh.y;
    var t1 = vec3<f32>(1.0, 0.0, 0.0);
    if (lensq > 0.0) { t1 = vec3<f32>(-vh.y, vh.x, 0.0) / sqrt(lensq); }
    let t2 = cross(vh, t1);
    let r = sqrt(u1);
    let phi = 2.0 * PI * u2;
    let p1 = r * cos(phi);
    let s = 0.5 * (1.0 + vh.z);
    let p2 = (1.0 - s) * sqrt(max(0.0, 1.0 - p1 * p1)) + s * r * sin(phi);
    let nh = p1 * t1 + p2 * t2 + sqrt(max(0.0, 1.0 - p1 * p1 - p2 * p2)) * vh;
    return normalize(vec3<f32>(a * nh.x, a * nh.y, max(0.0, nh.z)));
}

fn sample_opaque_dir(s: Surface, wo: vec3<f32>, r: ptr<function, Rng>) -> vec3<f32> {
    let nov = max(dot(s.n, wo), 1.0e-4);
    let frame = basis(s.n);
    if (rand(r) < spec_probability(s, nov)) {
        let a = s.roughness * s.roughness;
        let wo_local = transpose(frame) * wo;
        let h = frame * sample_vndf(wo_local, a, rand(r), rand(r));
        return reflect(-wo, h);
    }
    let u1 = rand(r);
    let phi = 2.0 * PI * rand(r);
    let rr = sqrt(u1);
    return frame * vec3<f32>(rr * cos(phi), rr * sin(phi), sqrt(max(0.0, 1.0 - u1)));
}

// ---------------------------------------------------------------- shadows

// How much light gets from p along d over dist: 0 behind an opaque surface, tinted through glass.
fn transmittance(p: vec3<f32>, d: vec3<f32>, dist: f32) -> vec3<f32> {
    var o = p;
    var left = dist;
    var t = vec3<f32>(1.0);
    for (var i = 0; i < 8; i = i + 1) {
        let hit = trace(o, d, left);
        if (hit.tri == 0xffffffffu) { return t; }
        let s = surface(hit, o, d);
        if (s.transmission <= 0.0) { return vec3<f32>(0.0); }
        let c = abs(dot(s.n, d));
        let eta = select(s.ior, 1.0 / s.ior, s.front || s.thin);
        t = t * s.transmission * (1.0 - fresnel_dielectric(c, eta)) * s.albedo;
        if (max(t.x, max(t.y, t.z)) < 1.0e-4) { return vec3<f32>(0.0); }
        o = s.p + d * EPS * 4.0;
        left = left - hit.t - EPS * 4.0;
        if (left <= 0.0) { return t; }
    }
    return vec3<f32>(0.0);
}

fn offset(s: Surface, d: vec3<f32>) -> vec3<f32> {
    let side = select(-1.0, 1.0, dot(d, s.ng) > 0.0);
    return s.p + s.ng * side * EPS * max(1.0, length(s.p) * 0.002);
}

// ---------------------------------------------------------------- emissive triangles

fn tri_area(i: u32) -> f32 {
    return 0.5 * length(cross(tris[3u * i + 1u].xyz, tris[3u * i + 2u].xyz));
}

fn emissive_pdf(tri: u32, emission: vec3<f32>, cos_l: f32, dist: f32) -> f32 {
    let area = tri_area(tri);
    let select_p = luminance(emission) * area / P.emissive_total;
    return select_p * dist * dist / max(cos_l * area, 1.0e-12);
}

fn sample_emissive(p: vec3<f32>, r: ptr<function, Rng>) -> LightSample {
    let u = rand(r);
    var a = 0u;
    var b = P.emissive_count - 1u;
    loop {
        if (a >= b) { break; }
        let m = (a + b) / 2u;
        if (emissive[m].y > u) { b = m; } else { a = m + 1u; }
    }
    let tri = bitcast<u32>(emissive[a].x);
    var u1 = rand(r);
    var u2 = rand(r);
    if (u1 + u2 > 1.0) { u1 = 1.0 - u1; u2 = 1.0 - u2; }
    let q = tris[3u * tri].xyz + tris[3u * tri + 1u].xyz * u1 + tris[3u * tri + 2u].xyz * u2;
    let to = q - p;
    let dist = length(to);
    let d = to / dist;
    let ng = normalize(cross(tris[3u * tri + 1u].xyz, tris[3u * tri + 2u].xyz));
    let cos_l = abs(dot(ng, d));
    let mat = bitcast<u32>(shade[5u * tri].w);
    let e = mats[3u * mat + 1u].xyz;
    if (cos_l < 1.0e-6) { return LightSample(d, dist, vec3<f32>(0.0), 0.0); }
    return LightSample(d, dist * (1.0 - 1.0e-3), e, emissive_pdf(tri, e, cos_l, dist));
}

// ---------------------------------------------------------------- next-event estimation

fn direct(s: Surface, wo: vec3<f32>, r: ptr<function, Rng>) -> vec3<f32> {
    var sum = vec3<f32>(0.0);
    var ls: LightSample;
    if (P.sun_cos < 1.0 && P.sun_dir.y > -0.1) {
        ls = sample_sun(r);
        let e = eval_opaque(s, wo, ls.dir);
        if (e.pdf > 0.0) {
            let t = transmittance(offset(s, ls.dir), ls.dir, INF);
            sum += t * e.f * ls.radiance * power_heuristic(ls.pdf, e.pdf) / ls.pdf;
        }
    }
    if (P.env_width > 0u) {
        ls = sample_env(r);
        let e = eval_opaque(s, wo, ls.dir);
        if (e.pdf > 0.0 && ls.pdf > 0.0) {
            let t = transmittance(offset(s, ls.dir), ls.dir, INF);
            sum += t * e.f * ls.radiance * power_heuristic(ls.pdf, e.pdf) / ls.pdf;
        }
    }
    if (P.emissive_count > 0u) {
        ls = sample_emissive(s.p, r);
        if (ls.pdf > 0.0) {
            let e = eval_opaque(s, wo, ls.dir);
            if (e.pdf > 0.0) {
                let o = offset(s, ls.dir);
                let t = transmittance(o, ls.dir, length(s.p + ls.dir * ls.dist - o));
                sum += t * e.f * ls.radiance * power_heuristic(ls.pdf, e.pdf) / ls.pdf;
            }
        }
    }
    if (P.light_count > 0u) {
        let k = min(u32(rand(r) * f32(P.light_count)), P.light_count - 1u);
        let l0 = lights[3u * k];
        let l1 = lights[3u * k + 1u];
        let l2 = lights[3u * k + 2u];
        var d: vec3<f32>;
        var dist = INF;
        var radiance = l2.xyz;
        if (l0.w < 0.5) {
            d = -l1.xyz;
        } else {
            let to = l0.xyz - s.p;
            dist = length(to);
            d = to / dist;
            radiance = radiance / (dist * dist);
            if (l0.w > 1.5) {
                let c = dot(-d, l1.xyz);
                radiance = radiance * smoothstep(l1.w, l2.w, c);
            }
        }
        let e = eval_opaque(s, wo, d);
        if (e.pdf > 0.0 && luminance(radiance) > 0.0) {
            let o = offset(s, d);
            let t = transmittance(o, d, select(dist, length(l0.xyz - o), dist < INF));
            sum += t * e.f * radiance * f32(P.light_count);
        }
    }
    return sum;
}

// radiance from the sky along an escaped ray, MIS-weighted against NEE when not after a delta bounce
fn escaped(d: vec3<f32>, bsdf_pdf: f32, delta: bool) -> vec3<f32> {
    var l = vec3<f32>(0.0);
    if (P.env_width > 0u) {
        let e = env_texel(d);
        var w = 1.0;
        if (!delta) { w = power_heuristic(bsdf_pdf, env_pdf(d, e.w)); }
        l += e.xyz * w;
    }
    if (P.sun_cos < 1.0 && dot(d, P.sun_dir) >= P.sun_cos) {
        var w = 1.0;
        if (!delta) { w = power_heuristic(bsdf_pdf, sun_pdf()); }
        l += P.sun_radiance * w;
    }
    return l;
}

// ---------------------------------------------------------------- the path

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= P.width || gid.y >= P.height) { return; }
    let pixel = gid.y * P.width + gid.x;
    var rng = rng_init(pixel, P.frame, P.seed);
    var o = P.cam_pos;
    var d = camera_ray(vec2<f32>(gid.xy) + vec2<f32>(rand(&rng), rand(&rng)));
    var throughput = vec3<f32>(1.0);
    var radiance = vec3<f32>(0.0);
    var last_pdf = 0.0;
    var delta = true;  // the camera ray counts as a delta bounce: what it sees is not MIS-weighted
    var first_albedo = vec3<f32>(0.0);
    var first_nd = vec4<f32>(0.0, 0.0, 0.0, -1.0);
    var found_first = false;

    for (var bounce = 0u; bounce <= P.max_bounces; bounce = bounce + 1u) {
        let hit = trace(o, d, INF);
        if (hit.tri == 0xffffffffu) {
            var contrib = throughput * escaped(d, last_pdf, delta);
            if (bounce > 0u && P.clamp_max > 0.0) { contrib = min(contrib, vec3<f32>(P.clamp_max)); }
            radiance += contrib;
            if (!found_first) { first_albedo = vec3<f32>(1.0); found_first = true; }
            break;
        }
        let s = surface(hit, o, d);
        if (!found_first && s.transmission < 0.5) {
            first_albedo = s.albedo;
            first_nd = vec4<f32>(s.n, length(s.p - P.cam_pos));
            found_first = true;
        }
        if (luminance(s.emission) > 0.0) {
            var w = 1.0;
            if (!delta && P.emissive_count > 0u) {
                let cos_l = abs(dot(s.ng, d));
                w = power_heuristic(last_pdf, emissive_pdf(hit.tri, s.emission, cos_l, hit.t));
            }
            radiance += throughput * s.emission * w;
        }
        if (bounce == P.max_bounces) { break; }
        let wo = -d;

        // the dielectric layer, chosen with probability `transmission`
        if (s.transmission > 0.0 && rand(&rng) < s.transmission) {
            let c = abs(dot(s.n, wo));
            let eta = select(s.ior, 1.0 / s.ior, s.front || s.thin);  // n_incident / n_transmitted
            let F = fresnel_dielectric(c, eta);
            if (rand(&rng) < F) {
                d = reflect(d, s.n);
                o = offset(s, d);
                delta = true;
            } else if (s.thin) {
                // straight through: the direction and the MIS state stay as they were
                throughput *= s.albedo;
                o = s.p + d * EPS * 4.0;
            } else {
                d = refract(d, s.n, eta);
                if (dot(d, d) < 0.5) { d = reflect(-wo, s.n); }
                throughput *= s.albedo;
                o = offset(s, d);
                delta = true;
            }
            continue;
        }

        // the opaque layer: light sampled directly, then the BSDF picks where to go next
        var nee = throughput * direct(s, wo, &rng);
        if (bounce > 0u && P.clamp_max > 0.0) { nee = min(nee, vec3<f32>(P.clamp_max)); }
        radiance += nee;
        let wi = sample_opaque_dir(s, wo, &rng);
        let e = eval_opaque(s, wo, wi);
        if (e.pdf <= 0.0) { break; }
        throughput *= e.f / e.pdf;
        last_pdf = e.pdf;
        delta = false;
        d = wi;
        o = offset(s, d);

        if (bounce >= 2u) {
            let q = clamp(max(throughput.x, max(throughput.y, throughput.z)), 0.05, 1.0);
            if (rand(&rng) > q) { break; }
            throughput /= q;
        }
    }
    if (any(radiance != radiance) || any(abs(radiance) > vec3<f32>(1.0e20))) { radiance = vec3<f32>(0.0); }
    accum[pixel] += vec4<f32>(radiance, 1.0);
    aov_albedo[pixel] += vec4<f32>(first_albedo, 1.0);
    aov_normal_depth[pixel] += first_nd;
}
