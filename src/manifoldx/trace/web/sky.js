// Light from infinitely far away, as manifoldx/trace/environment.py: a Preetham clear sky with
// a sun disk, and the luminance tables the path kernel samples it by.
//
// An environment is {radiance: Float32Array (H*W*3, row 0 at the zenith), width, height,
// intensity, sun_direction, sun_color, sun_illuminance, sun_angular_radius}.

export const SUN_ANGULAR_RADIUS = 0.2665 * Math.PI / 180;
const LUM = [0.2126, 0.7152, 0.0722];

/** Oikos's settings shape: {sun_direction, turbidity, ground}; Environment.sky's defaults. */
export function sky({ sun_direction = [0.4, 0.7, 0.3], turbidity = 3.0, sun_ratio = 5.0,
  ground = [0.25, 0.23, 0.2], width = 512, height = 256 } = {}) {
  const l = Math.hypot(...sun_direction);
  const s = sun_direction.map((x) => x / l);
  const radiance = new Float64Array(width * height * 3);
  const dir = (y, x) => {
    const lat = (0.5 - (y + 0.5) / height) * Math.PI;
    const phi = ((x + 0.5) / width - 0.5) * 2 * Math.PI;
    return [Math.cos(lat) * Math.cos(phi), Math.sin(lat), Math.cos(lat) * Math.sin(phi)];
  };
  const P = preetham(s, turbidity);
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const rgb = P(dir(y, x));
      radiance.set(rgb, (y * width + x) * 3);
    }
  }
  const skyE = horizontalIlluminance(radiance, width, height);
  const sunE = sun_ratio * skyE;
  const sunRgb = sunTint(Math.max(s[1], 0), turbidity);
  const eGround = skyE + sunE * Math.max(s[1], 0);
  const groundRad = ground.map((g) => g * eGround / Math.PI);
  const horizon = [0, 0, 0];
  const row = height / 2 - 1 | 0;
  for (let x = 0; x < width; x++) for (let c = 0; c < 3; c++) horizon[c] += radiance[(row * width + x) * 3 + c] / width;
  const out = new Float32Array(width * height * 3);
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const up = dir(y, x)[1];
      const i = (y * width + x) * 3;
      const blend = Math.min(Math.max(-up / 0.05, 0), 1);
      for (let c = 0; c < 3; c++) {
        out[i + c] = up >= 0 ? radiance[i + c] : horizon[c] * (1 - blend) + groundRad[c] * blend;
      }
    }
  }
  return { radiance: out, width, height, intensity: 1, sun_direction: s, sun_color: sunRgb,
    sun_illuminance: s[1] > -0.02 ? sunE : 0, sun_angular_radius: SUN_ANGULAR_RADIUS };
}

/** Environment.tables: (radiance and pdf per texel (H*W*4), marginal cdf (H), conditional
 * cdf (H*W), total weight). */
export function tables(env) {
  const { width: W, height: H } = env;
  const k = env.intensity ?? 1;
  const n = W * H;
  const lum = new Float64Array(n);
  let mean = 0;
  for (let i = 0; i < n; i++) {
    lum[i] = k * (env.radiance[i * 3] * LUM[0] + env.radiance[i * 3 + 1] * LUM[1] + env.radiance[i * 3 + 2] * LUM[2]);
    mean += lum[i] / n;
  }
  const w = new Float64Array(n);
  const rows = new Float64Array(H);
  for (let y = 0; y < H; y++) {
    const sin = Math.sin((y + 0.5) / H * Math.PI);
    for (let x = 0; x < W; x++) {
      const i = y * W + x;
      w[i] = (lum[i] + 1e-6 * Math.max(mean, 1e-12)) * sin;
      rows[y] += w[i];
    }
  }
  let total = 0;
  for (let y = 0; y < H; y++) total += rows[y];
  const marginal = new Float32Array(H);
  const conditional = new Float32Array(n);
  const radiance = new Float32Array(n * 4);
  let acc = 0;
  for (let y = 0; y < H; y++) {
    acc += rows[y];
    marginal[y] = acc / total;
    let racc = 0;
    for (let x = 0; x < W; x++) {
      const i = y * W + x;
      racc += w[i];
      conditional[i] = racc / rows[y];
      for (let c = 0; c < 3; c++) radiance[i * 4 + c] = env.radiance[i * 3 + c] * k;
      radiance[i * 4 + 3] = w[i] / total * H * W;
    }
    conditional[y * W + W - 1] = 1;
  }
  marginal[H - 1] = 1;
  return { radiance, marginal, conditional, total };
}

function horizontalIlluminance(rad, W, H) {
  let e = 0;
  for (let y = 0; y < H; y++) {
    const lat = (0.5 - (y + 0.5) / H) * Math.PI;
    const up = Math.sin(lat);
    const solid = Math.cos(lat) * (Math.PI / H) * (2 * Math.PI / W);
    const f = Math.max(up, 0) * solid;
    for (let x = 0; x < W; x++) {
      const i = (y * W + x) * 3;
      e += (rad[i] * LUM[0] + rad[i + 1] * LUM[1] + rad[i + 2] * LUM[2]) * f;
    }
  }
  return e;
}

function perez(theta, gamma, [A, B, C, D, E]) {
  return (1 + A * Math.exp(B / Math.max(Math.cos(theta), 0.01))) * (1 + C * Math.exp(D * gamma) + E * Math.cos(gamma) ** 2);
}

function preetham(sun, T) {
  const thetaS = Math.min(Math.acos(Math.min(Math.max(sun[1], -1), 1)), 89.5 * Math.PI / 180);
  const Y = [0.1787 * T - 1.4630, -0.3554 * T + 0.4275, -0.0227 * T + 5.3251, 0.1206 * T - 2.5771, -0.0670 * T + 0.3703];
  const x = [-0.0193 * T - 0.2592, -0.0665 * T + 0.0008, -0.0004 * T + 0.2125, -0.0641 * T - 0.8989, -0.0033 * T + 0.0452];
  const y = [-0.0167 * T - 0.2608, -0.0950 * T + 0.0092, -0.0079 * T + 0.2102, -0.0441 * T - 1.6537, -0.0109 * T + 0.0529];
  const t = thetaS;
  const t2 = t * t;
  const t3 = t2 * t;
  const zx = (0.00166 * t3 - 0.00375 * t2 + 0.00209 * t) * T * T
    + (-0.02903 * t3 + 0.06377 * t2 - 0.03202 * t + 0.00394) * T + (0.11693 * t3 - 0.21196 * t2 + 0.06052 * t + 0.25886);
  const zy = (0.00275 * t3 - 0.00610 * t2 + 0.00317 * t) * T * T
    + (-0.04214 * t3 + 0.08970 * t2 - 0.04153 * t + 0.00516) * T + (0.15346 * t3 - 0.26756 * t2 + 0.06670 * t + 0.26688);
  const pY = perez(0, thetaS, Y);
  const px = perez(0, thetaS, x);
  const py = perez(0, thetaS, y);
  return (d) => {
    const theta = Math.acos(Math.max(d[1], 0.001));
    const gamma = Math.acos(Math.min(Math.max(d[0] * sun[0] + d[1] * sun[1] + d[2] * sun[2], -1), 1));
    const lum = perez(theta, gamma, Y) / pY;
    const cx = zx * perez(theta, gamma, x) / px;
    const cy = zy * perez(theta, gamma, y) / py;
    const X = cx / cy * lum;
    const Z = (1 - cx - cy) / cy * lum;
    return [Math.max(3.2406 * X - 1.5372 * lum - 0.4986 * Z, 0),
      Math.max(-0.9689 * X + 1.8758 * lum + 0.0415 * Z, 0),
      Math.max(0.0557 * X - 0.2040 * lum + 1.0570 * Z, 0)];
  };
}

function sunTint(elevationSin, T) {
  const m = 1 / Math.max(elevationSin + 0.15 * (elevationSin * 90 + 3.885) ** -1.253, 0.02);
  const beta = 0.04608 * T - 0.04586;
  const tau = [0.68, 0.55, 0.44].map((lam) => Math.exp(-m * (0.008735 * lam ** -4.08 + beta * lam ** -1.3)));
  const l = tau[0] * LUM[0] + tau[1] * LUM[1] + tau[2] * LUM[2];
  return tau.map((c) => c / l);
}
