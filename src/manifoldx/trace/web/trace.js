// manifoldx.trace in the browser: the same WGSL kernels as manifoldx/trace/tracer.py, on the
// visitor's GPU through WebGPU.
//
//   import { loadGlb, createTracer } from './trace.js';
//   const scene = await loadGlb('scene.glb', { overrides, hide });
//   const tracer = await createTracer(scene, { width: 1024, height: 640, canvas,
//                                              environment: { sun_direction: [0.4, 0.7, 0.3] } });
//   tracer.setCamera({ position, target, up, fov });  // a camera that moved restarts the image
//   tracer.start();                                   // refines on the canvas until stop()
//
// The kernels are fetched from ../wgsl/ beside this file, as the package lays them out.

import { build } from './bvh.js';
import { sky, tables } from './sky.js';

export { loadGlb, readGlb } from './glb.js';
export { sky } from './sky.js';

const WG = 8;
const PARAMS_SIZE = 176;
const VIEW_SIZE = 32;
const LUM = [0.2126, 0.7152, 0.0722];
const TONEMAPS = { aces: 0, reinhard: 1, clamp: 2 };
const KINDS = { directional: 0, point: 1, spot: 2 };

/** Whether this browser can trace: WebGPU and an adapter. */
export async function supported() {
  return Boolean(navigator.gpu && await navigator.gpu.requestAdapter());
}

export async function createTracer(scene, options = {}) {
  const wgsl = options.wgsl || new URL('../wgsl/', import.meta.url);
  const names = ['common', 'path', 'prep', 'atrous', 'display'];
  const sources = Object.fromEntries(await Promise.all(names.map(async (n) => {
    const r = await fetch(new URL(`${n}.wgsl`, wgsl));
    if (!r.ok) throw new Error(`${n}.wgsl: HTTP ${r.status}`);
    return [n, await r.text()];
  })));
  const device = options.device || await requestDevice();
  return new Tracer(device, sources, scene, options);
}

export async function requestDevice() {
  if (!navigator.gpu) throw new Error('this browser has no WebGPU');
  const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
  if (!adapter) throw new Error('WebGPU has no adapter here');
  const { maxStorageBufferBindingSize, maxBufferSize } = adapter.limits;
  return adapter.requestDevice({ requiredLimits: { maxStorageBufferBindingSize, maxBufferSize } });
}

/** A three.js PerspectiveCamera as the tracer's camera, without importing three. */
export function cameraFromThree(camera) {
  camera.updateMatrixWorld();
  const m = camera.matrixWorld.elements;
  const position = [m[12], m[13], m[14]];
  return { position, target: [m[12] - m[8], m[13] - m[9], m[14] - m[10]], up: [m[4], m[5], m[6]], fov: camera.fov };
}

export class Tracer {
  constructor(device, sources, scene, { width = 640, height = 400, canvas = null, environment = null,
    maxBounces = 6, seed = 0, clamp = 0, diffuseOnly = false, camera = null } = {}) {
    this.device = device;
    this.sources = sources;
    this.scene = scene;
    this.maxBounces = maxBounces;
    this.seed = seed;
    this.clamp = clamp;
    this.diffuseOnly = diffuseOnly;
    if (environment && !environment.radiance) environment = sky(environment);
    this.environment = environment || scene.environment || sky();
    this.samples = 0;
    this.msPerSample = null;
    this.exposure = 'auto';
    this.tonemap = 'aces';
    this.pipelines = {};
    let t = performance.now();
    this.bvh = build(scene.positions);
    this.buildMs = performance.now() - t;
    t = performance.now();
    this.upload();
    this.uploadMs = performance.now() - t;
    this.camera = camera || scene.camera || defaultCamera(scene);
    this.resize(width, height);
    if (canvas) this.attach(canvas);
  }

  // ---------------------------------------------------------------- scene buffers

  buffer(data, usage = GPUBufferUsage.STORAGE) {
    const bytes = data.byteLength ? data : new Uint32Array(4);
    const size = Math.ceil(bytes.byteLength / 16) * 16;
    const buffer = this.device.createBuffer({ size, usage, mappedAtCreation: true });
    new Uint8Array(buffer.getMappedRange()).set(new Uint8Array(bytes.buffer, bytes.byteOffset, bytes.byteLength));
    buffer.unmap();
    return buffer;
  }

  upload() {
    const s = this.scene;
    const order = this.bvh.order;
    const T = order.length;
    const tris = new Float32Array(T * 12);
    const shade = new Float32Array(T * 20);
    const shadeBits = new Uint32Array(shade.buffer);
    const lo = [Infinity, Infinity, Infinity];
    const hi = [-Infinity, -Infinity, -Infinity];
    for (let j = 0; j < T; j++) {
      const i = order[j];
      const p = s.positions.subarray(i * 9, i * 9 + 9);
      for (let c = 0; c < 3; c++) {
        tris[j * 12 + c] = p[c];
        tris[j * 12 + 4 + c] = p[3 + c] - p[c];
        tris[j * 12 + 8 + c] = p[6 + c] - p[c];
        for (let k = 0; k < 3; k++) {
          lo[c] = Math.min(lo[c], p[k * 3 + c]);
          hi[c] = Math.max(hi[c], p[k * 3 + c]);
          shade[j * 20 + k * 4 + c] = s.normals[i * 9 + k * 3 + c];
        }
      }
      const uv = s.uvs.subarray(i * 6, i * 6 + 6);
      shadeBits[j * 20 + 3] = s.materialIds[i];
      shade[j * 20 + 7] = uv[0];
      shade[j * 20 + 11] = uv[1];
      shade.set(uv.subarray(2, 6), j * 20 + 12);
      for (let k = 0; k < 3; k++) shadeBits[j * 20 + 16 + k] = packColor(s.colors, i * 9 + k * 3);
    }
    this.sceneRadius = Math.hypot(hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2]) * 0.5 + 1e-3;

    const mats = new Float32Array(Math.max(s.materials.length, 1) * 12);
    s.materials.forEach((m, i) => {
      mats.set([...m.base_color, m.metallic, ...m.emission, m.roughness,
        m.transmission, m.ior, m.thin ? 1 : 0, m.texture], i * 12);
    });
    const table = new Uint32Array(Math.max(s.textures.length, 1) * 4);
    if (!s.textures.length) table.set([0, 1, 1, 0]);
    let texelCount = 0;
    s.textures.forEach((tex, i) => { table.set([texelCount, tex.width, tex.height, 0], i * 4); texelCount += tex.width * tex.height; });
    const texels = new Uint32Array(Math.max(texelCount, 1));
    let at = 0;
    for (const tex of s.textures) {
      texels.set(new Uint32Array(tex.data.buffer, tex.data.byteOffset, tex.width * tex.height), at);
      at += tex.width * tex.height;
    }

    const env = this.environment;
    const { radiance, marginal, conditional, total } = tables(env);
    let peak = 0;
    for (let i = 0; i < env.radiance.length; i++) peak = Math.max(peak, env.radiance[i]);
    this.envOn = peak * (env.intensity ?? 1) > 0;
    this.envShape = this.envOn ? [env.width, env.height] : [0, 0];
    this.envTotal = total;

    const lights = new Float32Array(Math.max(s.lights.length, 1) * 12);
    s.lights.forEach((l, i) => {
      const d = l.direction;
      const dl = Math.hypot(...d);
      lights.set([...l.position, KINDS[l.kind], ...d.map((x) => x / dl), Math.cos(l.outer_cone),
        ...l.color.map((c) => c * l.intensity), Math.cos(l.inner_cone)], i * 12);
    });

    // emissive triangles, in BVH order, with a CDF by power (luminance × area)
    const emission = s.materials.map((m) => m.emission[0] * LUM[0] + m.emission[1] * LUM[1] + m.emission[2] * LUM[2]);
    const hot = [];
    const weights = [];
    for (let j = 0; j < T; j++) {
      const e = emission[s.materialIds[order[j]]] || 0;
      if (e > 0) {
        const t = tris.subarray(j * 12, j * 12 + 12);
        const cx = t[5] * t[10] - t[6] * t[9];
        const cy = t[6] * t[8] - t[4] * t[10];
        const cz = t[4] * t[9] - t[5] * t[8];
        hot.push(j);
        weights.push(e * 0.5 * Math.hypot(cx, cy, cz));
      }
    }
    const emissive = new Float32Array(Math.max(hot.length, 1) * 4);
    const emissiveBits = new Uint32Array(emissive.buffer);
    this.emissiveTotal = weights.reduce((a, b) => a + b, 0);
    let acc = 0;
    hot.forEach((j, k) => { acc += weights[k]; emissiveBits[k * 4] = j; emissive[k * 4 + 1] = acc / this.emissiveTotal; });
    if (hot.length) emissive[(hot.length - 1) * 4 + 1] = 1;
    this.emissiveCount = this.emissiveTotal > 0 ? hot.length : 0;

    const { aux, offsets } = packAux([mats, table, texels, radiance, marginal, conditional, lights, emissive]);
    this.offsets = offsets;
    const U = GPUBufferUsage;
    this.params = this.device.createBuffer({ size: PARAMS_SIZE, usage: U.UNIFORM | U.COPY_DST });
    this.sceneBuffers = [this.params, this.buffer(this.bvh.nodes), this.buffer(tris), this.buffer(shade), this.buffer(aux)];
    this.gpuBytes = this.sceneBuffers.reduce((a, b) => a + b.size, 0);
  }

  // ---------------------------------------------------------------- size, camera, canvas

  resize(width, height) {
    if (width === this.width && height === this.height) return;
    for (const b of this.pixelBuffers || []) b.destroy();
    this.width = width;
    this.height = height;
    const n = width * height * 16;
    const U = GPUBufferUsage;
    const out = (size) => this.device.createBuffer({ size, usage: U.STORAGE | U.COPY_SRC | U.COPY_DST });
    this.accum = out(n);
    this.aov = out(2 * n);
    this.filterA = out(n);
    this.filterB = out(n);
    this.guide = out(n);
    this.scale = out(16);
    this.pixelBuffers = [this.accum, this.aov, this.filterA, this.filterB, this.guide, this.scale];
    this.groups = null;
    this.drawGroup = null;
    this.samples = 0;
  }

  setCamera(camera) {
    const c = { up: [0, 1, 0], fov: 60, ...camera };
    const same = this.camera && ['position', 'target', 'up'].every((k) =>
      c[k].every((v, i) => Math.abs(v - this.camera[k][i]) < 1e-6)) && Math.abs(c.fov - this.camera.fov) < 1e-6;
    this.camera = c;
    if (!same) this.reset();
    return !same;
  }

  reset() {
    this.samples = 0;
    const enc = this.device.createCommandEncoder();
    enc.clearBuffer(this.accum);
    enc.clearBuffer(this.aov);
    this.device.queue.submit([enc.finish()]);
  }

  attach(canvas) {
    this.canvas = canvas;
    this.context = canvas.getContext('webgpu');
    this.format = navigator.gpu.getPreferredCanvasFormat();
    this.context.configure({ device: this.device, format: this.format, alphaMode: 'opaque' });
  }

  // ---------------------------------------------------------------- pipelines

  pipeline(name, extra) {
    if (!this.pipelines[name]) {
      const C = GPUShaderStage.COMPUTE;
      const module = this.device.createShaderModule({ code: `${this.sources.common}\n${this.sources[name]}` });
      const entries = [{ binding: 0, visibility: C, buffer: { type: 'uniform' } }];
      for (let b = 1; b < 5; b++) entries.push({ binding: b, visibility: C, buffer: { type: 'read-only-storage' } });
      extra.forEach((ro, k) => entries.push({ binding: 5 + k, visibility: C, buffer: { type: ro ? 'read-only-storage' : 'storage' } }));
      const layout = this.device.createBindGroupLayout({ entries });
      const pipeline = this.device.createComputePipeline({
        layout: this.device.createPipelineLayout({ bindGroupLayouts: [layout] }),
        compute: { module, entryPoint: 'main' },
      });
      this.pipelines[name] = { pipeline, layout };
    }
    return this.pipelines[name];
  }

  bind(name, outputs, readOnly) {
    const { pipeline, layout } = this.pipeline(name, readOnly);
    const entries = [...this.sceneBuffers, ...outputs].map((buffer, binding) => ({ binding, resource: { buffer } }));
    return { pipeline, group: this.device.createBindGroup({ layout, entries }) };
  }

  bound() {
    if (!this.groups) {
      const f = [true, false, true];
      this.groups = {
        path: this.bind('path', [this.accum, this.aov], [false, false]),
        prep: this.bind('prep', [this.accum, this.aov, this.filterA, this.guide], [true, true, false, false]),
        atrous: [this.bind('atrous', [this.filterA, this.filterB, this.guide], f),
          this.bind('atrous', [this.filterB, this.filterA, this.guide], f)],
      };
    }
    return this.groups;
  }

  writeParams(frame = 0) {
    const cam = this.camera;
    const [f, r, u] = basis(cam);
    const buf = new Float32Array(PARAMS_SIZE / 4);
    const ints = new Uint32Array(buf.buffer);
    buf.set(cam.position, 0);
    buf[3] = Math.tan(cam.fov * Math.PI / 360);
    buf.set(f, 4);
    buf[7] = this.width / this.height;
    buf.set(r, 8);
    ints[11] = this.width;
    buf.set(u, 12);
    ints[15] = this.height;
    ints.set([frame, this.seed, this.maxBounces, this.scene.lights.length], 16);
    ints[20] = this.envShape[0];
    ints[21] = this.envShape[1];
    buf[22] = this.environment.intensity ?? 1;
    ints[23] = this.emissiveCount;
    const env = this.environment;
    if (env.sun_direction && env.sun_illuminance > 0) {
      const d = env.sun_direction;
      const l = Math.hypot(...d);
      const cos = Math.cos(env.sun_angular_radius);
      buf.set(d.map((x) => x / l), 24);
      buf[27] = cos;
      buf.set(env.sun_color.map((c) => c * env.sun_illuminance / (2 * Math.PI * (1 - cos))), 28);
    } else {
      buf[27] = 1;
    }
    buf[31] = this.envTotal;
    buf[32] = this.clamp;
    buf[33] = this.sceneRadius;
    buf[34] = this.emissiveTotal;
    ints[35] = this.diffuseOnly ? 1 : 0;
    ints.set(this.offsets, 36);
    this.device.queue.writeBuffer(this.params, 0, buf);
  }

  dispatch(enc, { pipeline, group }) {
    const pass = enc.beginComputePass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, group);
    pass.dispatchWorkgroups(Math.ceil(this.width / WG), Math.ceil(this.height / WG));
    pass.end();
  }

  // ---------------------------------------------------------------- tracing

  /** Queue n samples per pixel. */
  step(n = 1) {
    const { path } = this.bound();
    for (let k = 0; k < n; k++) {
      this.writeParams(this.samples);
      const enc = this.device.createCommandEncoder();
      this.dispatch(enc, path);
      this.device.queue.submit([enc.finish()]);
      this.samples++;
    }
  }

  /** Queue the denoiser: prep, then `passes` à-trous passes; returns the buffer holding the result. */
  denoise(passes = 5) {
    const { prep, atrous } = this.bound();
    this.writeParams();
    let enc = this.device.createCommandEncoder();
    this.dispatch(enc, prep);
    this.device.queue.submit([enc.finish()]);
    for (let k = 0; k < passes; k++) {
      this.writeParams(1 << k);
      enc = this.device.createCommandEncoder();
      this.dispatch(enc, atrous[k % 2]);
      this.device.queue.submit([enc.finish()]);
    }
    return passes % 2 ? this.filterB : this.filterA;
  }

  /** Draw the current image on the attached canvas. */
  draw({ denoise = true } = {}) {
    if (!this.context) throw new Error('no canvas: pass one to createTracer or call attach(canvas)');
    const texture = this.context.getCurrentTexture();
    this.render(texture.createView(), this.format, this.canvas.width, this.canvas.height, denoise);
  }

  /** The current image as {width, height, data: Uint8Array RGBA sRGB}, as Tracer.image(). Needs no
   * canvas: it draws into a texture of the trace's size and reads it back. */
  async image({ denoise = true } = {}) {
    const U = GPUTextureUsage;
    const texture = this.device.createTexture({ size: [this.width, this.height], format: 'rgba8unorm',
      usage: U.RENDER_ATTACHMENT | U.COPY_SRC });
    this.render(texture.createView(), 'rgba8unorm', this.width, this.height, denoise);
    const row = Math.ceil(this.width * 4 / 256) * 256;
    const staging = this.device.createBuffer({ size: row * this.height, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    const enc = this.device.createCommandEncoder();
    enc.copyTextureToBuffer({ texture }, { buffer: staging, bytesPerRow: row }, [this.width, this.height]);
    this.device.queue.submit([enc.finish()]);
    await staging.mapAsync(GPUMapMode.READ);
    const padded = new Uint8Array(staging.getMappedRange());
    const data = new Uint8Array(this.width * this.height * 4);
    for (let y = 0; y < this.height; y++) data.set(padded.subarray(y * row, y * row + this.width * 4), y * this.width * 4);
    staging.destroy();
    texture.destroy();
    return { width: this.width, height: this.height, data };
  }

  render(target, format, outWidth, outHeight, denoise) {
    const filtered = denoise ? this.denoise() : this.filterB;
    if (!this.display) {
      const module = this.device.createShaderModule({ code: this.sources.display });
      const C = GPUShaderStage.COMPUTE;
      const entries = [{ binding: 0, visibility: C, buffer: { type: 'uniform' } }];
      for (let b = 1; b < 4; b++) entries.push({ binding: b, visibility: C, buffer: { type: 'read-only-storage' } });
      const exposeLayout = this.device.createBindGroupLayout({ entries: [...entries,
        { binding: 4, visibility: C, buffer: { type: 'storage' } }] });
      // the fragment stage reads no storage it could write: its layout stops at binding 3
      const drawLayout = this.device.createBindGroupLayout({ entries: entries.map((e) => ({ ...e, visibility: GPUShaderStage.FRAGMENT })) });
      this.display = {
        module, drawLayout, exposeLayout, draws: {},
        view: this.device.createBuffer({ size: VIEW_SIZE, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST }),
        expose: this.device.createComputePipeline({
          layout: this.device.createPipelineLayout({ bindGroupLayouts: [exposeLayout] }),
          compute: { module, entryPoint: 'expose' } }),
      };
    }
    const d = this.display;
    if (!d.draws[format]) {
      d.draws[format] = this.device.createRenderPipeline({
        layout: this.device.createPipelineLayout({ bindGroupLayouts: [d.drawLayout] }),
        vertex: { module: d.module, entryPoint: 'vs' },
        fragment: { module: d.module, entryPoint: 'fs', targets: [{ format }] } });
    }
    if (!this.drawGroup || this.drawGroup.filtered !== filtered) {
      const entries = [d.view, this.accum, this.aov, filtered, this.scale].map((buffer, binding) => ({ binding, resource: { buffer } }));
      this.drawGroup = { filtered, expose: this.device.createBindGroup({ layout: d.exposeLayout, entries }),
        draw: this.device.createBindGroup({ layout: d.drawLayout, entries: entries.slice(0, 4) }) };
    }
    const view = new ArrayBuffer(VIEW_SIZE);
    const f = new Float32Array(view);
    const u = new Uint32Array(view);
    u[0] = this.width;
    u[1] = this.height;
    u[2] = denoise ? 1 : 0;
    f[3] = this.exposure === 'auto' ? 0 : this.exposure;
    f[4] = outWidth;
    f[5] = outHeight;
    u[6] = TONEMAPS[this.tonemap];
    this.device.queue.writeBuffer(d.view, 0, view);
    const enc = this.device.createCommandEncoder();
    const cp = enc.beginComputePass();
    cp.setPipeline(d.expose);
    cp.setBindGroup(0, this.drawGroup.expose);
    cp.dispatchWorkgroups(1);
    cp.end();
    enc.copyBufferToBuffer(this.scale, 0, d.view, 12, 4);
    const rp = enc.beginRenderPass({ colorAttachments: [{ view: target,
      loadOp: 'clear', storeOp: 'store', clearValue: { r: 0, g: 0, b: 0, a: 1 } }] });
    rp.setPipeline(d.draws[format]);
    rp.setBindGroup(0, this.drawGroup.draw);
    rp.draw(3);
    rp.end();
    this.device.queue.submit([enc.finish()]);
  }

  /** As many samples as fit in budget ms (one at first), then a denoised draw. */
  async frame({ budget = 30, maxPerFrame = 16, denoise = true } = {}) {
    const n = this.msPerSample ? Math.max(1, Math.min(maxPerFrame, Math.floor(budget / this.msPerSample))) : 1;
    const t = performance.now();
    this.step(n);
    await this.device.queue.onSubmittedWorkDone();
    const ms = (performance.now() - t) / n;
    this.msPerSample = this.msPerSample ? 0.7 * this.msPerSample + 0.3 * ms : ms;
    if (this.context) this.draw({ denoise });
    return n;
  }

  /** Refine on the canvas every animation frame until stop() or maxSamples. onFrame(tracer) runs after each. */
  start({ maxSamples = 1024, budget = 30, onFrame = null } = {}) {
    this.stop();
    const run = { on: true };
    this.running = run;
    const tick = async () => {
      if (!run.on) return;
      if (this.samples < maxSamples) {
        await this.frame({ budget });
        if (onFrame) onFrame(this);
      }
      if (run.on && this.samples < maxSamples) requestAnimationFrame(tick);
      else run.on = false;
    };
    requestAnimationFrame(tick);
  }

  stop() {
    if (this.running) this.running.on = false;
    this.running = null;
  }

  // ---------------------------------------------------------------- reading back (tests, tools)

  async read(buffer) {
    const staging = this.device.createBuffer({ size: buffer.size, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    const enc = this.device.createCommandEncoder();
    enc.copyBufferToBuffer(buffer, 0, staging, 0, buffer.size);
    this.device.queue.submit([enc.finish()]);
    await staging.mapAsync(GPUMapMode.READ);
    const out = new Float32Array(staging.getMappedRange().slice(0));
    staging.destroy();
    return out;
  }

  /** Mean linear radiance per pixel, Float32Array (h*w*3), as Tracer.radiance(). */
  async radiance() {
    const acc = await this.read(this.accum);
    const out = new Float32Array(this.width * this.height * 3);
    for (let i = 0; i < this.width * this.height; i++) {
      const n = Math.max(acc[i * 4 + 3], 1);
      for (let c = 0; c < 3; c++) out[i * 3 + c] = acc[i * 4 + c] / n;
    }
    return out;
  }

  /** Linear radiance after the à-trous filter, as Tracer.denoised(). */
  async denoised(passes = 5) {
    const filtered = await this.read(this.denoise(passes));
    const acc = await this.read(this.accum);
    const aov = await this.read(this.aov);
    const out = new Float32Array(this.width * this.height * 3);
    for (let i = 0; i < this.width * this.height; i++) {
      const n = Math.max(acc[i * 4 + 3], 1);
      for (let c = 0; c < 3; c++) out[i * 3 + c] = filtered[i * 4 + c] * Math.max(aov[i * 8 + c] / n, 0.02);
    }
    return out;
  }

  destroy() {
    this.stop();
    for (const b of [...this.sceneBuffers, ...this.pixelBuffers]) b.destroy();
  }
}

/** Concatenate tables into one Float32Array of vec4s, each padded to whole vec4s, and the vec4
 * offset of each; as tracer.pack_aux. Uint32 tables are stored as their bits. */
export function packAux(tablesIn) {
  const sizes = tablesIn.map((t) => Math.ceil(t.length / 4) * 4);
  const aux = new Float32Array(sizes.reduce((a, b) => a + b, 0));
  const bits = new Uint32Array(aux.buffer);
  const offsets = [];
  let at = 0;
  tablesIn.forEach((t, k) => {
    offsets.push(at / 4);
    if (t instanceof Uint32Array) bits.set(t, at); else aux.set(t, at);
    at += sizes[k];
  });
  return { aux, offsets };
}

function packColor(colors, i) {
  const q = (v, m) => Math.round(Math.min(Math.max(v, 0), 1) * m);
  return (q(colors[i], 2047) | (q(colors[i + 1], 2047) << 11) | (q(colors[i + 2], 1023) << 22)) >>> 0;
}

/** (forward, right, up) unit vectors, as Camera.basis. */
export function basis(cam) {
  const norm = (v) => { const l = Math.hypot(...v); return v.map((x) => x / l); };
  const cross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
  const f = norm(cam.target.map((t, i) => t - cam.position[i]));
  let r = cross(f, cam.up || [0, 1, 0]);
  if (Math.hypot(...r) < 1e-9) r = cross(f, [0, 0, 1]);
  r = norm(r);
  return [f, r, cross(r, f)];
}

function defaultCamera(scene) {
  const p = scene.positions;
  const lo = [Infinity, Infinity, Infinity];
  const hi = [-Infinity, -Infinity, -Infinity];
  for (let i = 0; i < p.length; i += 3) for (let c = 0; c < 3; c++) { lo[c] = Math.min(lo[c], p[i + c]); hi[c] = Math.max(hi[c], p[i + c]); }
  const center = lo.map((l, c) => (l + hi[c]) / 2);
  const r = Math.hypot(hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2]) / 2;
  return { position: center.map((c, k) => c + [0.6, 0.5, 1.0][k] * r * 1.6), target: center, up: [0, 1, 0], fov: 50 };
}
