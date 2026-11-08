// Read a binary glTF (.glb) into a trace scene: every mesh instance baked to world space.
// A port of manifoldx/trace/gltf_read.py and scene.assemble, so the page traces what Python does.
//
//   const scene = await loadGlb(url, { overrides, hide });
//
// overrides maps a node or material name (glob) to Material fields, named as in Python
// ({"glass": {"transmission": 1, "ior": 1.5, "thin": true}}); node names win, and
// {"texture": null} drops a texture. hide lists node-name globs whose subtrees are left out.

export const MAX_TEXTURE = 2048;

const COMPONENTS = { 5120: Int8Array, 5121: Uint8Array, 5122: Int16Array, 5123: Uint16Array,
  5125: Uint32Array, 5126: Float32Array };
const WIDTH = { SCALAR: 1, VEC2: 2, VEC3: 3, VEC4: 4, MAT4: 16 };
const NORM = { 5120: 127, 5121: 255, 5122: 32767, 5123: 65535 };
const FIELDS = ['name', 'base_color', 'alpha', 'metallic', 'roughness', 'emission', 'transmission',
  'ior', 'thin', 'texture', 'vertex_colors'];

export function material(fields = {}) {
  return { name: '', base_color: [0.8, 0.8, 0.8], alpha: 1, metallic: 0, roughness: 1,
    emission: [0, 0, 0], transmission: 0, ior: 1.5, thin: true, texture: -1, vertex_colors: false,
    ...fields };
}

/** fnmatch.fnmatchcase: *, ?, [seq] and [!seq]. */
export function globRegex(pattern) {
  let re = '';
  for (let i = 0; i < pattern.length; i++) {
    const c = pattern[i];
    if (c === '*') re += '.*';
    else if (c === '?') re += '.';
    else if (c === '[') {
      const end = pattern.indexOf(']', i + 2);
      if (end < 0) { re += '\\['; continue; }
      let body = pattern.slice(i + 1, end).replace(/\\/g, '\\\\');
      if (body[0] === '!') body = '^' + body.slice(1);
      else if (body[0] === '^') body = '\\' + body;
      re += `[${body}]`;
      i = end;
    } else re += c.replace(/[.+^${}()|\\/\]-]/g, '\\$&');
  }
  return new RegExp(`^${re}$`, 's');
}

export async function loadGlb(url, options = {}) {
  const response = await fetch(url);
  if (!response.ok) throw new Error(`${url}: HTTP ${response.status}`);
  return readGlb(await response.arrayBuffer(), options);
}

export function parseGlb(buffer) {
  const view = new DataView(buffer);
  if (view.getUint32(0, true) !== 0x46546c67) throw new Error('not a binary glTF file');
  if (view.getUint32(4, true) !== 2) throw new Error('only glTF 2 is read');
  let doc = null;
  let bin = new Uint8Array(0);
  for (let off = 12; off < buffer.byteLength;) {
    const size = view.getUint32(off, true);
    const kind = view.getUint32(off + 4, true);
    const chunk = new Uint8Array(buffer, off + 8, size);
    if (kind === 0x4e4f534a) doc = JSON.parse(new TextDecoder().decode(chunk));
    else if (kind === 0x004e4942) bin = chunk;
    off += 8 + size;
  }
  if (!doc) throw new Error('glb has no JSON chunk');
  return { doc, bin };
}

export async function readGlb(buffer, { overrides = null, hide = [] } = {}) {
  const { doc, bin } = parseGlb(buffer);
  const reader = new Reader(doc, bin);
  await reader.loadTextures();
  return reader.scene(overrides, (hide || []).map(globRegex));
}

class Reader {
  constructor(doc, bin) {
    this.doc = doc;
    this.bin = bin;
    this.textures = [];
    this.textureOf = new Map();
  }

  view(index) {
    const v = this.doc.bufferViews[index];
    if ((v.buffer ?? 0) !== 0) throw new Error("only the glb's own binary buffer is read");
    const start = this.bin.byteOffset + (v.byteOffset ?? 0);
    return { bytes: new Uint8Array(this.bin.buffer, start, v.byteLength), stride: v.byteStride };
  }

  /** Float64 rows of an accessor, normalized integers scaled as glTF says. */
  accessor(index) {
    const a = this.doc.accessors[index];
    const Type = COMPONENTS[a.componentType];
    const width = WIDTH[a.type];
    const out = new Float64Array(a.count * width);
    out.width = width;
    if (a.bufferView === undefined) return out;
    const { bytes, stride } = this.view(a.bufferView);
    const item = Type.BYTES_PER_ELEMENT * width;
    const step = stride || item;
    const base = bytes.byteOffset + (a.byteOffset ?? 0);
    const copy = new Uint8Array(item);
    const typed = new Type(copy.buffer);
    for (let i = 0; i < a.count; i++) {
      copy.set(new Uint8Array(bytes.buffer, base + i * step, item));
      for (let k = 0; k < width; k++) out[i * width + k] = typed[k];
    }
    const norm = a.normalized && NORM[a.componentType];
    if (norm) for (let i = 0; i < out.length; i++) out[i] = Math.max(out[i] / norm, -1);
    return out;
  }

  async loadTextures() {
    const used = new Set();
    for (const m of this.doc.materials || []) {
      const tex = m.pbrMetallicRoughness?.baseColorTexture;
      if (tex) used.add(tex.index);
    }
    for (const index of [...used].sort((a, b) => a - b)) await this.texture(index);
    this.materials = (this.doc.materials || []).map((m) => this.material(m));
  }

  async texture(index) {
    const source = this.doc.textures[index].source;
    if (source === undefined) { this.textureOf.set(index, -1); return; }
    const image = this.doc.images[source];
    const { bytes } = this.view(image.bufferView);
    const blob = new Blob([bytes], { type: image.mimeType || 'image/png' });
    const bitmap = await createImageBitmap(blob, { colorSpaceConversion: 'none', premultiplyAlpha: 'none' });
    let w = bitmap.width;
    let h = bitmap.height;
    if (Math.max(w, h) > MAX_TEXTURE) {
      const scale = MAX_TEXTURE / Math.max(w, h);
      w = Math.max(1, Math.floor(w * scale));
      h = Math.max(1, Math.floor(h * scale));
    }
    const canvas = new OffscreenCanvas(w, h);
    const ctx = canvas.getContext('2d', { colorSpace: 'srgb' });
    ctx.drawImage(bitmap, 0, 0, w, h);
    const data = new Uint8Array(ctx.getImageData(0, 0, w, h).data.buffer);
    this.textures.push({ width: w, height: h, data });
    this.textureOf.set(index, this.textures.length - 1);
  }

  material(m) {
    const pbr = m.pbrMetallicRoughness || {};
    const ext = m.extensions || {};
    const base = pbr.baseColorFactor || [1, 1, 1, 1];
    const strength = ext.KHR_materials_emissive_strength?.emissiveStrength ?? 1;
    let transmission = ext.KHR_materials_transmission?.transmissionFactor ?? 0;
    const alpha = base[3];
    if (m.alphaMode === 'BLEND' && alpha < 1) transmission = Math.max(transmission, 1 - alpha);
    const tex = pbr.baseColorTexture;
    return material({
      name: m.name || '', base_color: base.slice(0, 3), alpha,
      metallic: pbr.metallicFactor ?? 1, roughness: pbr.roughnessFactor ?? 1,
      emission: (m.emissiveFactor || [0, 0, 0]).map((c) => c * strength), transmission,
      ior: ext.KHR_materials_ior?.ior ?? 1.5,
      thin: (ext.KHR_materials_volume?.thicknessFactor ?? 0) <= 0,
      texture: tex ? this.textureOf.get(tex.index) ?? -1 : -1,
      vertex_colors: m.extras?.mx?.vertex_colors ?? true,
    });
  }

  scene(overrides, hide) {
    const doc = this.doc;
    const nodes = doc.nodes || [];
    let roots;
    if (doc.scenes) roots = doc.scenes[doc.scene ?? 0].nodes || [];
    else {
      const children = new Set(nodes.flatMap((n) => n.children || []));
      roots = nodes.map((_, i) => i).filter((i) => !children.has(i));
    }
    const lightDefs = doc.extensions?.KHR_lights_punctual?.lights || [];
    const parts = [];
    const lights = [];
    let camera = null;
    const stack = roots.map((i) => [i, IDENTITY]);
    while (stack.length) {
      const [i, parent] = stack.pop();
      const node = nodes[i];
      if (hide.some((re) => re.test(node.name || ''))) continue;
      const world = mul(parent, local(node));
      for (const c of node.children || []) stack.push([c, world]);
      if (node.mesh !== undefined && node.skin === undefined) parts.push(...this.mesh(node, world));
      if (node.camera !== undefined && camera === null) camera = cameraOf(doc.cameras[node.camera], world);
      const light = node.extensions?.KHR_lights_punctual;
      if (light) lights.push(lightOf(lightDefs[light.light], world));
    }
    return assemble(parts, this.materials, { textures: this.textures, camera, lights,
      environment: this.environment(), overrides });
  }

  environment() {
    const scenes = this.doc.scenes || [{}];
    const env = scenes[this.doc.scene ?? 0]?.extras?.manifoldx?.environment;
    if (!env || env.bufferView === undefined) return null;
    const { bytes } = this.view(env.bufferView);
    const radiance = new Float32Array(bytes.slice().buffer, 0, env.width * env.height * 3);
    return { radiance, width: env.width, height: env.height, intensity: env.intensity ?? 1 };
  }

  * mesh(node, world) {
    const name = node.name || '';
    const normalM = transpose3(inverse3(world));
    const flip = det3(world) < 0;
    for (const prim of this.doc.meshes[node.mesh].primitives) {
      if ((prim.mode ?? 4) !== 4) continue;
      const attrs = prim.attributes;
      const p = this.accessor(attrs.POSITION);
      const pos = new Float64Array(p.length);
      for (let i = 0; i < p.length; i += 3) {
        for (let r = 0; r < 3; r++) {
          pos[i + r] = world[r] * p[i] + world[4 + r] * p[i + 1] + world[8 + r] * p[i + 2] + world[12 + r];
        }
      }
      let tri;
      if (prim.indices !== undefined) tri = Uint32Array.from(this.accessor(prim.indices));
      else tri = Uint32Array.from({ length: p.length / 3 }, (_, i) => i);
      if (flip) for (let i = 0; i < tri.length; i += 3) [tri[i], tri[i + 2]] = [tri[i + 2], tri[i]];
      const out = {};
      if (attrs.NORMAL !== undefined) {
        const n = this.accessor(attrs.NORMAL);
        out.normals = new Float64Array(n.length);
        for (let i = 0; i < n.length; i += 3) {
          for (let r = 0; r < 3; r++) {
            out.normals[i + r] = normalM[r] * n[i] + normalM[3 + r] * n[i + 1] + normalM[6 + r] * n[i + 2];
          }
        }
      }
      if (attrs.TEXCOORD_0 !== undefined) out.uvs = this.accessor(attrs.TEXCOORD_0);
      const mat = prim.material ?? -1;
      if (attrs.COLOR_0 !== undefined && (mat < 0 || this.materials[mat].vertex_colors)) {
        out.colors = this.accessor(attrs.COLOR_0);
      }
      yield { name, pos, tri, attrs: out, mat };
    }
  }
}

// ------------------------------------------------------------------ assembling the scene

/** parts: {name, pos (V*3), tri (T*3), attrs: {normals, uvs, colors}, mat}; as scene.assemble. */
export function assemble(parts, materials, { textures = [], camera = null, lights = [], environment = null,
  overrides = null } = {}) {
  materials = materials.map((m) => ({ ...m }));
  const derived = new Map();
  let total = 0;
  for (const part of parts) total += part.tri.length / 3;
  if (!total) throw new Error('the scene has no triangles');
  const positions = new Float32Array(total * 9);
  const normals = new Float32Array(total * 9);
  const uvs = new Float32Array(total * 6);
  const colors = new Float32Array(total * 9).fill(1);
  const materialIds = new Int32Array(total);
  const nodeNames = [];
  let t = 0;
  for (const { name, pos, tri, attrs, mat: partMat } of parts) {
    let mat = partMat;
    if (mat === undefined || mat === null || mat < 0) {
      materials.push(material({ name: 'default' }));
      mat = materials.length - 1;
    }
    mat = override(materials, derived, mat, name, overrides);
    const first = t;
    const cw = attrs.colors ? attrs.colors.width : 0;
    for (let f = 0; f < tri.length; f += 3, t++) {
      for (let k = 0; k < 3; k++) {
        const v = tri[f + k];
        for (let c = 0; c < 3; c++) positions[t * 9 + k * 3 + c] = pos[v * 3 + c];
      }
      const o = t * 9;
      const e1 = [positions[o + 3] - positions[o], positions[o + 4] - positions[o + 1], positions[o + 5] - positions[o + 2]];
      const e2 = [positions[o + 6] - positions[o], positions[o + 7] - positions[o + 1], positions[o + 8] - positions[o + 2]];
      let fn = [e1[1] * e2[2] - e1[2] * e2[1], e1[2] * e2[0] - e1[0] * e2[2], e1[0] * e2[1] - e1[1] * e2[0]];
      const fl = Math.max(Math.hypot(...fn), 1e-20);
      fn = fn.map((x) => x / fl);
      for (let k = 0; k < 3; k++) {
        const v = tri[f + k];
        let n = fn;
        if (attrs.normals) {
          const raw = [attrs.normals[v * 3], attrs.normals[v * 3 + 1], attrs.normals[v * 3 + 2]];
          const l = Math.max(Math.hypot(...raw), 1e-20);
          const unit = raw.map((x) => x / l);
          if (unit.every(Number.isFinite) && Math.hypot(...unit) >= 0.5) n = unit;
        }
        for (let c = 0; c < 3; c++) normals[o + k * 3 + c] = n[c];
        if (attrs.uvs) { uvs[t * 6 + k * 2] = attrs.uvs[v * 2]; uvs[t * 6 + k * 2 + 1] = attrs.uvs[v * 2 + 1]; }
        if (attrs.colors) for (let c = 0; c < 3; c++) colors[o + k * 3 + c] = attrs.colors[v * cw + c];
      }
      materialIds[t] = mat;
    }
    nodeNames.push([name, first, t - first]);
  }
  return { positions, normals, uvs, colors, materialIds, materials, textures, camera, lights, environment,
    nodeNames, triangleCount: total };
}

function override(materials, derived, mat, nodeName, overrides) {
  if (!overrides) return mat;
  const changes = {};
  const entries = Object.entries(overrides).map(([p, v]) => [globRegex(p), v]);
  for (const [re, values] of entries) if (re.test(materials[mat].name || '')) Object.assign(changes, values);
  for (const [re, values] of entries) if (nodeName && re.test(nodeName)) Object.assign(changes, values);
  const keys = Object.keys(changes);
  if (!keys.length) return mat;
  const unknown = keys.filter((k) => !FIELDS.includes(k));
  if (unknown.length) throw new Error(`unknown material fields in overrides: ${unknown.sort().join(', ')}`);
  if ('texture' in changes && changes.texture === null) changes.texture = -1;
  const key = `${mat}|${JSON.stringify(keys.sort().map((k) => [k, changes[k]]))}`;
  if (!derived.has(key)) {
    materials.push({ ...materials[mat], ...changes });
    derived.set(key, materials.length - 1);
  }
  return derived.get(key);
}

// ------------------------------------------------------------------ 4x4 matrices, column-major

const IDENTITY = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1];

function mul(a, b) {
  const out = new Array(16);
  for (let c = 0; c < 4; c++) {
    for (let r = 0; r < 4; r++) {
      out[c * 4 + r] = a[r] * b[c * 4] + a[4 + r] * b[c * 4 + 1] + a[8 + r] * b[c * 4 + 2] + a[12 + r] * b[c * 4 + 3];
    }
  }
  return out;
}

function local(node) {
  if (node.matrix) return node.matrix.slice();
  const [x, y, z, w] = node.rotation || [0, 0, 0, 1];
  const [sx, sy, sz] = node.scale || [1, 1, 1];
  const [tx, ty, tz] = node.translation || [0, 0, 0];
  return [
    (1 - 2 * (y * y + z * z)) * sx, (2 * (x * y + z * w)) * sx, (2 * (x * z - y * w)) * sx, 0,
    (2 * (x * y - z * w)) * sy, (1 - 2 * (x * x + z * z)) * sy, (2 * (y * z + x * w)) * sy, 0,
    (2 * (x * z + y * w)) * sz, (2 * (y * z - x * w)) * sz, (1 - 2 * (x * x + y * y)) * sz, 0,
    tx, ty, tz, 1,
  ];
}

// the upper 3x3 of a column-major 4x4, as a column-major 3x3
const m3 = (m) => [m[0], m[1], m[2], m[4], m[5], m[6], m[8], m[9], m[10]];

function det3(m4) {
  const [a, b, c, d, e, f, g, h, i] = m3(m4);
  return a * (e * i - f * h) - d * (b * i - c * h) + g * (b * f - c * e);
}

function inverse3(m4) {
  const [a, b, c, d, e, f, g, h, i] = m3(m4);  // columns (a b c), (d e f), (g h i)
  const det = det3(m4);
  return [
    (e * i - f * h) / det, (c * h - b * i) / det, (b * f - c * e) / det,
    (f * g - d * i) / det, (a * i - c * g) / det, (c * d - a * f) / det,
    (d * h - e * g) / det, (b * g - a * h) / det, (a * e - b * d) / det,
  ];
}

function transpose3(m) {
  return [m[0], m[3], m[6], m[1], m[4], m[7], m[2], m[5], m[8]];
}

const unit = (v) => { const l = Math.hypot(...v); return v.map((x) => x / l); };

function cameraOf(cam, world) {
  const pos = [world[12], world[13], world[14]];
  const forward = unit([-world[8], -world[9], -world[10]]);
  const up = unit([world[4], world[5], world[6]]);
  const fov = (cam.perspective?.yfov ?? Math.PI / 3) * 180 / Math.PI;
  return { position: pos, target: pos.map((p, k) => p + forward[k]), up, fov };
}

function lightOf(d, world) {
  const spot = d.spot || {};
  return { kind: d.type, color: d.color || [1, 1, 1], intensity: d.intensity ?? 1,
    position: [world[12], world[13], world[14]], direction: unit([-world[8], -world[9], -world[10]]),
    inner_cone: spot.innerConeAngle ?? 0, outer_cone: spot.outerConeAngle ?? Math.PI / 4 };
}
