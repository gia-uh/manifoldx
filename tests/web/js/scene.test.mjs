// The exported scene in three.js, in Node: the vendored GLTFLoader parses web_walk's
// scene.glb (untextured, so Node's globals suffice) and buildScene runs on a stub renderer.

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { register } from "node:module";
import test from "node:test";

import { createEngine } from "../../../src/manifoldx/web/js/manifoldx.js";

// The bare "three" and "three/addons/" of the vendored files resolve to the vendor folder, as
// the page's importmap does.
const VENDOR = new URL("../../../src/manifoldx/web/vendor/three/", import.meta.url).href;
const HOOKS = `
const V = ${JSON.stringify(VENDOR)};
export async function resolve(spec, ctx, next) {
  if (spec === "three") return { url: V + "three.module.js", shortCircuit: true };
  if (spec.startsWith("three/addons/")) return { url: V + spec.slice(6), shortCircuit: true };
  return next(spec, ctx);
}`;
register("data:text/javascript," + encodeURIComponent(HOOKS));

const THREE = await import("three");
const { GLTFLoader } = await import("three/addons/loaders/GLTFLoader.js");
const { CALIBRATION, buildScene, equirectTexture } = await import("../../../src/manifoldx/web/js/scene.js");

// The folder export --web wrote for examples/web_walk.py, set by tests/web/test_node_suites.py.
const WEB_WALK = process.env.MX_WEB_WALK_EXPORT;
const skip = !WEB_WALK && "MX_WEB_WALK_EXPORT not set";

const stubRenderer = () => ({ capabilities: { maxTextureSize: 4096 }, shadowMap: { enabled: false } });

async function webWalk() {
  const buf = readFileSync(`${WEB_WALK}/scene.glb`);
  const gltf = await new GLTFLoader().parseAsync(buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength), "./");
  const state = JSON.parse(readFileSync(`${WEB_WALK}/state.json`));
  return { gltf, state, mx: await buildScene(THREE, gltf, stubRenderer()) };
}

// Every node the exporter marked with extras.mx, and the meshes under it.
function marked(root) {
  const out = new Map();
  root.traverse((o) => {
    if (!o.userData.mx) return;
    const meshes = [];
    o.traverse((m) => { if (m.isMesh) meshes.push(m); });
    out.set(o.userData.mx.entity, { meshes, material: o.userData.mx.material });
  });
  return out;
}

test("CALIBRATION holds the factors the visual parity test measured", () => {
  assert.deepEqual(CALIBRATION, { environment: 1.45, sun: 1.0 });
});

test("entities holds every mx.entity and the meshes under each", { skip }, async () => {
  const { gltf, mx } = await webWalk();
  const nodes = marked(gltf.scene);
  assert.equal(nodes.size, 7);
  assert.deepEqual([...mx.entities.keys()].sort((a, b) => a - b), [...nodes.keys()].sort((a, b) => a - b));
  for (const [entity, { meshes }] of nodes) {
    assert.ok(meshes.length > 0);
    assert.deepEqual(mx.entities.get(entity), meshes);
  }
});

test("materials holds every id of mx_materials, gold included; setMaterial swaps them", { skip }, async () => {
  const { gltf, mx } = await webWalk();
  const ids = Object.keys(gltf.scene.userData.manifoldx.mx_materials).map(Number);
  assert.deepEqual([...mx.materials.keys()].sort((a, b) => a - b), ids.sort((a, b) => a - b));
  const nodes = marked(gltf.scene);
  const used = new Set([...nodes.values()].map((n) => n.material));
  const gold = ids.find((id) => !used.has(id));
  assert.ok(gold !== undefined, "web_walk has a material no entity uses at export");
  for (const [entity, { meshes, material }] of nodes) {
    assert.equal(mx.materialOf(entity), material);
    for (const m of meshes) assert.equal(m.material, mx.materials.get(material));
  }
  const box = [...nodes.keys()].at(-1);
  mx.setMaterial(box, gold);
  for (const m of mx.entities.get(box)) assert.equal(m.material, mx.materials.get(gold));
  assert.equal(mx.materialOf(box), gold);
  assert.equal(mx.materialOf(9999), null);
  assert.throws(() => mx.setMaterial(box, 9999), /material 9999/);
});

test("BasicMaterial and PhongMaterial become the mx basic material, same linear colour, no fog", { skip }, async () => {
  const { gltf, mx } = await webWalk();
  const json = gltf.parser.json;
  for (const [id, index] of Object.entries(gltf.scene.userData.manifoldx.mx_materials)) {
    const written = json.materials[index];
    assert.ok(["BasicMaterial", "PhongMaterial"].includes(written.name));
    const m = mx.materials.get(Number(id));
    assert.ok(m.isMeshBasicMaterial);
    assert.equal(m.userData.mxBasic, true);
    assert.equal(m.fog, false);
    const [r, g, b] = written.pbrMetallicRoughness.baseColorFactor;
    assert.deepEqual([m.color.r, m.color.g, m.color.b].map(Math.fround), [r, g, b].map(Math.fround));
  }
});

test("the mx basic material lights the colour as the desktop's fixed light", { skip }, async () => {
  const { mx } = await webWalk();
  const m = mx.materials.values().next().value;
  const shader = { vertexShader: THREE.ShaderLib.basic.vertexShader, fragmentShader: THREE.ShaderLib.basic.fragmentShader, uniforms: {} };
  m.onBeforeCompile(shader);
  assert.match(shader.vertexShader, /vMxNormal = normalize\( mat3\( modelMatrix \) \* normal \);/);
  assert.match(shader.fragmentShader, /diffuseColor\.rgb \*= 0\.3 \+ 0\.7 \* max\( dot\( normalize\( vMxNormal \), normalize\( vec3\( 1\.0 \) \) \), 0\.0 \);/);
  // The brightness comes after the colour is read and before the output.
  const f = shader.fragmentShader;
  assert.ok(f.indexOf("vMxNormal ), normalize") > f.indexOf("#include <color_fragment>"));
  assert.ok(f.indexOf("vMxNormal ), normalize") < f.indexOf("#include <opaque_fragment>"));
});

test("background is the extras' linear RGB; no environment, fog or sun in web_walk", { skip }, async () => {
  const { gltf, mx } = await webWalk();
  const [r, g, b] = gltf.scene.userData.manifoldx.background;
  assert.deepEqual([mx.scene.background.r, mx.scene.background.g, mx.scene.background.b], [r, g, b]);
  assert.equal(mx.scene.environment, null);
  assert.equal(mx.scene.fog, null);
  assert.ok(mx.scene.children.includes(gltf.scene));
});

test("sync puts the engine's camera on the three camera", { skip }, async () => {
  const { state, mx } = await webWalk();
  const engine = createEngine({ ...state.engine, camera: { ...state.engine.camera, fov: 50, up: [0, 0.8, 0.6], near: 0.2, far: 300 } });
  engine.camera.position.set([1.5, 2.25, -3]);
  engine.camera.target.set([4, 1, 2.5]);
  engine.w = 640;
  engine.h = 480;
  mx.sync(engine);
  const cam = mx.camera;
  assert.ok(cam.isPerspectiveCamera);
  assert.deepEqual(cam.position.toArray(), Array.from(engine.camera.position));
  assert.deepEqual(cam.up.toArray(), Array.from(engine.camera.up));
  assert.deepEqual([cam.fov, cam.near, cam.far, cam.aspect], [50, 0.2, 300, 640 / 480]);
  const dir = cam.getWorldDirection(new THREE.Vector3()).toArray();
  const forward = engine.camera.get_forward();
  for (let i = 0; i < 3; i++) assert.ok(Math.abs(dir[i] - forward[i]) < 1e-6, `${dir} != ${forward}`);
  // The projection follows the aspect.
  assert.equal(cam.projectionMatrix.elements[0], cam.projectionMatrix.elements[5] / (640 / 480));
});

// ---------------------------------------------------------------- a lit scene

// What the loader returns for a ground, a box and a sun, with the scene extras given: a
// stand-in for a glb with fog and shadows, which web_walk has not.
function litGltf(extras) {
  const root = new THREE.Group();
  root.userData.manifoldx = { background: [0.5, 0.6, 0.7], mx_materials: {}, ...extras };
  const ground = new THREE.Mesh(new THREE.BoxGeometry(20, 0.2, 20), new THREE.MeshStandardMaterial());
  const box = new THREE.Mesh(new THREE.BoxGeometry(1, 1, 1), new THREE.MeshStandardMaterial());
  box.position.set(3, 1, 0);
  const sun = new THREE.DirectionalLight(0xffffff, 3);
  sun.rotation.set(-Math.PI / 3, Math.PI / 4, 0);
  sun.target.position.set(0, 0, -1);
  sun.add(sun.target);
  root.add(ground, box, sun);
  return { scene: root, parser: { json: { materials: [] }, getDependency: async () => { throw new Error("no dependency"); } } };
}

const sunOf = (scene) => {
  let light = null;
  scene.traverse((o) => { if (!light && o.isDirectionalLight) light = o; });
  return light;
};

test("fog is THREE.Fog from start, end and the linear colour", async () => {
  const mx = await buildScene(THREE, litGltf({ fog: { start: 20, end: 60, color: [0.25, 0.5, 0.75] } }), stubRenderer());
  assert.ok(mx.scene.fog.isFog);
  assert.deepEqual([mx.scene.fog.near, mx.scene.fog.far], [20, 60]);
  assert.deepEqual(mx.scene.fog.color.toArray(), [0.25, 0.5, 0.75]);
});

test("the sun is scaled by CALIBRATION.sun and setCalibration rescales it", async () => {
  const mx = await buildScene(THREE, litGltf({}), stubRenderer());
  const sun = sunOf(mx.scene);
  assert.equal(sun.intensity, 3 * CALIBRATION.sun);
  mx.setCalibration({ sun: 0.5 });
  assert.equal(sun.intensity, 1.5);
  mx.setCalibration({ environment: 2 });
  assert.equal(sun.intensity, 1.5);
  assert.deepEqual(CALIBRATION, { environment: 1.45, sun: 1.0 });
});

test("shadows with auto_fit: every mesh casts and receives, the shadow camera covers the bounding sphere", async () => {
  const shadows = { target: [0, 0, 0], extent: 10, resolution: 8192, near: 0.1, far: 50, bias: 0.005, pcf_radius: 2, auto_fit: true };
  const renderer = stubRenderer();
  const gltf = litGltf({ shadows });
  const before = sunOf(gltf.scene);
  gltf.scene.updateMatrixWorld(true);
  const dir = new THREE.Vector3().setFromMatrixPosition(before.target.matrixWorld)
    .sub(new THREE.Vector3().setFromMatrixPosition(before.matrixWorld)).normalize();
  const mx = await buildScene(THREE, gltf, renderer);
  assert.equal(renderer.shadowMap.enabled, true);
  const meshes = [];
  mx.scene.traverse((o) => { if (o.isMesh) meshes.push(o); });
  assert.equal(meshes.length, 2);
  for (const m of meshes) assert.deepEqual([m.castShadow, m.receiveShadow], [true, true]);
  const sun = sunOf(mx.scene);
  assert.equal(sun.castShadow, true);
  assert.deepEqual([sun.shadow.mapSize.x, sun.shadow.mapSize.y], [4096, 4096]);
  assert.deepEqual([sun.shadow.bias, sun.shadow.normalBias, sun.shadow.radius], [-0.0005, 0.05, 2]);
  // The desktop's fit: the scene's bounding sphere, padded by max(5%, 0.1).
  const sphere = new THREE.Box3().setFromObject(gltf.scene).getBoundingSphere(new THREE.Sphere());
  const pad = Math.max(sphere.radius * 0.05, 0.1), r = sphere.radius + pad;
  const cam = sun.shadow.camera;
  assert.deepEqual([cam.left, cam.right, cam.top, cam.bottom, cam.near], [-r, r, r, -r, 0.01]);
  assert.ok(Math.abs(cam.far - (2 * r + pad)) < 1e-9);
  mx.scene.updateMatrixWorld(true);
  const from = new THREE.Vector3().setFromMatrixPosition(sun.matrixWorld);
  const to = new THREE.Vector3().setFromMatrixPosition(sun.target.matrixWorld);
  assert.ok(to.distanceTo(sphere.center) < 1e-9);
  assert.ok(Math.abs(from.distanceTo(to) - (r + pad)) < 1e-9);
  assert.ok(to.clone().sub(from).normalize().distanceTo(dir) < 1e-9, "the sun keeps its direction");
});

test("shadows without auto_fit use target, extent, near and far", async () => {
  const shadows = { target: [1, 0, 2], extent: 12, resolution: 1024, near: 0.5, far: 40, bias: 0.005, pcf_radius: 1, auto_fit: false };
  const mx = await buildScene(THREE, litGltf({ shadows }), stubRenderer());
  const sun = sunOf(mx.scene);
  const cam = sun.shadow.camera;
  assert.deepEqual([cam.left, cam.right, cam.top, cam.bottom, cam.near, cam.far], [-12, 12, 12, -12, 0.5, 40]);
  assert.deepEqual([sun.shadow.mapSize.x, sun.shadow.radius], [1024, 1]);
  mx.scene.updateMatrixWorld(true);
  const from = new THREE.Vector3().setFromMatrixPosition(sun.matrixWorld);
  const to = new THREE.Vector3().setFromMatrixPosition(sun.target.matrixWorld);
  assert.deepEqual(to.toArray(), [1, 0, 2]);
  assert.ok(Math.abs(from.distanceTo(to) - 20) < 1e-9, "back by far / 2, as the desktop");
});

test("without shadows nothing casts", async () => {
  const mx = await buildScene(THREE, litGltf({}), stubRenderer());
  mx.scene.traverse((o) => { if (o.isMesh || o.isLight) assert.equal(o.castShadow, false); });
});

test("the environment: the float32 rows, zenith first, to a half-float texture with rows bottom-up", () => {
  // Two rows of two texels: the zenith row (1, 2) then the nadir row (3, 4).
  const rgb = new Float32Array([1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4]);
  const tex = equirectTexture(THREE, rgb, 2, 2);
  assert.ok(tex.isDataTexture);
  assert.deepEqual([tex.image.width, tex.image.height, tex.type, tex.format], [2, 2, THREE.HalfFloatType, THREE.RGBAFormat]);
  assert.equal(tex.mapping, THREE.EquirectangularReflectionMapping);
  assert.equal(tex.colorSpace, THREE.LinearSRGBColorSpace);
  const texels = Array.from(tex.image.data, (h) => THREE.DataUtils.fromHalfFloat(h));
  assert.deepEqual(texels, [3, 3, 3, 1, 4, 4, 4, 1, 1, 1, 1, 1, 2, 2, 2, 1]);
});

test("resetMaterials puts the glTF's material back on every mesh, as a rebuilt scene has it", { skip }, async () => {
  const { gltf, mx } = await webWalk();
  const nodes = marked(gltf.scene);
  const [box, { material }] = [...nodes].at(-1);
  const before = mx.entities.get(box).map((m) => m.material);
  const other = [...mx.materials.keys()].find((id) => id !== material);
  mx.setMaterial(box, other);
  assert.equal(mx.materialOf(box), other);
  mx.resetMaterials();
  assert.equal(mx.materialOf(box), material);
  assert.deepEqual(mx.entities.get(box).map((m) => m.material), before);
});

test("dispose frees every geometry, material and texture of the scene once", { skip }, async () => {
  const { mx } = await webWalk();
  const freed = new Map();
  mx.scene.traverse((o) => {
    for (const thing of [o.geometry, ...[o.material ?? []].flat()]) {
      if (thing && !freed.has(thing)) {
        freed.set(thing, 0);
        thing.dispose = () => freed.set(thing, freed.get(thing) + 1);
      }
    }
  });
  for (const m of mx.materials.values()) {
    if (!freed.has(m)) {
      freed.set(m, 0);
      m.dispose = () => freed.set(m, freed.get(m) + 1);
    }
  }
  mx.dispose();
  assert.ok(freed.size > 2);
  assert.deepEqual([...new Set(freed.values())], [1]);
});
