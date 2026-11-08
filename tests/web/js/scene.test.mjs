// The exported scene in three.js, in Node: the vendored GLTFLoader parses web_walk's
// scene.glb (untextured, so Node's globals suffice) and buildScene runs on a stub renderer.

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { register } from "node:module";
import test from "node:test";
import { pathToFileURL } from "node:url";

import { createEngine } from "../../../src/manifoldx/web/js/manifoldx.js";
import { load } from "../../../src/manifoldx/web/js/replay.mjs";

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

// The folders export --web wrote for examples/web_walk.py, tests/web/apps/spawner.py and
// tests/web/apps/skinned.py, set by tests/web/test_node_suites.py.
const WEB_WALK = process.env.MX_WEB_WALK_EXPORT;
const SPAWNER = process.env.MX_SPAWNER_EXPORT;
const SKINNED = process.env.MX_SKINNED_EXPORT;
const skip = !WEB_WALK && "MX_WEB_WALK_EXPORT not set";
const skipSpawner = !SPAWNER && "MX_SPAWNER_EXPORT not set";
const skipSkinned = !SKINNED && "MX_SKINNED_EXPORT not set";

const stubRenderer = () => ({ capabilities: { maxTextureSize: 4096 }, shadowMap: { enabled: false } });

async function parse(folder) {
  const buf = readFileSync(`${folder}/scene.glb`);
  return new GLTFLoader().parseAsync(buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength), "./");
}

async function webWalk() {
  const gltf = await parse(WEB_WALK);
  const state = JSON.parse(readFileSync(`${WEB_WALK}/state.json`));
  return { gltf, state, mx: await buildScene(THREE, gltf, stubRenderer()) };
}

// An export's scene and its app, loaded as the replay loads it, with the store.
async function built(folder = WEB_WALK) {
  const gltf = await parse(folder);
  const app = await load(pathToFileURL(`${folder}/`));
  return { gltf, app, scene: await buildScene(THREE, gltf, stubRenderer()) };
}

// The batch whose instances draw slot `index`, by the instance matrix it carries.
const batchKey = (store, index) => `${store.column("Mesh")[index]}:${store.column("Material")[index]}`;

// The material ids of mx_materials no alive entity draws: web_walk's gold.
function unused(gltf, store) {
  const ids = Object.keys(gltf.scene.userData.manifoldx.mx_materials).map(Number);
  const drawn = new Set(Array.from(store.query(["Material"]), (i) => store.column("Material")[i]));
  return ids.filter((id) => !drawn.has(id));
}

test("CALIBRATION holds the factors the visual parity test measured", () => {
  assert.deepEqual(CALIBRATION, { environment: 1.45, sun: 1.0 });
});

test("geometries holds every id of mx_geometries as a BufferGeometry, and the glb has no entity nodes", { skip }, async () => {
  const { gltf, mx } = await webWalk();
  const ids = Object.keys(gltf.scene.userData.manifoldx.mx_geometries).map(Number);
  assert.equal(ids.length, 1, "web_walk draws one cube geometry");
  assert.deepEqual([...mx.geometries.keys()], ids);
  for (const g of mx.geometries.values()) {
    assert.ok(g.isBufferGeometry);
    assert.ok(g.getAttribute("position").count > 0);
  }
  let marked = 0;
  gltf.scene.traverse((o) => { if (o.userData.mx) marked++; });
  assert.equal(marked, 0);
  assert.equal(mx.entities, undefined);
  assert.equal(mx.setMaterial, undefined);
  assert.equal(mx.resetMaterials, undefined);
});

test("materials holds every id of mx_materials, gold included; materialOf reads the store's column", { skip }, async () => {
  const { gltf, app, scene } = await built();
  const ids = Object.keys(gltf.scene.userData.manifoldx.mx_materials).map(Number);
  assert.deepEqual([...scene.materials.keys()].sort((a, b) => a - b), ids.sort((a, b) => a - b));
  const { store } = app.engine;
  assert.equal(unused(gltf, store).length, 1, "web_walk has a material no entity uses at export");
  scene.sync(app.engine);
  for (const i of store.query(["Material"])) assert.equal(scene.materialOf(i), store.column("Material")[i]);
  assert.equal(scene.materialOf(9999), null);
});

test("one InstancedMesh per (geometry, material) with a count per member", { skip }, async () => {
  const { scene, app } = await built();
  scene.sync(app.engine);
  const counts = [...scene.batches.values()].map((b) => b.count).sort();
  assert.deepEqual(counts, [1, 1, 1, 1, 1, 1, 1]); // the ground and six boxes, each its own material
  assert.equal(scene.batches.size, 7);
  const { store } = app.engine;
  for (const i of store.query(["Mesh"])) {
    const b = scene.batches.get(batchKey(store, i));
    assert.ok(b.isInstancedMesh);
    assert.equal(b.material, scene.materials.get(store.column("Material")[i]));
    assert.equal(b.geometry, scene.geometries.get(store.column("Mesh")[i]));
    assert.equal(b.visible, true);
  }
});

test("set_material moves an instance between batches; an emptied batch draws nothing", { skip }, async () => {
  const { gltf, scene, app } = await built();
  const { engine } = app, { store } = engine;
  scene.sync(engine);
  const box = store.query(["Mesh"]).at(-1);
  const old = batchKey(store, box);
  const [gold] = unused(gltf, store);
  engine.set_material(engine.resolve("entity", box), engine.resolve("material", gold));
  scene.sync(engine);
  const goldBatch = scene.batches.get(`${store.column("Mesh")[box]}:${gold}`);
  assert.equal(goldBatch.count, 1);
  assert.equal(goldBatch.visible, true);
  assert.equal(goldBatch.material, scene.materials.get(gold));
  assert.equal(scene.batches.get(old).count, 0);
  assert.equal(scene.batches.get(old).visible, false);
  assert.equal(scene.materialOf(box), gold);
  // A later member reuses the emptied batch, at its own matrix.
  const [oldMid] = old.split(":").slice(1).map(Number);
  const other = store.query(["Mesh"])[0];
  engine.set_material(engine.resolve("entity", other), engine.resolve("material", oldMid));
  scene.sync(engine);
  const reused = scene.batches.get(old);
  assert.equal(reused.count, 1);
  const t = store.column("Transform").subarray(other * 10, other * 10 + 10);
  const want = new THREE.Matrix4().compose(
    new THREE.Vector3(t[0], t[1], t[2]), new THREE.Quaternion(t[3], t[4], t[5], t[6]), new THREE.Vector3(t[7], t[8], t[9]));
  assert.deepEqual(reused.getMatrixAt(0, new THREE.Matrix4()).elements, want.elements);
});

test("instance matrices are the desktop's TRS of the Transform row", { skip }, async () => {
  const { scene, app } = await built();
  const { store } = app.engine;
  const box = store.query(["Mesh"]).at(-1);
  const q = new THREE.Quaternion().setFromEuler(new THREE.Euler(0.3, -0.7, 1.1));
  store.write("Transform", box, 0, [1, 2, 3, q.x, q.y, q.z, q.w, 2, 1, 0.5]);
  scene.sync(app.engine);
  const b = scene.batches.get(batchKey(store, box));
  const t = store.column("Transform").subarray(box * 10, box * 10 + 10); // as float32 rounded them
  const want = new THREE.Matrix4().compose(
    new THREE.Vector3(t[0], t[1], t[2]), new THREE.Quaternion(t[3], t[4], t[5], t[6]), new THREE.Vector3(t[7], t[8], t[9]));
  const got = b.getMatrixAt(0, new THREE.Matrix4()).elements;
  for (let k = 0; k < 16; k++) assert.equal(got[k], Math.fround(want.elements[k]), `element ${k}`);
  // The desktop's rows (renderer._recompute_matrices): R scaled per column, the translation last.
  const [x, y, z, w] = [t[3], t[4], t[5], t[6]];
  assert.ok(Math.abs(got[0] - (1 - 2 * (y * y + z * z)) * t[7]) < 1e-6);
  assert.ok(Math.abs(got[4] - 2 * (x * y - w * z) * t[8]) < 1e-6);
  assert.deepEqual(got.slice(12, 15), [1, 2, 3]);
});

test("only dirty rows are recomputed, and a moved batch's bounding sphere follows", { skip }, async () => {
  const { scene, app } = await built();
  const { store } = app.engine;
  scene.sync(app.engine);
  const [first, box] = [store.query(["Mesh"])[0], store.query(["Mesh"]).at(-1)];
  const moved = scene.batches.get(batchKey(store, box)), still = scene.batches.get(batchKey(store, first));
  const centre = moved.boundingSphere.center.clone();
  const [v0, v1] = [moved.instanceMatrix.version, still.instanceMatrix.version];
  const version = store.version;
  const t = store.column("Transform");
  store.write("Transform", box, 0, [t[box * 10] + 5, t[box * 10 + 1], t[box * 10 + 2]]);
  assert.equal(store.version, version, "a Transform write is not a membership change");
  scene.sync(app.engine);
  assert.ok(moved.instanceMatrix.version > v0);
  assert.equal(still.instanceMatrix.version, v1);
  assert.ok(Math.abs(moved.boundingSphere.center.x - (centre.x + 5)) < 1e-4, `${moved.boundingSphere.center.x} vs ${centre.x}`);
  assert.equal(store.takeDirty(), null, "the scene took the dirty range");
});

test("a batch grows past its capacity and keeps every instance", { skip }, async () => {
  const { gltf, scene, app } = await built();
  const { engine } = app, { store } = engine;
  scene.sync(engine);
  const gid = store.column("Mesh")[store.query(["Mesh"])[0]];
  const [gold] = unused(gltf, store);
  const pos = new Float32Array(30 * 10);
  for (let r = 0; r < 30; r++) pos.set([r, 0, 0, 0, 0, 0, 1, 1, 1, 1], r * 10);
  store.spawn(30, new Map([["Mesh", Uint32Array.of(gid)], ["Material", Uint32Array.of(gold)], ["Transform", pos]]));
  scene.sync(engine);
  const b = scene.batches.get(`${gid}:${gold}`);
  assert.equal(b.count, 30);
  assert.ok(b.instanceMatrix.count >= 30);
  const xs = Array.from({ length: 30 }, (_, k) => b.getMatrixAt(k, new THREE.Matrix4()).elements[12]).sort((a, c) => a - c);
  assert.deepEqual(xs, Array.from({ length: 30 }, (_, k) => k));
});

test("a destroyed entity leaves its batch at the next sync", { skip }, async () => {
  const { scene, app } = await built();
  const { store } = app.engine;
  scene.sync(app.engine);
  const box = store.query(["Mesh"]).at(-1);
  const key = batchKey(store, box);
  store.destroy([box]);
  scene.sync(app.engine);
  assert.equal(scene.batches.get(key).count, 0);
  assert.equal(scene.batches.get(key).visible, false);
  assert.equal(scene.materialOf(box), null);
});

test("a spawn of a library geometry no entity used at export draws", { skip: skipSpawner }, async () => {
  const { gltf, scene, app } = await built(SPAWNER);
  const { store } = app.engine;
  scene.sync(app.engine);
  const used = new Set(Array.from(store.query(["Mesh"]), (i) => store.column("Mesh")[i]));
  const ball = Object.keys(gltf.scene.userData.manifoldx.mx_geometries).map(Number).find((g) => !used.has(g));
  assert.ok(ball !== undefined, "spawner.py holds a geometry no entity draws at export");
  const mid = Number(Object.keys(gltf.scene.userData.manifoldx.mx_materials)[0]);
  const t = Float32Array.of(0, 2, 0, 0, 0, 0, 1, 1, 1, 1);
  const [i] = store.spawn(1, new Map([["Mesh", Uint32Array.of(ball)], ["Material", Uint32Array.of(mid)], ["Transform", t]]));
  scene.sync(app.engine);
  const b = scene.batches.get(`${ball}:${mid}`);
  assert.equal(b.count, 1);
  assert.equal(b.geometry, scene.geometries.get(ball));
  assert.ok(b.parent !== null, "the batch is in the scene");
  assert.equal(scene.materialOf(i), mid);
});

test("a mesh without a Material draws with a white basic material", { skip }, async () => {
  const { scene, app } = await built();
  const { store } = app.engine;
  const gid = store.column("Mesh")[store.query(["Mesh"])[0]];
  store.spawn(1, new Map([["Mesh", Uint32Array.of(gid)], ["Transform", Float32Array.of(0, 0, 0, 0, 0, 0, 1, 1, 1, 1)]]));
  scene.sync(app.engine);
  const b = scene.batches.get(`${gid}:0`);
  assert.equal(b.count, 1);
  assert.equal(b.material.userData.mxBasic, true);
  assert.deepEqual(b.material.color.toArray(), [1, 1, 1]);
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
  // An instance's matrix comes before the model matrix, as three.js's own chunk does.
  assert.match(
    shader.vertexShader,
    /#ifdef USE_INSTANCING\s+mat3 mxNormalMatrix = mat3\( modelMatrix \) \* mat3\( instanceMatrix \);/,
  );
  assert.match(shader.vertexShader, /vMxNormal = normalize\( mxNormalMatrix \* normal \);/);
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

// An engine with no store, for the stand-in scenes: sync places the camera and fits the shadow.
const bare = () => createEngine({ w: 320, h: 200, camera: { position: [0, 2, 8], target: [0, 0, 0], fov: 60, up: [0, 1, 0], near: 0.1, far: 100 } });

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
  assert.equal(renderer.shadowMap.enabled, false, "the fit waits for the first sync, so auto_fit sees the batches");
  mx.sync(bare());
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

test("with shadows every batch casts and receives, and auto_fit covers the batches", { skip }, async () => {
  const gltf = await parse(WEB_WALK);
  const app = await load(pathToFileURL(`${WEB_WALK}/`));
  const sun = new THREE.DirectionalLight(0xffffff, 3);
  sun.position.set(1, 4, 2);
  gltf.scene.add(sun, sun.target);
  gltf.scene.userData.manifoldx.shadows = { target: [0, 0, 0], extent: 10, resolution: 1024, near: 0.1, far: 50, bias: 0.005, pcf_radius: 1, auto_fit: true };
  const mx = await buildScene(THREE, gltf, stubRenderer());
  const { store } = app.engine;
  const far = store.query(["Mesh"]).at(-1);
  store.write("Transform", far, 0, [80, 0, 0]);
  mx.sync(app.engine);
  assert.ok(mx.batches.size > 0);
  for (const b of mx.batches.values()) assert.deepEqual([b.castShadow, b.receiveShadow], [true, true]);
  const cam = sun.shadow.camera;
  assert.ok(cam.right > 40, `the fit reaches the box at x = 80: extent ${cam.right}`);
});

test("shadows without auto_fit use target, extent, near and far", async () => {
  const shadows = { target: [1, 0, 2], extent: 12, resolution: 1024, near: 0.5, far: 40, bias: 0.005, pcf_radius: 1, auto_fit: false };
  const mx = await buildScene(THREE, litGltf({ shadows }), stubRenderer());
  mx.sync(bare());
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
  mx.sync(bare());
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

test("a new store, a swapped version's, rebuilds the batches from its own rows", { skip }, async () => {
  const { gltf, scene, app } = await built();
  const { engine } = app, { store } = engine;
  scene.sync(engine);
  const box = store.query(["Mesh"]).at(-1);
  const [gold] = unused(gltf, store);
  engine.set_material(engine.resolve("entity", box), engine.resolve("material", gold));
  scene.sync(engine);
  // The kept scene under run --web meets the next version's app, whose store starts at
  // version 0 again with the export's materials.
  const next = await load(pathToFileURL(`${WEB_WALK}/`));
  assert.equal(next.engine.store.version, 0);
  scene.sync(next.engine);
  const goldKey = `${next.engine.store.column("Mesh")[box]}:${gold}`;
  assert.equal(scene.batches.get(goldKey).count, 0);
  assert.equal(scene.materialOf(box), next.engine.store.column("Material")[box]);
  assert.equal(scene.batches.get(batchKey(next.engine.store, box)).count, 1);
});

test("dispose frees every geometry, material and texture of the scene once", { skip }, async () => {
  const { scene: mx, app } = await built();
  mx.sync(app.engine);
  const freed = new Map();
  mx.scene.traverse((o) => {
    for (const thing of [o.geometry, ...[o.material ?? []].flat()]) {
      if (thing && !freed.has(thing)) {
        freed.set(thing, 0);
        thing.dispose = () => freed.set(thing, freed.get(thing) + 1);
      }
    }
  });
  for (const m of [...mx.materials.values(), ...mx.geometries.values()]) {
    if (!freed.has(m)) {
      freed.set(m, 0);
      m.dispose = () => freed.set(m, freed.get(m) + 1);
    }
  }
  mx.dispose();
  assert.ok(freed.size > 2);
  assert.deepEqual([...new Set(freed.values())], [1]);
});

// ---------------------------------------------------------------- skinned entities

// skinned.py: figures in slots 0 to 2 (figure 0 posed at setup), the box in slot 3. Frame 0
// poses nothing; frame 1 poses the crowd.
const IDLE = { keys: [], just: [], d: [0, 0], buttons: [], clicks: [], vp: [320, 200] };
const FIGURES = [0, 1, 2], BOX = 3;

// Joint m of a row-major pose as a three.js Matrix4.
const poseMatrix = (pose, m) => new THREE.Matrix4().set(...pose.subarray(m * 16, m * 16 + 16));

// A bone matrix (column-major) equals the row-major pose matrix, transposed.
function assertBones(mesh, pose, j) {
  const bm = mesh.skeleton.boneMatrices;
  for (let m = 0; m < j; m++) {
    for (let r = 0; r < 4; r++) {
      for (let c = 0; c < 4; c++) {
        const want = pose === undefined ? (r === c ? 1 : 0) : pose[m * 16 + r * 4 + c];
        assert.equal(bm[m * 16 + c * 4 + r], want, `joint ${m} row ${r} col ${c}`);
      }
    }
  }
}

// The app runs on its folder's own runtime (replay.mjs, manifoldx.js), whose NdArray its
// modules build: the source's would not take them for arrays.
async function skinnedScene(frames = 0) {
  const folder = pathToFileURL(`${SKINNED}/`);
  const { load: loadFolder } = await import(new URL("replay.mjs", folder).href);
  const { advance } = await import(new URL("manifoldx.js", folder).href);
  const gltf = await parse(SKINNED);
  const app = await loadFolder(folder);
  const scene = await buildScene(THREE, gltf, stubRenderer());
  const step = () => advance(app, IDLE, 1 / 60);
  for (let f = 0; f < frames; f++) step();
  scene.sync(app.engine);
  return { gltf, app, scene, step };
}

test("a skinned entity draws as its own SkinnedMesh, not in a batch", { skip: skipSkinned }, async () => {
  const { scene, app } = await skinnedScene();
  const { store } = app.engine;
  assert.equal(scene.skinned.size, 3);
  assert.deepEqual([...scene.skinned.keys()].sort(), FIGURES);
  const fig = store.column("Mesh")[0];
  for (const i of FIGURES) {
    const mesh = scene.skinned.get(i);
    assert.ok(mesh.isSkinnedMesh);
    assert.equal(mesh.geometry, scene.geometries.get(fig));
    assert.ok(mesh.parent !== null, "the mesh is in the scene");
    assert.equal(mesh.skeleton.bones.length, store.skinned.get(fig));
    assert.ok(mesh.skeleton.bones.every((b) => b.parent === null && !b.matrixAutoUpdate));
  }
  assert.equal(scene.batches.get(batchKey(store, BOX)).count, 1);
  const drawn = [...scene.batches.values()].reduce((n, b) => n + b.count, 0);
  assert.equal(drawn, 1, "only the box is batched");
  for (const b of scene.batches.values()) assert.notEqual(b.geometry, scene.geometries.get(fig));
});

test("bone matrices are the store's pose", { skip: skipSkinned }, async () => {
  const { scene, app, step } = await skinnedScene(2);
  const { store } = app.engine;
  for (const i of FIGURES) {
    assert.ok(store.joints.has(i), `figure ${i} is posed by frame 1`);
    assertBones(scene.skinned.get(i), store.joints.get(i), 3);
  }
  const before = Float32Array.from(scene.skinned.get(0).skeleton.boneMatrices);
  step();
  scene.sync(app.engine);
  assert.notDeepEqual(Array.from(scene.skinned.get(0).skeleton.boneMatrices), Array.from(before));
  assertBones(scene.skinned.get(0), store.joints.get(0), 3);
});

test("a vertex lands at TRS times its joint's matrix", { skip: skipSkinned }, async () => {
  const { scene, app } = await skinnedScene(2);
  const { store } = app.engine;
  const q = new THREE.Quaternion().setFromEuler(new THREE.Euler(0.3, -0.7, 1.1));
  store.write("Transform", 0, 0, [1, 2, 3, q.x, q.y, q.z, q.w, 2, 1, 0.5]);
  scene.sync(app.engine);
  scene.scene.updateMatrixWorld(true);
  const mesh = scene.skinned.get(0), pose = store.joints.get(0);
  const t = store.column("Transform").subarray(0, 10);
  const trs = new THREE.Matrix4().compose(
    new THREE.Vector3(t[0], t[1], t[2]), new THREE.Quaternion(t[3], t[4], t[5], t[6]), new THREE.Vector3(t[7], t[8], t[9]));
  const skinIndex = mesh.geometry.getAttribute("skinIndex"), position = mesh.geometry.getAttribute("position");
  const seen = new Set();
  for (let i = 0; i < position.count; i++) {
    const j = skinIndex.getX(i);
    seen.add(j);
    const want = new THREE.Vector3().fromBufferAttribute(position, i).applyMatrix4(poseMatrix(pose, j)).applyMatrix4(trs);
    const got = mesh.getVertexPosition(i, new THREE.Vector3()).applyMatrix4(mesh.matrixWorld);
    assert.ok(got.distanceTo(want) < 1e-5, `vertex ${i}: ${got.toArray()} vs ${want.toArray()}`);
  }
  assert.deepEqual([...seen].sort(), [0, 1, 2]);
});

test("a pose set at setup is drawn at the first sync", { skip: skipSkinned }, async () => {
  const { scene, app } = await skinnedScene(0);
  const { store } = app.engine;
  assert.deepEqual([...store.joints.keys()], [0], "only figure 0 is posed at export");
  const pose = store.joints.get(0);
  assert.ok(pose.some((x, k) => x !== [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1][k % 16]), "not the identity");
  assertBones(scene.skinned.get(0), pose, 3);
  assertBones(scene.skinned.get(1), undefined, 3);
});

test("a destroyed figure's mesh leaves; its slot respawned draws at rest", { skip: skipSkinned }, async () => {
  const { scene, app } = await skinnedScene(2);
  const { store } = app.engine;
  const old = scene.skinned.get(2);
  assert.ok(store.joints.has(2));
  store.destroy([2]);
  scene.sync(app.engine);
  assert.equal(scene.skinned.has(2), false);
  assert.equal(old.parent, null);
  const t = Float32Array.of(0, 0, 0, 0, 0, 0, 1, 1, 1, 1);
  const rows = new Map([["Mesh", Uint32Array.of(store.column("Mesh")[0])], ["Material", Uint32Array.of(store.column("Material")[0])], ["Transform", t]]);
  const [i] = store.spawn(1, rows);
  assert.equal(i, 2);
  scene.sync(app.engine);
  assertBones(scene.skinned.get(2), undefined, 3);
  // Posed, then destroyed and respawned between two syncs: at rest too.
  store.setJoints([2], store.joints.get(0) ?? new Float32Array(48), 3);
  scene.sync(app.engine);
  store.destroy([2]);
  store.spawn(1, rows);
  scene.sync(app.engine);
  assertBones(scene.skinned.get(2), undefined, 3);
});

test("set_material on a skinned entity swaps its mesh's material", { skip: skipSkinned }, async () => {
  const { scene, app } = await skinnedScene(1);
  const { engine } = app, { store } = engine;
  const grey = store.column("Material")[BOX], skin = store.column("Material")[0];
  assert.notEqual(grey, skin);
  assert.equal(scene.skinned.get(0).material, scene.materials.get(skin));
  const mesh = scene.skinned.get(0);
  engine.set_material(engine.resolve("entity", 0), engine.resolve("material", grey));
  scene.sync(engine);
  assert.equal(scene.skinned.get(0), mesh, "the same mesh, its material swapped");
  assert.equal(mesh.material, scene.materials.get(grey));
  assert.equal(scene.materialOf(0), grey);
  assert.equal(scene.batches.get(batchKey(store, BOX)).count, 1, "no batch gained the figure");
});

test("the bounding sphere is the desktop's padded sphere", { skip: skipSkinned }, async () => {
  const { scene } = await skinnedScene(0);
  const mesh = scene.skinned.get(0), position = mesh.geometry.getAttribute("position");
  const lo = [Infinity, Infinity, Infinity], hi = [-Infinity, -Infinity, -Infinity];
  for (let i = 0; i < position.count; i++) {
    for (let a = 0; a < 3; a++) {
      lo[a] = Math.min(lo[a], position.getComponent(i, a));
      hi[a] = Math.max(hi[a], position.getComponent(i, a));
    }
  }
  const c = lo.map((x, a) => (x + hi[a]) / 2);
  let r = 0;
  for (let i = 0; i < position.count; i++) {
    r = Math.max(r, Math.hypot(...[0, 1, 2].map((a) => position.getComponent(i, a) - c[a])));
  }
  assert.deepEqual(mesh.boundingSphere.center.toArray(), c);
  assert.ok(Math.abs(mesh.boundingSphere.radius - (r * 1.5 + 0.5)) < 1e-12);
});
