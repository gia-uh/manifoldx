// A skinned entity of Engine.export_gltf in three.js, in Node: the vendored GLTFLoader must put
// each vertex where the desktop draws it, TRS . P[joint] . (x, y, z, 1).

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { register } from "node:module";
import { dirname } from "node:path";
import test from "node:test";

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

// The glb tests/web/test_node_suites.py writes, expected.json beside it.
const GLB = process.env.MX_SKIN_GLB;
const skip = !GLB && "MX_SKIN_GLB not set";

test("GLTFLoader puts each vertex where the desktop draws it", { skip }, async () => {
  const buf = readFileSync(GLB);
  const gltf = await new GLTFLoader().parseAsync(buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength), "./");
  const expected = JSON.parse(readFileSync(`${dirname(GLB)}/expected.json`));
  gltf.scene.updateMatrixWorld(true);
  const meshes = [];
  gltf.scene.traverse((o) => o.isSkinnedMesh && meshes.push(o));
  assert.equal(meshes.length, 1);
  const [mesh] = meshes;
  mesh.skeleton.update();
  assert.equal(mesh.geometry.attributes.position.count, expected.length);
  const v = new THREE.Vector3();
  for (let i = 0; i < expected.length; i++) {
    mesh.getVertexPosition(i, v);
    v.applyMatrix4(mesh.matrixWorld);
    const d = Math.hypot(v.x - expected[i][0], v.y - expected[i][1], v.z - expected[i][2]);
    assert.ok(d < 1e-5, `vertex ${i}: [${v.toArray()}] is not [${expected[i]}]`);
  }
});
