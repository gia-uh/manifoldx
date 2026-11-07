// The exported scene in three.js: the glTF's manifoldx extras (background, fog, environment,
// sun, shadows) applied once with manifoldx's calibration, the desktop's fixed light for
// BasicMaterial and PhongMaterial, the glb's library of geometries and materials by id, and one
// InstancedMesh per (geometry, material) drawing the entity store. THREE is passed in, so this
// module imports nothing.

// three.js against the desktop at the same intensities, measured by tests/web/browser/test_visual.py
// on tests/web/apps/lit_yard.py (sky environment, sun, shadows, fog) at 320x200, lavapipe for the
// desktop and SwiftShader for Chromium. The desktop's mean colour is (131.7, 141.3, 157.9); the
// page gives (126.5, 133.6, 146.4) uncalibrated (4.0, 5.5, 7.3% darker), (132.3, 142.0, 159.7)
// at 1.5, the first consumer's factor (worst channel 1.1%), and (131.7, 141.2, 158.4) at 1.45,
// the best of a sweep from 1.30 to 1.70 in steps of 0.05 (worst channel 0.3%; each step moves
// the means about 0.5%). The sun needs none.
export const CALIBRATION = { environment: 1.45, sun: 1.0 };

// The desktop lights BasicMaterial and PhongMaterial with a fixed light from (1, 1, 1) in world
// space: 0.3 ambient + 0.7 Lambert (resources.py, the basic shader).
const BASIC_VERTEX = "\tvMxNormal = normalize( mat3( modelMatrix ) * normal );";
const BASIC_FRAGMENT = "\tdiffuseColor.rgb *= 0.3 + 0.7 * max( dot( normalize( vMxNormal ), normalize( vec3( 1.0 ) ) ), 0.0 );";

function mxBasic(THREE, base) {
  const m = new THREE.MeshBasicMaterial({
    name: base.name,
    color: base.color.clone(),
    transparent: base.transparent,
    opacity: base.opacity,
    side: base.side,
    vertexColors: base.vertexColors,
    fog: false, // the desktop fogs only StandardMaterial
  });
  m.userData = { ...base.userData, mxBasic: true };
  m.onBeforeCompile = (shader) => {
    shader.vertexShader = shader.vertexShader
      .replace("void main() {", "varying vec3 vMxNormal;\nvoid main() {")
      .replace("#include <project_vertex>", `#include <project_vertex>\n${BASIC_VERTEX}`);
    shader.fragmentShader = shader.fragmentShader
      .replace("void main() {", "varying vec3 vMxNormal;\nvoid main() {")
      .replace("#include <color_fragment>", `#include <color_fragment>\n${BASIC_FRAGMENT}`);
  };
  return m;
}

// A glTF material as the desktop draws it, by the material class name the writer gave it.
function convert(THREE, base) {
  if (base.name === "BasicMaterial" || base.name === "PhongMaterial") return mxBasic(THREE, base);
  if (base.name !== "StandardMaterial") base.fog = false;
  return base;
}

// The exported sky (linear RGB float32, row 0 at the zenith) as a half-float equirectangular
// texture, whose rows run bottom-up.
export function equirectTexture(THREE, rgb, w, h) {
  const data = new Uint16Array(w * h * 4);
  const one = THREE.DataUtils.toHalfFloat(1);
  for (let r = 0; r < h; r++) {
    const dst = (h - 1 - r) * w;
    for (let c = 0; c < w; c++) {
      const i = (r * w + c) * 3, o = (dst + c) * 4;
      data[o] = THREE.DataUtils.toHalfFloat(rgb[i]);
      data[o + 1] = THREE.DataUtils.toHalfFloat(rgb[i + 1]);
      data[o + 2] = THREE.DataUtils.toHalfFloat(rgb[i + 2]);
      data[o + 3] = one;
    }
  }
  const tex = new THREE.DataTexture(data, w, h, THREE.RGBAFormat, THREE.HalfFloatType);
  tex.mapping = THREE.EquirectangularReflectionMapping;
  tex.colorSpace = THREE.LinearSRGBColorSpace;
  tex.needsUpdate = true;
  return tex;
}

// Point the sun's shadow camera as the desktop does (renderer._sun_light_view_proj): with
// auto_fit, at the scene's bounding sphere padded by max(5%, 0.1), once; otherwise at target,
// extent, near and far, from far / 2 back. The sun keeps its direction.
function fitShadow(THREE, scene, root, sun, cfg, renderer) {
  root.updateMatrixWorld(true);
  const from = new THREE.Vector3().setFromMatrixPosition(sun.matrixWorld);
  const dir = new THREE.Vector3().setFromMatrixPosition(sun.target.matrixWorld).sub(from).normalize();
  let center, extent, near, far, back;
  if (cfg.auto_fit) {
    const sphere = new THREE.Box3().setFromObject(root).getBoundingSphere(new THREE.Sphere());
    const pad = Math.max(sphere.radius * 0.05, 0.1);
    center = sphere.center;
    extent = sphere.radius + pad;
    [near, far, back] = [0.01, 2 * extent + pad, extent + pad];
  } else {
    center = new THREE.Vector3(...cfg.target);
    [extent, near, far, back] = [cfg.extent, cfg.near, cfg.far, cfg.far * 0.5];
  }
  const target = new THREE.Object3D();
  target.position.copy(center);
  scene.add(target);
  sun.remove(sun.target);
  sun.target = target;
  const at = center.clone().addScaledVector(dir, -back);
  sun.position.copy(sun.parent ? sun.parent.worldToLocal(at) : at);

  const size = Math.min(cfg.resolution, renderer.capabilities.maxTextureSize);
  sun.castShadow = true;
  sun.shadow.mapSize.set(size, size);
  Object.assign(sun.shadow.camera, { left: -extent, right: extent, top: extent, bottom: -extent, near, far });
  sun.shadow.camera.updateProjectionMatrix();
  sun.shadow.bias = -0.0005;
  sun.shadow.normalBias = 0.05;
  sun.shadow.radius = cfg.pcf_radius;
  renderer.shadowMap.enabled = true;
}

// The Transform row of slot i (pos, the quaternion x y z w, scale) as the desktop's model
// matrix, T R S (renderer._recompute_matrices), into instance k of batch b.
function writeInstance(b, k, rows, i, m, p, q, sc) {
  const o = i * 10;
  p.set(rows[o], rows[o + 1], rows[o + 2]);
  q.set(rows[o + 3], rows[o + 4], rows[o + 5], rows[o + 6]);
  sc.set(rows[o + 7], rows[o + 8], rows[o + 9]);
  b.setMatrixAt(k, m.compose(p, q, sc));
}

// The glTF the loader parsed, as an MxScene: .scene and .camera to render, .geometries
// (geometry id to its BufferGeometry, the glb's library), .materials (material id to its
// three.js material), .batches ("gid:mid" to the InstancedMesh drawing those entities),
// .materialOf(entity) (the store's Material column, or null), .sync(engine) each frame (the
// camera, then the batches from engine.store), .setCalibration(partial) over a copy of
// CALIBRATION, and .dispose() (frees its GPU resources). The renderer is used for its
// capabilities, the shadow map switch and, when the scene has an environment, PMREM.
export async function buildScene(THREE, gltf, renderer) {
  const extras = gltf.scene.userData.manifoldx ?? {};
  // gltf.scene is a Group; background, fog and environment only apply to a Scene.
  const scene = new THREE.Scene();
  scene.add(gltf.scene);
  if (extras.background) scene.background = new THREE.Color().setRGB(...extras.background);
  if (extras.fog) scene.fog = new THREE.Fog(new THREE.Color().setRGB(...extras.fog.color), extras.fog.start, extras.fog.end);

  const converted = new Map(); // loader material -> the one drawn
  const drawn = (base) => {
    if (!converted.has(base)) converted.set(base, convert(THREE, base));
    return converted.get(base);
  };
  const materials = new Map(), bases = new Map();
  for (const [id, index] of Object.entries(extras.mx_materials ?? {})) {
    const base = await gltf.parser.getDependency("material", index);
    bases.set(Number(id), base);
    materials.set(Number(id), drawn(base));
  }
  const geometries = new Map();
  for (const [id, index] of Object.entries(extras.mx_geometries ?? {})) {
    geometries.set(Number(id), (await gltf.parser.getDependency("mesh", index)).geometry);
  }
  // A mesh without a Material, which the desktop draws white with its basic light.
  const white = mxBasic(THREE, new THREE.MeshBasicMaterial({ name: "BasicMaterial", color: 0xffffff }));
  const coloured = new Map(); // material id -> its copy with vertex colours on

  // The material batch (gid, mid) draws: the library's, a copy with vertex colours when the
  // material asks for them and the geometry has them, or white for no Material.
  const batchMaterial = (gid, mid) => {
    if (mid === 0) return white;
    const m = materials.get(mid);
    if (m === undefined) throw new Error(`material ${mid} is not in the exported scene`);
    if (!(m.userData.mx?.vertex_colors && geometries.get(gid).hasAttribute("color"))) return m;
    if (!coloured.has(mid)) {
      const base = bases.get(mid).clone();
      base.vertexColors = true;
      coloured.set(mid, convert(THREE, base));
    }
    return coloured.get(mid);
  };

  if (extras.environment) {
    const env = extras.environment;
    const rgb = new Float32Array(await gltf.parser.getDependency("bufferView", env.bufferView));
    const tex = equirectTexture(THREE, rgb, env.width, env.height);
    const pmrem = new THREE.PMREMGenerator(renderer);
    scene.environment = pmrem.fromEquirectangular(tex).texture;
    tex.dispose();
    pmrem.dispose();
  }

  let sun = null;
  gltf.scene.traverse((o) => { if (!sun && o.isDirectionalLight) sun = o; });
  const sunIntensity = sun ? sun.intensity : 0;
  const shadows = Boolean(sun && extras.shadows);
  let fitted = false;

  const calibration = { ...CALIBRATION };
  const calibrate = () => {
    if (sun) sun.intensity = sunIntensity * calibration.sun;
    if (extras.environment) scene.environmentIntensity = extras.environment.intensity * calibration.environment;
  };
  calibrate();

  // The batches: one InstancedMesh per (geometry, material) under gltf.scene, its members the
  // store's slots in ascending order; slotOf maps a slot to [batch key, instance].
  const batches = new Map(), members = new Map();
  let slotOf = new Map(), seen = null, seenVersion = -1;
  const m4 = new THREE.Matrix4(), p = new THREE.Vector3(), q = new THREE.Quaternion(), sc = new THREE.Vector3();

  const batch = (gid, mid) => {
    const key = `${gid}:${mid}`;
    if (!batches.has(key)) {
      const b = new THREE.InstancedMesh(geometries.get(gid), batchMaterial(gid, mid), 1);
      b.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
      b.count = 0;
      b.castShadow = b.receiveShadow = shadows;
      b.name = key;
      gltf.scene.add(b);
      batches.set(key, b);
      members.set(key, []);
    }
    return key;
  };

  // A full pass when the store's membership moved (a spawn, a destroy, a Mesh or Material
  // write) or the store is another one (a new version's); otherwise the dirty Transform rows.
  const syncBatches = (store) => {
    const rows = store.column("Transform");
    const changed = new Set();
    if (store !== seen || store.version !== seenVersion) {
      store.takeDirty();
      seen = store;
      seenVersion = store.version;
      for (const list of members.values()) list.length = 0;
      slotOf = new Map();
      const mesh = store.member("Mesh"), gids = store.column("Mesh");
      const mat = store.member("Material"), mids = store.column("Material");
      for (let i = 0; i < store.highWater; i++) {
        if (!store.alive[i] || !mesh?.[i] || !geometries.has(gids[i])) continue;
        const key = batch(gids[i], mat?.[i] ? mids[i] : 0);
        const list = members.get(key);
        slotOf.set(i, [key, list.length]);
        list.push(i);
      }
      for (const [key, b] of batches) {
        const list = members.get(key);
        if (list.length > b.instanceMatrix.count) {
          let cap = b.instanceMatrix.count;
          while (cap < list.length) cap *= 2;
          b.instanceMatrix = new THREE.InstancedBufferAttribute(new Float32Array(cap * 16), 16);
          b.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
        }
        list.forEach((i, k) => writeInstance(b, k, rows, i, m4, p, q, sc));
        b.count = list.length;
        b.visible = list.length > 0;
        changed.add(b);
      }
    } else {
      const dirty = store.takeDirty();
      if (dirty !== null) {
        for (let i = dirty[0]; i < dirty[1]; i++) {
          const at = slotOf.get(i);
          if (at === undefined) continue;
          const b = batches.get(at[0]);
          writeInstance(b, at[1], rows, i, m4, p, q, sc);
          changed.add(b);
        }
      }
    }
    for (const b of changed) {
      b.instanceMatrix.needsUpdate = true;
      b.boundingBox = null;
      b.computeBoundingSphere();
    }
  };

  const camera = new THREE.PerspectiveCamera();
  return {
    scene,
    camera,
    geometries,
    materials,
    batches,
    materialOf(entity) {
      const store = seen;
      if (store === null || !(entity < store.capacity && store.alive[entity] && store.member("Material")?.[entity])) return null;
      return store.column("Material")[entity];
    },
    dispose() {
      const done = new Set();
      const free = (thing) => {
        if (!thing || done.has(thing)) return;
        done.add(thing);
        if (thing.isMaterial) for (const v of Object.values(thing)) if (v?.isTexture) free(v);
        thing.dispose();
      };
      scene.traverse((o) => {
        free(o.geometry);
        for (const m of [o.material ?? []].flat()) free(m);
      });
      for (const g of geometries.values()) free(g);
      for (const m of materials.values()) free(m);
      for (const m of converted.values()) free(m);
      for (const m of coloured.values()) free(m);
      free(white);
      for (const b of batches.values()) b.dispose(); // the instance buffers
      free(scene.environment);
    },
    sync(engine) {
      const c = engine.camera;
      camera.position.set(c.position[0], c.position[1], c.position[2]);
      camera.up.set(c.up[0], c.up[1], c.up[2]);
      camera.lookAt(c.target[0], c.target[1], c.target[2]);
      camera.fov = c.fov;
      camera.near = c.near;
      camera.far = c.far;
      camera.aspect = engine.w / engine.h;
      camera.updateProjectionMatrix();
      if (engine.store) syncBatches(engine.store);
      // The desktop fits the shadow once, at its first frame: after the batches, so auto_fit sees them.
      if (shadows && !fitted) {
        fitted = true;
        fitShadow(THREE, scene, gltf.scene, sun, extras.shadows, renderer);
        scene.traverse((o) => { if (o.isMesh) o.castShadow = o.receiveShadow = true; });
      }
    },
    setCalibration(partial) {
      Object.assign(calibration, partial);
      calibrate();
    },
  };
}
